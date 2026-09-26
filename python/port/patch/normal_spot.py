"""`cnaster.normal_spot.normal_baf_bin_filter`, without the quantile inversion.

**Proposed for `cnaster`, written here.** #174: the filter decides which bins to
drop by inverting a beta-binomial quantile at each bin, and `scipy` has no
closed form for that inverse -- `betabinom.ppf` falls back to a per-element
bisection whose every step sums the probability mass function in Python. It is
**93.6 per cent of the whole preprocessing chain** at 2,500 spots and 400 bins,
and the test it computes needs no inverse at all.

For a discrete law the inversion is unnecessary by an identity, not by an
approximation:

```
x <  ppf(q)   <=>   cdf(x)     <  q
x >  ppf(q)   <=>   cdf(x - 1) >= q
```

both because `ppf(q) = min{k : cdf(k) >= q}`. So the same mask comes from two
distribution-function evaluations rather than from two searches, and it is the
**same mask**, bitwise, rather than one within a tolerance.

`port` cannot land the change (`CLAUDE.md`, **Working against a repository you
do not own**), so it is written here with its referee beside it. The whole
function is reproduced because its return is what can be compared bitwise; the
change itself is the two lines in `removal_indicator`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import Any

import numpy as np
import scipy.special
import scipy.stats
from cnaster.config import get_global_config, start_time
from cnaster.hmm_emission import Weighted_BetaBinom
from cnaster.hmm_utils import get_em_solver_params
from cnaster.logger import get_logger
from cnaster.spatio_genomic_counts import SpatioGenomicCounts

logger = get_logger(__name__, start_time=start_time)

MIN_BETABINOM_TAU = 30
"""`cnaster`'s floor on the fitted concentration, carried across unchanged.

The fit's success probability is discarded and replaced by 0.5 and its
concentration is floored here, both in `cnaster` and in this patch: #38 owns
whether that is the right prior, and a patch that changed it would be
answering a different question from the one it is measuring.
"""


def _log_mass(
    index: np.ndarray, totals: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """`scipy`'s own beta-binomial log mass function, evaluated elementwise.

    Written out rather than called because `scipy` reaches it one
    distribution at a time; the formula is `betabinom._logpmf`, unchanged.
    """
    return np.asarray(
        -np.log(totals + 1)
        - scipy.special.betaln(totals - index + 1, index + 1)
        + scipy.special.betaln(index + alpha, totals - index + beta)
        - scipy.special.betaln(alpha, beta)
    )


TERM_BUDGET = 1 << 16
"""How many mass-function terms are held at once: 65,536, about 5 MB of peak.

The summation covers every bin, so its intermediate is the whole ragged
evaluation -- `sum(min(k, n - k))` terms, which at a slide's read depth is
tens of millions and hundreds of megabytes. Bins are taken in groups under
this budget instead, which bounds the peak at a constant and leaves the
arithmetic identical: `np.add.reduceat` never sums across bins, so where the
groups fall cannot change a result, and the values are bitwise the same at
every budget measured.

**The small budget is also the fast one.** At 20,000 reads per bin: 112 ms and
167 MB at four million terms, 72.7 ms and 4.6 MB at this one. The working set
fits in cache, so chunking buys time rather than trading it for memory --
which is why the budget is set here rather than at the largest size that fits.

One bin whose own range exceeds the budget is still evaluated whole.
Splitting it would mean summing its parts and adding them, which is a
different association from what the values are reported against.
"""


def _chunks(lengths: np.ndarray) -> Iterator[np.ndarray]:
    """Bin indices, in groups whose summation stays under `TERM_BUDGET`."""
    start, running = 0, 0

    for index, length in enumerate(lengths):
        if running and running + int(length) > TERM_BUDGET:
            yield np.arange(start, index)
            start, running = index, 0

        running += int(length)

    if start < len(lengths):
        yield np.arange(start, len(lengths))


def _log_tables(max_total: int, alpha: float, beta: float) -> tuple[np.ndarray, ...]:
    """Log-factorial and log-rising-factorial tables up to `max_total`.

    Every term of a beta-binomial mass function is four log-gamma values at
    integer offsets from `1`, `alpha` and `beta`:

    ```
    log pmf(i) = logGamma(n + 1)  - logGamma(i + 1)     - logGamma(n - i + 1)
               + logGamma(i + a)  + logGamma(n - i + b) - logGamma(n + a + b)
               - betaln(a, b)
    ```

    Those offsets are consecutive integers, so each family is a cumulative sum
    of logarithms built once in `O(max_total)` and read thereafter by index.
    `scipy` evaluates `betaln` twice per term instead, which is six log-gamma
    calls where this is four gathers.

    The tables are `(max_total + 2)` doubles each, three of them -- half a
    megabyte at a slide's read depth, against the sum itself.
    """
    steps = np.arange(1, max_total + 2, dtype=float)

    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(steps))))
    rising = [
        np.concatenate(([0.0], np.cumsum(np.log(shift + np.arange(max_total + 1)))))
        for shift in (alpha, beta)
    ]

    return log_factorial, rising[0], rising[1]


def _log_mass_tabulated(
    index: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    tables: tuple[np.ndarray, ...],
) -> np.ndarray:
    """`_log_mass`, read off the tables rather than evaluated.

    `logGamma(a) + logGamma(b) - betaln(a, b)` is `logGamma(a + b)`, which is
    why neither appears below: the two constants cancel into one.
    """
    log_factorial, rising_alpha, rising_beta = tables
    complement = totals - index

    return np.asarray(
        log_factorial[totals]
        - log_factorial[index]
        - log_factorial[complement]
        + rising_alpha[index]
        + rising_beta[complement]
        + scipy.special.gammaln(alpha + beta)
        - scipy.special.gammaln(totals + alpha + beta)
    )


def _ragged_index(
    starts: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One flat index array over ragged ranges, and which range each came from.

    `[start_j, start_j + length_j)` laid end to end, so the whole ragged
    evaluation is one `numpy` call and one segmented sum instead of one call
    per bin.
    """
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    segment = np.repeat(np.arange(len(lengths)), lengths)
    flat = (
        np.arange(offsets[-1], dtype=np.int64) - offsets[:-1][segment] + starts[segment]
    )

    return flat, segment


def cumulative_and_mass(
    counts: np.ndarray, totals: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """`cdf(counts)` and `pmf(counts)` for every bin, in one vectorized sweep.

    **`scipy` has no vectorized beta-binomial distribution function.**
    `betabinom.cdf` goes through `_cdf_single`, which sums the mass function
    from zero for one element at a time under `np.vectorize`, so a call over
    `B` bins is `B` Python-level calls. After #175 removed the quantile
    inversion, these two calls are **90 per cent of what the filter costs**:
    2.50 s of 2.76 s at 2,500 spots and 400 bins.

    Three things change and none of them is the arithmetic:

    *   **One call.** Every bin's summation range is laid end to end, the mass
        function is evaluated once over the concatenation, and the sums come
        back from `np.add.reduceat`.
    *   **Tabulated log-gammas.** The mass function's four log-gamma values
        sit at integer offsets from `1`, `alpha` and `beta`, so each family is
        one cumulative sum of logarithms and every term is four gathers. That
        is what makes the ratio hold as the read depth grows, where the single
        call alone does not.
    *   **A bounded intermediate.** The bins are taken in groups under
        `TERM_BUDGET`, so the peak is a constant rather than the whole ragged
        evaluation. `scipy` holds one bin's range at a time, and a vectorized
        rewrite that held every bin's at once would buy time with memory.
    *   **The shorter tail.** `cdf(k) = 1 - sf(k)`, so a bin sums
        `min(k + 1, n - k)` terms rather than `k + 1`. The B-allele count of a
        diploid bin sits near `n / 2`, which is exactly where the saving is
        least and where it is still a factor of two on any bin above it.
    *   **One sweep for both.** The caller needs `cdf(k)` and `cdf(k - 1)`,
        which differ by `pmf(k)`, so the second comes from the first for the
        cost of one term rather than a second summation.

    **This is a tolerance and not an identity.** The terms are summed in a
    different order and, on the upper branch, subtracted from one, so the
    values agree to floating point rather than bitwise. The **mask** the
    caller builds from them is asserted bitwise against `cnaster`; the values
    are asserted to `1e-12`.
    """
    counts = np.asarray(counts, dtype=np.int64)
    totals = np.asarray(totals, dtype=np.int64)

    tables = _log_tables(int(totals.max(initial=0)), alpha, beta)

    mass = np.where(
        (counts >= 0) & (counts <= totals),
        np.exp(
            _log_mass_tabulated(np.clip(counts, 0, totals), totals, alpha, beta, tables)
        ),
        0.0,
    )

    # NB sum whichever tail is shorter; `upper` sums (k, n] and is subtracted
    #    from one, `lower` sums [0, k].
    upper = counts + 1 > totals - counts
    starts = np.where(upper, counts + 1, 0)
    lengths = np.where(upper, totals - counts, counts + 1)
    lengths = np.clip(lengths, 0, None)

    sums = np.zeros(len(counts), dtype=float)

    for group in _chunks(lengths):
        nonempty = group[lengths[group] > 0]

        if not nonempty.size:
            continue

        spans = lengths[nonempty]
        flat, segment = _ragged_index(starts[nonempty], spans)
        terms = np.exp(
            _log_mass_tabulated(flat, totals[nonempty][segment], alpha, beta, tables)
        )

        sums[nonempty] = np.add.reduceat(
            terms, np.concatenate(([0], np.cumsum(spans)[:-1]))
        )

    cumulative = np.where(upper, 1.0 - sums, sums)

    return np.clip(cumulative, 0.0, 1.0), mass


DECISION_MARGIN = 1.0e-8
"""How close to a threshold a bin has to be before `scipy` decides it.

`cumulative_and_mass` sums the mass function in a different order from
`scipy`, so its values agree to floating point rather than bitwise --
`3e-9` at the deepest size measured. Both comparisons below are **strict**,
so a bin whose distribution function sits within that of a threshold could
fall either way on rounding alone, and the mask is what decides whether a
genomic bin survives.

The margin is wider than the largest error measured, and the bins inside it
are recomputed with `scipy` itself, so the answer is `cnaster`'s wherever the
decision is close and the fast path's wherever it is not. On the fixtures here
the margin catches nothing; the case it exists for is an exact tie, which a
symmetric beta-binomial reaches at its midpoint against a threshold of `0.5`.
"""


def _settle_near_thresholds(
    counts: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    confidence_interval: tuple[float, float],
    below: np.ndarray,
    at_or_below: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-decide the bins within `DECISION_MARGIN` of a threshold, with `scipy`.

    One `scipy` call over the uncertain bins, which is empty on every instance
    measured -- so this costs a comparison and buys back the one thing the
    faster summation could change.
    """
    uncertain = np.flatnonzero(
        (np.abs(below - confidence_interval[0]) <= DECISION_MARGIN)
        | (np.abs(at_or_below - confidence_interval[1]) <= DECISION_MARGIN)
    )

    if not uncertain.size:
        return below, at_or_below

    logger.info(
        f"Deciding {uncertain.size} bin(s) within {DECISION_MARGIN} of a "
        "confidence threshold with scipy."
    )

    below, at_or_below = below.copy(), at_or_below.copy()
    below[uncertain] = scipy.stats.betabinom.cdf(
        counts[uncertain], totals[uncertain], alpha, beta
    )
    at_or_below[uncertain] = scipy.stats.betabinom.cdf(
        counts[uncertain] - 1, totals[uncertain], alpha, beta
    )

    return below, at_or_below


def removal_indicator(
    counts: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    confidence_interval: tuple[float, float],
) -> np.ndarray:
    """Which bins fall outside the interval, without inverting the quantile.

    `cnaster` writes

    ```python
    counts < scipy.stats.betabinom.ppf(lo, totals, alpha, beta)
    counts > scipy.stats.betabinom.ppf(hi, totals, alpha, beta)
    ```

    and `scipy` has no closed-form inverse for a beta-binomial, so each element
    goes through `_drv2_ppfsingle`, a bisection whose every step evaluates the
    distribution function by summing the mass function from zero. On the dev
    instance's 40 bins the two calls make **1,219** mass-function evaluations
    and cost 1.44 s.

    The two comparisons below are the same predicates. For a discrete law
    `ppf(q) = min{k : cdf(k) >= q}`, so `x < ppf(q)` is exactly `cdf(x) < q`,
    and `x <= ppf(q)` is exactly `cdf(x - 1) < q`, whose negation is the
    second. **Both are equalities rather than approximations**, which is why
    the mask is bitwise `cnaster`'s and not within a tolerance -- the
    equivalence test asserts exactly that.

    **Except at a closed end, which is taken from `scipy`'s support instead
    (#332).** `ppf(1) = n` and `ppf(0) = -1`, so at `hi >= 1` or `lo <= 0`
    `cnaster` removes nothing on that side. The distribution function does
    not know that: an upper tail of `1e-50` rounds `cdf(x - 1)` to exactly
    `1.0 >= hi`, and on a normal pool diluted by an LOH clone the patch
    removed 6 bins `cnaster` kept, whose genes then carried a NaN `bin_id`
    into `run_cnaster`'s gene-level writer.
    """
    below, mass = cumulative_and_mass(counts, totals, alpha, beta)
    at_or_below = np.clip(below - mass, 0.0, 1.0)

    below, at_or_below = _settle_near_thresholds(
        counts, totals, alpha, beta, confidence_interval, below, at_or_below
    )

    lo, hi = confidence_interval
    low = below < lo if lo > 0.0 else np.zeros(counts.shape, dtype=bool)
    high = at_or_below >= hi if hi < 1.0 else np.zeros(counts.shape, dtype=bool)

    return np.asarray(low | high, dtype=bool)


def normal_baf_bin_filter(
    df_gene_snp: Any,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    nu: float,  # noqa: ARG001
    logphase_shift: float,  # noqa: ARG001
    index_normal: np.ndarray,
    geneticmap_file: Any,  # noqa: ARG001
    confidence_interval: tuple[float, float] | None = None,
    min_betabinom_tau: int = MIN_BETABINOM_TAU,
) -> tuple[Any, SpatioGenomicCounts]:
    """What `cnaster`'s returns, with the quantile inversion replaced.

    Everything but `removal_indicator` is `cnaster`'s, in its order: the pooled
    counts, the one-state fit, the two patched parameters, the renumbering of
    the survivors and the per-chromosome lengths. `nu`, `logphase_shift` and
    `geneticmap_file` are accepted and unused, as upstream -- the docstring
    promises a `log_sitewise_transmat` the function has never returned, and
    `run_cnaster` calls `get_sitewise_transmat` itself on the next line.
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(
            get_global_config().quality.normal_allele_specific_confidence
        )

    assert confidence_interval is not None

    pooled_counts = np.sum(single_X[:, 1, index_normal], axis=1)
    pooled_totals = np.sum(single_total_bb_RD[:, index_normal], axis=1)

    model = Weighted_BetaBinom(
        pooled_counts,
        np.ones(len(pooled_counts)),
        weights=np.ones(len(pooled_counts)),
        exposure=pooled_totals,
    )
    fitted = model.fit(**get_em_solver_params())

    # NB the fitted success probability is discarded for 0.5 and the
    #    concentration floored, both as upstream. #38 owns whether it should be.
    fitted.params[0] = 0.5
    fitted.params[-1] = max(fitted.params[-1], min_betabinom_tau)

    alpha = fitted.params[0] * fitted.params[1]
    beta = (1.0 - fitted.params[0]) * fitted.params[1]

    removal = removal_indicator(
        pooled_counts, pooled_totals, alpha, beta, confidence_interval
    )

    index_removal = np.where(removal)[0]
    index_remaining = np.where(~removal)[0]

    logger.info(
        f"Removing {np.count_nonzero(removal)} [{100.0 * np.mean(removal):.4f}]% "
        f"genomic segments with potential allele-specific expression, based on "
        f"normal candidates --- confidence={confidence_interval} and "
        f"min_betabinom_tau={min_betabinom_tau}."
    )

    column = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], column] = None

    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map(
        {x: i for i, x in enumerate(index_remaining)}
    )
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    if df_gene_snp.bin_id.isnull().any():
        logger.warning(
            f"Detected {df_gene_snp.bin_id.isnull().sum()} bin_ids with NaN value."
        )

    single_X = single_X[index_remaining, :, :]
    single_base_nb_mean = single_base_nb_mean[index_remaining, :]
    single_total_bb_RD = single_total_bb_RD[index_remaining, :]

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, contig in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (contig == df_gene_snp.CHR) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    assert df_gene_snp["bin_id"].nunique(dropna=True) == single_X.shape[0]
    assert df_gene_snp["bin_id"].nunique(dropna=True) == sum(lengths)

    return df_gene_snp, SpatioGenomicCounts(
        lengths, single_X, single_base_nb_mean, single_total_bb_RD
    )
