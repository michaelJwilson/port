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
from typing import Any

import numpy as np
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
    """
    below = scipy.stats.betabinom.cdf(counts, totals, alpha, beta)
    at_or_below = scipy.stats.betabinom.cdf(counts - 1, totals, alpha, beta)

    return np.asarray(
        (below < confidence_interval[0]) | (at_or_below >= confidence_interval[1]),
        dtype=bool,
    )


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
