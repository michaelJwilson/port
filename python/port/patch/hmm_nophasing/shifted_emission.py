r"""`hmm_nophasing`, with the per-clone library normalizer folded in (#276).

**`cnaster` computes the shift and throws it away.** `compute_logmu_shifts`
is defined at `hmm_nophasing.py:133` and its only call site is commented out
at `:279`, under a `logger.warning("logmu_shifts are not currently
supported.")` and a `# TODO fold in logmu_shifts`. So every fitted RDR
parameter a run reports is the one without it.

This folds it in:

.. math::
    \exp(\theta_i) \longrightarrow \exp(\theta_i - \log Z_{c(g)})

with :math:`\log Z_c` the quantity `cnaster`'s own `compute_logmu_shifts`
returns. **Upstream's function is called, not reimplemented** -- the patch
applies a quantity the dependency defines rather than deriving a second one
that would then need refereeing against the first.

## Off by default, because this changes every fitted RDR parameter

`CLAUDE.md` forbids a silent behaviour change and enabling the shift is one:
it debiases :math:`\log\mu` and every downstream number moves. Off, the call
goes to `cnaster`'s own coded emission unchanged, which is the path
`tests/test_buffered_emission.py` pins bitwise. The flag is a class
attribute because `port` does not call this method -- `optimize_params` does,
from inside `cnaster` -- so a keyword would have to be threaded through a
function this repository does not replace.

## The shift is computed once, and carried per clone

`compute_logmu_shifts` is **patched** rather than called
(`port.patch.hmm_nophasing.logmu_shift`), for two reasons that are both
about the call site rather than the arithmetic.

*It returns one value per segment.* `n_clones` distinct numbers in
`n_segments` floats, so it has to be indexed by a running offset; indexed by
clone -- which is what it looks like it wants -- it hands every clone the
first clone's shift, silently, on any instance whose first clone is longer
than the clone count. The patch returns `(n_clones,)`, which cannot be read
that way, and `np.repeat` recovers upstream's array bitwise.

*Upstream's own call site would compute it `n_states` times.* The commented
block at `hmm_nophasing.py:275-279` sits inside `for i in range(n_states)`,
and the shift does not depend on the state. It is computed once here.

The reduction itself is upstream's compiled loop, kept: a
`scipy.special.logsumexp` over per-clone views measured **2.1x slower** at
the stress size and 3.9x at the gate one, so the loop is not the thing worth
replacing.

## The key is ``(clone, obs, total)``, and the duplication is the minimum

The shift multiplies :math:`\mu`, which enters `_nb_logpmf_1d` and nothing
else; `p_binom` and `taus` are untouched. So the allele channel keeps
`CountEncoder` and the whole-genome path, and only the read-depth channel
pays anything.

`CountEncoder` compresses to unique ``(obs, total)`` pairs over the whole
concatenated genome, which is what makes the coded emission fast. The shift
breaks that: two segments in different clones sharing a pair now need
different rates, and one entry cannot hold two. **Adding the clone to the
key separates exactly those and nothing else** -- the unique rows are, per
clone, that clone's unique pairs -- so the compression lost is the
compression the shift makes impossible. Measured at 10 clones x 29,000:
10,987 unique pairs unshifted against **76,245** triples, and one
`CountEncoder` per clone reaches the same 76,245. The duplication is the
shift's, not the encoding's.

Against that per-clone alternative, which is what #276 deferred, the triple
is the simpler structure at the same cost and **not** a speedup: one
`np.unique` and one gather instead of `n_clones` encoders, `n_clones` sparse
mappings and a buffer written in blocks. Scoring is 0.95 ms against 1.08 at
the gate size and 33.01 against 31.96 at the stress one -- a wash either way
-- and the build, which happens once per fit and is cached, is 332.6 ms
against 225.5. What it buys is one code path; what it costs is a slower
build once.

The whole call, against the unshifted emission upstream runs: **1.12x** at
3 x 1,000 and **1.83x** at 10 x 29,000, `K = 7`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import exp
from typing import Any, NamedTuple

import numpy as np
from cnaster.config import get_global_config
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_nophasing import hmm_nophasing as UPSTREAM

from port.patch.hmm_nophasing.logmu_shift import shifts as logmu_shifts
from port.patch.plotting.clone_paths import state_vector

__all__ = ["UPSTREAM", "hmm_nophasing", "logmu_shift", "neutral_state"]

NEUTRAL_BAF_TOLERANCE = 0.05
"""How far from 0.5 a state's allele fraction may sit and still be neutral.

The neutral state is the one with a balanced allele fraction and the lowest
`mu` (#293). `p` is fitted, so "balanced" is a tolerance; 0.05 admits the
0.4873-0.4999 #292's fits return for the planted 0.5 and refuses the next
planted state, 0.42.
"""


class _Triples(NamedTuple):
    """The genome-wide `(clone, obs, total)` compression, and how to undo it.

    `bounds[c]:bounds[c + 1]` is clone `c`'s block of unique rows, contiguous
    because the clone index is the **first** column and `np.unique` sorts
    lexicographically. That is what lets one shift be applied per block with
    a scalar rather than per entry with a gather.
    """

    obs: np.ndarray
    total: np.ndarray
    inverse: np.ndarray
    bounds: np.ndarray


def _triples(
    obs_count: np.ndarray, total_count: np.ndarray, lengths: tuple[int, ...]
) -> _Triples:
    """Compress `(clone, obs, total)` once over the whole genome.

    **The clone index is what lets a shared pair carry two rates.**
    `CountEncoder` compresses `(obs, total)` genome-wide, so two segments in
    different clones sharing a pair collapse to one entry -- and under the
    shift they need different rates, which one entry cannot hold. Adding the
    clone to the key separates exactly those and nothing else: the unique
    rows are, per clone, that clone's unique pairs.

    The alternative is one `CountEncoder` per clone, which is what #276
    deferred at **1.69x**. This keeps one pass and one mapping, so the
    duplication it admits is the minimum the shift requires rather than a
    re-encoding of the genome per clone.

    Rounding follows `CountEncoder.construct_unique_encoding`: non-integer
    counts are rounded to the configured decimals before the compare, so two
    entries that upstream would collapse are not separated here by a float
    the encoder never looked at.
    """
    clones = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)

    counts = np.column_stack(
        [clones.astype(np.float64), np.asarray(obs_count), np.asarray(total_count)]
    )

    if not np.issubdtype(np.asarray(total_count).dtype, np.integer):
        counts = counts.round(decimals=get_global_config().hmm.compression_decimals)

    unique, inverse = np.unique(counts, axis=0, return_inverse=True)

    return _Triples(
        obs=np.ascontiguousarray(unique[:, 1]),
        total=np.ascontiguousarray(unique[:, 2]),
        inverse=inverse.reshape(-1),
        bounds=np.searchsorted(unique[:, 0], np.arange(len(lengths) + 1)),
    )


def _current(lengths: tuple[int, ...], n_segments: int) -> tuple[int, ...]:
    """The clone lengths of the sequence actually being fitted.

    **`cnaster` passes stale ones once clones merge.** `hmrf.py:564` sets
    `clone_lengths` from the initial pseudobulk, before the loop, and never
    updates it; the HMRF then merges clones, so on #292's genome it still
    says six clones of 300 bins while the fit is over three -- 1,800 against
    900. Every clone carries the whole genome, so the length is the one bin
    count and the current number of clones is the decode's size over it.
    Anything that does not divide is refused rather than guessed.
    """
    if int(sum(lengths)) == n_segments:
        return lengths

    if lengths and len(set(lengths)) == 1 and n_segments % lengths[0] == 0:
        return (lengths[0],) * (n_segments // lengths[0])

    msg = f"clone lengths {lengths} do not tile the {n_segments} segments decoded"
    raise ValueError(msg)


def _stacked(normal_log_lambda: Any, lengths: tuple[int, ...]) -> np.ndarray:
    """`log lambda` over the clone-stacked sequence the decode indexes.

    **`cnaster` passes it per genome bin.** `hmrf.py:476` builds
    `normal_lambda` by summing the baseline over spots, so it has one entry
    per bin, while the decode and the reduction walk `sum(lengths)` stacked
    segments. The reduction is a `numba` loop without bounds checks, so the
    short array would be read past its end rather than refused. Every clone
    shares the one normal profile, so the stacked form is the per-bin one
    repeated clone after clone.
    """
    values = np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)
    total = int(sum(lengths))

    if values.size == total:
        return values

    if lengths and all(length == values.size for length in lengths):
        return np.tile(values, len(lengths))

    msg = (
        f"normal_log_lambda has {values.size} entries; expected one per genome "
        f"bin ({lengths[0] if lengths else 0}) or per stacked segment ({total})"
    )
    raise ValueError(msg)


def neutral_state(log_mu: np.ndarray, p_binom: np.ndarray) -> int:
    """The balanced state with the lowest `mu`: the one pinned to `mu = 1`.

    Balanced is within :data:`NEUTRAL_BAF_TOLERANCE` of 0.5, in either
    allele's convention. Where no state is, the one closest to 0.5 is taken
    rather than none, because the shifted likelihood has no scale without a
    pin and an unpinned fit is not comparable to anything.
    """
    rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    distance = np.abs(np.asarray(p_binom, dtype=np.float64).reshape(-1) - 0.5)
    balanced = np.flatnonzero(distance <= NEUTRAL_BAF_TOLERANCE)

    if balanced.size == 0:
        return int(np.argmin(distance))

    return int(balanced[np.argmin(rates[balanced])])


class hmm_nophasing(UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_nophasing`, with the shift applied when the flag is set.

    A subclass rather than a mixin: it replaces a named `cnaster` class, which
    is what `CLAUDE.md`'s four-job rule asks a `patch/` module to do, and
    every method it does not override is upstream's by inheritance rather
    than by delegation.

    The name is lower-case because `cnaster`'s is, and a drop-in that renamed
    the thing it replaces would not be one.
    """

    apply_logmu_shift: bool = False
    """Off by default. :func:`logmu_shift` is what turns it on.

    A class attribute rather than a keyword, because the caller is
    `optimize_params` inside `cnaster` and a keyword would have to reach it
    through a function this repository does not replace.
    """

    def _clone_triples(self, encoder: Any, lengths: tuple[int, ...]) -> _Triples:
        """`(clone, obs, total)` compressed once over the whole genome.

        One `np.unique`, not one per clone. Built once per
        `(encoder, lengths)` and cached, keyed on the encoder's identity
        **and holding a reference to it**, so an id cannot be reused by a
        later object while the entry is live. `optimize_params` builds the
        encoder once per fit, so this is one build per fit rather than one
        per optimizer iteration.
        """
        cache: dict[tuple[int, tuple[int, ...]], tuple[Any, _Triples]]
        cache = getattr(self, "_triple_cache", None) or {}
        self._triple_cache = cache

        key = (id(encoder), lengths)

        if key not in cache:
            cache[key] = (
                encoder,
                _triples(
                    np.asarray(encoder.obs_count).reshape(-1),
                    np.asarray(encoder.total_count).reshape(-1),
                    lengths,
                ),
            )

        return cache[key][1]

    def _decode(self) -> np.ndarray | None:
        """The hard decode the shift is taken at, or `None` if unavailable.

        Upstream's commented-out block reads `self.get_state_posteriors()`
        and then `self.get_copy_states(log_gamma)`; this reads the posteriors
        the instance is carrying and applies upstream's own `get_copy_states`
        to them. Returning `None` rather than a default is deliberate -- a
        shift computed from a decode nobody supplied is a number nobody asked
        for.
        """
        posteriors = getattr(self, "state_posteriors", None)

        if posteriors is None:
            return None

        decoded: np.ndarray = self.get_copy_states(np.asarray(posteriors))

        return decoded

    _row_shift: np.ndarray | None = None
    """The last shifted fit's shift, one entry per clone-stacked segment.

    **Class state, and deliberately so.** `hmm.py:155` rescores the fit
    through `hmmclass.compute_emission_probability_nb_betabinom`, a static
    method called on the class with no argument that could carry the shift.
    The pipeline is sequential, the call follows `optimize` directly, and the
    override applies it only to an input of exactly this length, so a stale
    value cannot reach a different problem unnoticed.
    """

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X: np.ndarray,
        base_nb_mean: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        total_bb_RD: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's dense emission, with the last fit's shift applied.

        The shift enters the negative binomial through its mean,
        `base * exp(log_mu - shift)`, which is `base * exp(-shift)` against
        the unshifted `exp(log_mu)`. So it is applied to the exposure and the
        rest is upstream's kernel unchanged.
        """
        shift = hmm_nophasing._row_shift

        if (
            hmm_nophasing.apply_logmu_shift
            and shift is not None
            and shift.size == np.asarray(X).shape[0]
        ):
            # NB **recentred, because the rates have no scale.** The shifted
            #    likelihood is flat along `mu -> c mu`, and the fit wanders
            #    along it: measured, `log mu` and the shift both near -7,024 on
            #    #292's realization 3. `exp(-shift) * exp(log_mu)` is then
            #    `inf * 0`. Taking a common `c` off both leaves the product --
            #    the emission -- unchanged and each factor near one.
            centre = float(np.mean(shift))
            base_nb_mean = np.asarray(base_nb_mean) * np.exp(-(shift - centre))[:, None]
            log_mu = np.asarray(log_mu) - centre

        scored: tuple[np.ndarray, np.ndarray]
        scored = UPSTREAM.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        return scored

    def optimize(
        self,
        X: np.ndarray,
        lengths: np.ndarray,
        n_states: int,
        base_nb_mean: np.ndarray,
        total_bb_RD: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Upstream's fit, then a shifted decode.

        Inside the fit the shift is already applied: the M step's objective
        and the E step's posteriors both come from the coded emission below.
        What upstream does after it is not shifted -- `hmm_nophasing.py:1085`
        rescored the fit through the dense emission for `log_gamma` -- so
        that is redone here with the shift, and the shift is recorded for
        `hmm.py:155`, which rescores it once more.

        The rates are returned as fitted. The shifted mean `base * mu / sum
        lambda mu` is unchanged by `mu -> c mu`, so their scale is arbitrary
        here; `port.patch.hmrf.run_core_inference` pins it once, after the
        whole optimization.
        """
        hmm_nophasing._row_shift = None

        res: dict[str, Any] = super().optimize(
            X, lengths, n_states, base_nb_mean, total_bb_RD, *args, **kwargs
        )

        normal_lambda = kwargs.get("normal_lambda")
        clone_lengths = kwargs.get("clone_lengths")
        decode = self._decode()

        if (
            not self.apply_logmu_shift
            or "m" not in self.params
            or normal_lambda is None
            or clone_lengths is None
            or decode is None
        ):
            return res

        rates = state_vector(res["new_log_mu"])

        n_segments = int(np.asarray(X).shape[0])
        current = _current(
            tuple(int(length) for length in np.asarray(clone_lengths)), n_segments
        )

        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            _stacked(np.log(np.asarray(normal_lambda, dtype=np.float64)), current),
            np.asarray(current, dtype=np.int64),
        )
        hmm_nophasing._row_shift = np.repeat(shifts, current)

        log_emission_rdr, log_emission_baf = (
            self.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD,
                res["new_p_binom"],
                res["new_taus"],
            )
        )
        log_gamma = self.get_state_posteriors(
            lengths,
            res["new_log_transmat"],
            res["new_log_startprob"],
            log_emission_rdr + log_emission_baf,
            kwargs.get("log_sitewise_transmat"),
        )

        res["log_gamma"] = log_gamma
        res["pred_cnv"] = np.argmax(log_gamma, axis=0)

        return res

    def compute_emission_probability_nb_betabinom_coded(
        self,
        nbEncoder: Any,
        bbEncoder: Any,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
        clone_stack: bool = True,
        scratch_rdr: Any = None,
        scratch_baf: Any = None,
        normal_log_lambda: Any = None,
        clone_lengths: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's emission, with `log_mu` debiased per clone.

        Hands the call on unchanged unless **all** of the flag, the exposures,
        the clone lengths and a decode are present. A shift needs every one of
        them, and computing it from a default would be a number nobody asked
        for -- which is also upstream's own reason for warning rather than
        guessing at `hmm_nophasing.py:279`.
        """
        decode = self._decode()

        if (
            not self.apply_logmu_shift
            or normal_log_lambda is None
            or clone_lengths is None
            or decode is None
        ):
            unshifted: tuple[np.ndarray, np.ndarray]
            unshifted = super().compute_emission_probability_nb_betabinom_coded(
                nbEncoder,
                bbEncoder,
                log_mu,
                alphas,
                p_binom,
                taus,
                clone_stack=clone_stack,
                scratch_rdr=scratch_rdr,
                scratch_baf=scratch_baf,
                normal_log_lambda=normal_log_lambda,
                clone_lengths=clone_lengths,
            )
            return unshifted

        if nbEncoder.n_spots != 1 or bbEncoder.n_spots != 1:
            msg = (
                f"one spot only: got {nbEncoder.n_spots} and {bbEncoder.n_spots}. "
                "The shift is per clone along the genomic axis (#276)."
            )
            raise ValueError(msg)

        # NB `(n_states,)`, normalized once at the edge. The fit returns
        #    `(n_states, 1)` and nothing else can reach here (#278), so the
        #    column is dropped rather than indexed at four call sites below.
        rates = state_vector(log_mu)
        dispersions = state_vector(alphas)
        probabilities = state_vector(p_binom)
        concentrations = state_vector(taus)

        n_states = rates.shape[0]
        lengths = _current(
            tuple(int(length) for length in np.asarray(clone_lengths)),
            int(np.asarray(decode).size),
        )

        # NB **once per call, not once per state.** Upstream's commented-out
        #    call sits inside `for i in range(n_states)` at
        #    `hmm_nophasing.py:275-279`, so folding it in as written would
        #    recompute the whole reduction `n_states` times over an
        #    `n_segments` array for a quantity that does not depend on the
        #    state. That is the efficiency here; the reduction itself is
        #    upstream's own loop, kept (`logmu_shift`).
        #
        #    `(n_clones,)`, so it is indexed by clone. Upstream's shape is
        #    `(n_segments,)` and indexing *that* by clone is silently wrong.
        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            _stacked(normal_log_lambda, lengths),
            np.asarray(lengths, dtype=np.int64),
        )

        # NB the allele channel is untouched by the shift, so it keeps the
        #    whole-genome encoder and upstream's path.
        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        baf_uniq = (
            scratch_baf[0] if scratch_baf else np.zeros((n_states, len(bb_endog)))
        )

        for state in range(n_states):
            _bb_logpmf_1d(
                bb_endog,
                bb_exposure,
                probabilities[state],
                concentrations[state],
                baf_uniq[state, :],
            )

        log_emit_baf = bbEncoder.decode_array(baf_uniq, 0)

        # NB scored once per unique `(clone, obs, total)`, then decoded by a
        #    single gather. The clone's block is contiguous, so its shift is
        #    a scalar the kernel already takes -- no per-entry rate array and
        #    no second mapping.
        triples = self._clone_triples(nbEncoder, lengths)
        rdr_uniq = np.zeros((n_states, triples.obs.size))

        for clone in range(len(lengths)):
            first, last = int(triples.bounds[clone]), int(triples.bounds[clone + 1])

            if first == last:
                continue

            for state in range(n_states):
                _nb_logpmf_1d(
                    triples.obs[first:last],
                    triples.total[first:last],
                    exp(rates[state] - shifts[clone]),
                    dispersions[state],
                    rdr_uniq[state, first:last],
                )

        log_emit_rdr = rdr_uniq[:, triples.inverse]

        if clone_stack:
            return log_emit_rdr, log_emit_baf

        return log_emit_rdr[:, :, None], log_emit_baf[:, :, None]


@contextmanager
def logmu_shift() -> Iterator[None]:
    """Turn the shift on for the block, and back to what it was after.

    The flag is a class attribute, so a run that set it and left it would make
    every later comparison in the same process a shifted one. Restored rather
    than cleared, so nesting does not lie.
    """
    previous = hmm_nophasing.apply_logmu_shift
    hmm_nophasing.apply_logmu_shift = True

    try:
        yield
    finally:
        hmm_nophasing.apply_logmu_shift = previous
