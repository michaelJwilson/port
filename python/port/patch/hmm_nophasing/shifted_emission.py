r"""Replaces `cnaster.hmm_nophasing.hmm_nophasing` with the per-clone library normalizer folded in (#276).

.. math::
    \exp(\theta_i) \longrightarrow \exp(\theta_i - \log Z_{c(g)})

`cnaster` defines :math:`\log Z_c` (`compute_logmu_shifts`) but never applies
it. Off by default (class flag); the `SHIFT_SWAPS` row binds it on. The shift
is computed once per call, `(n_clones,)`, and read-depth scored per unique
`(clone, obs, total)`; the allele channel is unchanged.
"""

from __future__ import annotations

from math import exp
from typing import Any, NamedTuple

import numpy as np
from cnaster.config import get_global_config, start_time
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_nophasing import hmm_nophasing as UPSTREAM
from cnaster.logger import get_logger
from sal.ragged import Ragged

from port.patch._clone_paths import state_vector
from port.patch.hmm_nophasing.gradient import EmGradient, analytic_bfgs
from port.patch.hmm_nophasing.logmu_shift import shifts as logmu_shifts

logger = get_logger(__name__, start_time=start_time)

__all__ = [
    "UPSTREAM",
    "hmm_nophasing",
    "neutral_state",
    "normal_clone",
    "release",
    "shifted",
]

NEUTRAL_BAF_TOLERANCE = 0.05
"""Max distance of a state's allele fraction from 0.5 for it to count as balanced (#293)."""


class _Triples(NamedTuple):
    """Genome-wide `(clone, obs, total)` compression; `bounds[c]:bounds[c + 1]` is clone `c`'s rows."""

    obs: np.ndarray
    total: np.ndarray
    inverse: np.ndarray
    bounds: np.ndarray


def _clone_major(
    channel: Any, lengths: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """One clone-stacked channel as a contiguous array, and each entry's clone (#349)."""
    values = np.ascontiguousarray(np.asarray(channel).reshape(-1))
    layout = Ragged(values=values, lengths=lengths)
    clones = np.repeat(np.arange(layout.n_segments, dtype=np.int64), layout.lengths)

    return values, clones


def clone_count_triples(
    obs_count: np.ndarray, total_count: np.ndarray, lengths: tuple[int, ...]
) -> _Triples:
    """Compress `(clone, obs, total)` once over the whole genome.

    Rounds non-integer counts as `CountEncoder.construct_unique_encoding` does.
    """
    obs, clones = _clone_major(obs_count, lengths)
    total, _ = _clone_major(total_count, lengths)

    counts = np.column_stack([clones.astype(np.float64), obs, total])

    if not np.issubdtype(total.dtype, np.integer):
        counts = counts.round(decimals=get_global_config().hmm.compression_decimals)

    unique, inverse = np.unique(counts, axis=0, return_inverse=True)

    return _Triples(
        obs=np.ascontiguousarray(unique[:, 1]),
        total=np.ascontiguousarray(unique[:, 2]),
        inverse=inverse.reshape(-1),
        bounds=np.searchsorted(unique[:, 0], np.arange(len(lengths) + 1)),
    )


def current_clone_lengths(lengths: tuple[int, ...], n_segments: int) -> tuple[int, ...]:
    """The clone lengths of the sequence being fitted.

    `cnaster` passes stale lengths after clones merge (`hmrf.py:564`); equal
    lengths are re-tiled to `n_segments`. Raises `ValueError` if they do not tile.
    """
    if int(sum(lengths)) == n_segments:
        return lengths

    if lengths and len(set(lengths)) == 1 and n_segments % lengths[0] == 0:
        return (lengths[0],) * (n_segments // lengths[0])

    msg = f"clone lengths {lengths} do not tile the {n_segments} segments decoded"
    raise ValueError(msg)


def stacked_log_lambda(normal_log_lambda: Any, lengths: tuple[int, ...]) -> np.ndarray:
    """`log lambda` tiled over the clone-stacked sequence; `cnaster` passes it per bin.

    Raises `ValueError` if the size is neither per bin nor per stacked segment.
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


def normal_clone(p_binom: np.ndarray, path: np.ndarray) -> tuple[int, float]:
    """The normal clone of `path`, `(n_obs, n_clones)`, and its share of balanced bins (#299, #389)."""
    balanced = (
        np.abs(np.asarray(p_binom, dtype=np.float64).reshape(-1) - 0.5)
        <= NEUTRAL_BAF_TOLERANCE
    )
    decoded = np.asarray(path, dtype=np.int64)
    share = balanced[decoded.reshape(decoded.shape[0], -1)].mean(axis=0)
    normal = int(np.argmax(share))
    return normal, float(share[normal])


def neutral_state(
    log_mu: np.ndarray, p_binom: np.ndarray, path: np.ndarray | None = None
) -> int:
    """The state pinned to `mu = 1`: the normal clone's most occupied balanced state (#299).

    Without a usable path, the balanced state with the lowest `mu`; with no
    balanced state, the one closest to 0.5.
    """
    rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    distance = np.abs(np.asarray(p_binom, dtype=np.float64).reshape(-1) - 0.5)
    balanced = distance <= NEUTRAL_BAF_TOLERANCE

    if not balanced.any():
        return int(np.argmin(distance))

    if path is not None:
        decoded = np.asarray(path, dtype=np.int64)
        decoded = decoded.reshape(decoded.shape[0], -1)
        normal, share = normal_clone(p_binom, decoded)

        if share > 0.0:
            counts = np.bincount(decoded[:, normal], minlength=rates.size)
            counts = np.where(balanced, counts, -1)

            return int(np.argmax(counts))

    candidates = np.flatnonzero(balanced)

    return int(candidates[np.argmin(rates[candidates])])


class hmm_nophasing(UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_nophasing`, with the shift applied when the flag is set."""

    # NB options are class attributes: the caller is `optimize_params` inside
    #    `cnaster`; set on a subclass the `SHIFT_SWAPS` row installs (#517).

    apply_logmu_shift: bool = False
    """Off here, as `cnaster` is; the `SHIFT_SWAPS` row binds it on."""

    analytic_gradient: bool = True
    """M step's gradient in closed form (#433); a stated departure from `cnaster` (T- #617)."""

    emission_kernels: str = "cnaster"
    """`cnaster` (default) or `sal`: which kernels score the coded emission (#425)."""

    def _clone_triples(self, encoder: Any, lengths: tuple[int, ...]) -> _Triples:
        """`(clone, obs, total)` triples, cached per encoder identity and `lengths`."""
        cache: dict[tuple[int, tuple[int, ...]], tuple[Any, _Triples]]
        cache = getattr(self, "_triple_cache", None) or {}
        self._triple_cache = cache

        key = (id(encoder), lengths)

        if key not in cache:
            cache[key] = (
                encoder,
                clone_count_triples(encoder.obs_count, encoder.total_count, lengths),
            )

        return cache[key][1]

    def _decode(self) -> np.ndarray | None:
        """The hard decode of the carried posteriors, or `None` if there are none."""
        posteriors = getattr(self, "state_posteriors", None)

        if posteriors is None:
            return None

        decoded: np.ndarray = self.get_copy_states(np.asarray(posteriors))

        return decoded

    _row_shift: np.ndarray | None = None
    """The last shifted fit's shift per clone-stacked segment; read by `hmm.py:155`'s static rescore."""

    @classmethod
    def compute_emission_probability_nb_betabinom(
        cls,
        X: np.ndarray,
        base_nb_mean: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        total_bb_RD: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's dense emission, with the last fit's shift applied to the exposure."""
        shift = hmm_nophasing._row_shift

        if (
            cls.apply_logmu_shift
            and shift is not None
            and shift.size == np.asarray(X).shape[0]
        ):
            # NB recentred: the shifted likelihood is flat along `mu -> c mu`,
            #    so both factors can drift to `inf * 0`; the product is unchanged.
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
        """Upstream's fit, then a shifted decode recorded for `hmm.py:155`.

        Rates are returned unpinned; `port.patch.hmrf.run_core_inference` pins the scale.
        """
        hmm_nophasing._row_shift = None

        # NB analytic gradient via `minimize`'s callable `method` (#433), BFGS only (#448).
        if (
            self.analytic_gradient
            and not args
            and kwargs.get("optimizer", "BFGS") == "BFGS"
        ):
            kwargs["optimizer"] = analytic_bfgs(
                EmGradient.for_fit(
                    self, X, n_states, base_nb_mean, total_bb_RD, **kwargs
                )
            )

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
        current = current_clone_lengths(
            tuple(int(length) for length in np.asarray(clone_lengths)), n_segments
        )

        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            stacked_log_lambda(
                np.log(np.asarray(normal_lambda, dtype=np.float64)), current
            ),
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

        Delegates unchanged unless the flag, exposure, clone lengths and a decode are all present.
        """
        decode = self._decode()

        if (
            not self.apply_logmu_shift
            or normal_log_lambda is None
            or clone_lengths is None
            or decode is None
        ):
            if self.apply_logmu_shift:
                # NB expected (no BAF exposure, no first decode); exposure withheld
                #    so upstream does not warn (#362).
                logger.debug(
                    "logmu shift not applied on this call: %s",
                    ", ".join(
                        name
                        for name, missing in (
                            ("no exposure", normal_log_lambda is None),
                            ("no clone lengths", clone_lengths is None),
                            ("no decode yet", decode is None),
                        )
                        if missing
                    ),
                )
                normal_log_lambda = None
            if self.emission_kernels == "sal":
                from port.patch.hmm_nophasing.dense_emission import coded_emission

                return coded_emission(
                    nbEncoder,
                    bbEncoder,
                    log_mu,
                    alphas,
                    p_binom,
                    taus,
                    clone_stack=clone_stack,
                )

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

        # NB `(n_states,)`: the fit returns `(n_states, 1)` (#278).
        rates = state_vector(log_mu)
        dispersions = state_vector(alphas)
        probabilities = state_vector(p_binom)
        concentrations = state_vector(taus)

        n_states = rates.shape[0]
        lengths = current_clone_lengths(
            tuple(int(length) for length in np.asarray(clone_lengths)),
            int(np.asarray(decode).size),
        )

        # NB once per call, not per state; `(n_clones,)`, indexed by clone.
        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            stacked_log_lambda(normal_log_lambda, lengths),
            np.asarray(lengths, dtype=np.int64),
        )

        # NB the allele channel is untouched by the shift.
        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        baf_uniq = (
            scratch_baf[0] if scratch_baf else np.zeros((n_states, len(bb_endog)))
        )

        if self.emission_kernels == "sal":
            from port.patch.hmm_nophasing.dense_emission import bb_states

            baf_uniq[:] = bb_states(
                bb_endog,
                bb_exposure,
                probabilities,
                concentrations,
            )
        else:
            for state in range(n_states):
                _bb_logpmf_1d(
                    bb_endog,
                    bb_exposure,
                    probabilities[state],
                    concentrations[state],
                    baf_uniq[state, :],
                )

        log_emit_baf = bbEncoder.decode_array(baf_uniq, 0)

        # NB scored per unique `(clone, obs, total)`; each clone's block takes a scalar shift.
        triples = self._clone_triples(nbEncoder, lengths)
        rdr_uniq = np.zeros((n_states, triples.obs.size))

        for clone in range(len(lengths)):
            first, last = int(triples.bounds[clone]), int(triples.bounds[clone + 1])

            if first == last:
                continue

            if self.emission_kernels == "sal":
                from port.patch.hmm_nophasing.dense_emission import nb_states

                rdr_uniq[:, first:last] = nb_states(
                    triples.obs[first:last],
                    triples.total[first:last],
                    np.exp(rates - shifts[clone]),
                    dispersions,
                )
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


def shifted(model: Any) -> bool:
    """Whether `model`, a class or an instance, fits with the shift applied (#517)."""
    return bool(getattr(model, "apply_logmu_shift", False))


def release() -> None:
    """Drop the last fit's shift; called by `port.pipeline.patched` on exit (#517)."""
    hmm_nophasing._row_shift = None
