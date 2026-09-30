"""Integer copies by the likelihood the HMM maximised (#327, #362).

`cnaster` decodes a clone's integer copies with an L1 cost on its fitted
`(mu, p)` (`integer_copy.py`). Here they are fitted by the pseudobulk NB/BB
likelihood itself, on the same counts, the same normal baseline and each
clone's `logmu_shift`. Two entry points:

- :func:`lattice_decode`, the default: one HMM state per `(A, B)` with
  `A + B <= max_total_copy`, decoded per clone by Viterbi, in an EM whose
  M-step fits each clone's shift and tumour fraction and the shared
  dispersions. A tumour clone's spots are a fraction `rho` tumour and the
  rest normal: depth `rho (A + B) / 2 + 1 - rho`, allele share
  `(rho A + 1 - rho) / (rho (A + B) + 2 (1 - rho))`. A per-bin log-prior
  `-parsimony |A + B - 2|` decides among pairs the counts cannot separate:
  where read depth barely fixes the total, a fraction and a total trade, and
  `(0, 3)` at 0.78 has `(0, 1)`'s allele share at 0.92.
- :func:`shared_decode`, the pipeline's: the continuous fit's states and
  paths held, each state's pair the one maximizing the likelihood summed over
  every clone's bins in it. It is what `cnaster`'s per-state interface can
  carry (`port.patch.integer_copy`).

Measured on CalicoST's simulated samples (#362), pure and admixed, easy and
hard, with planted and fitted clones: the lattice decode is best on 6 of 8
fits by copy ARI and within 0.004 on the other 2, and scores 0.97-0.99 of
altered clone-bins exactly (phase-free) on the pure samples against about
0.6 on the admixed ones. What it was chosen over -- tempered E-steps, EMs
over the continuous states, fixed or relaxed dispersions, CalicoST's own
decoders -- is in `port.sandbox.integer_decoding`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, NamedTuple

import numpy as np
from scipy.special import gammaln, xlogy

from port.patch.hmm_nophasing.gradient import DISPERSION_FLOOR

__all__ = [
    "CopyFit",
    "Pseudobulk",
    "candidates",
    "capture",
    "captured_chain",
    "captured_clones",
    "captured_normal",
    "lattice_decode",
    "shared_decode",
    "viterbi_oracle",
]

PARSIMONY = 0.5
"""Nats per bin per unit of `|A + B - 2|`: the prior :func:`lattice_decode` uses."""

ALPHA_BOUNDS = (np.log(1e-8), np.log(10.0))
"""Where `alpha` is searched, in logs: from Poisson to ten times overdispersed."""

TAU_BOUNDS = (0.0, np.log(1e8))
"""Where `tau` is searched, in logs: from binomial to a flat allele share."""

SHIFT_WINDOW = 0.35
"""How far the start moves a clone's shift: less than `log 2`.

`(2A, 2B)` at `shift + log 2` has `(A, B)`'s depth and allele share exactly,
so a clone's ploidy is not identifiable under a free per-clone shift; a
window under `log 2` keeps the continuous fit's scale rather than doubling
it. On the pure easy fixture an unbounded search found the doubled genome.
"""

PURITY_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
"""Where the start looks for a clone's tumour fraction."""


class Pseudobulk(NamedTuple):
    """One clone's summed counts and what the fit held fixed."""

    counts_nb: np.ndarray
    base_nb_mean: np.ndarray
    counts_bb: np.ndarray
    total_bb_RD: np.ndarray
    normal_log_lambda: np.ndarray
    dispersion: float
    taus: float


def candidates(max_total_copy: int) -> np.ndarray:
    """Every `(A, B)` with `0 < A + B <= max_total_copy`, `(n, 2)`."""
    return np.array(
        [
            (a, b)
            for a in range(max_total_copy + 1)
            for b in range(max_total_copy + 1)
            if 0 < a + b <= max_total_copy
        ],
        dtype=np.int64,
    )


def _emission(
    log_rate: np.ndarray, p: np.ndarray, bulk: Pseudobulk, bins: np.ndarray
) -> np.ndarray:
    """NB + BB log pmf per bin, in `port.extensions.jax_hmm.emission`'s terms.

    `alpha = 0` is the Poisson and `tau = inf` the binomial, exactly.
    `log_rate` and `p` broadcast against the bins: a leading state axis gives
    every state's row at once. The terms that do not depend on the state are
    computed once, in the order the sum reads them, so each element is the
    per-state expression's bitwise (#512).

    The negative binomial is in log space with `a = alpha * mean`:
    `log p = -log1p(a)`, `log(1 - p) = log(a) - log1p(a)` (#560). Forming
    `p = 1 / (1 + a)` rounded `p` to 1 below `a` of about 1.1e-16, where a
    count of 0 scored NaN and a count of 1000 `-inf` (truth -43,420 at
    `a = 1e-19`), and lost digits of `log(1 - p)` below about 1e-4. `alpha`
    is floored in `a` as in `r`, as `port.patch.hmm_nophasing.nb_logpmf`
    does.
    """
    x = bulk.counts_nb[bins]
    exposure = bulk.base_nb_mean[bins]
    mean = exposure * np.exp(log_rate)

    with np.errstate(divide="ignore", invalid="ignore"):
        if bulk.dispersion <= 0.0:
            depth = np.where(
                mean <= 0.0, 0.0, x * np.log(mean) - mean - gammaln(x + 1.0)
            )
        else:
            dispersion = max(bulk.dispersion, DISPERSION_FLOOR)
            size = 1.0 / dispersion
            scaled = dispersion * mean
            fixed = gammaln(x + size) - gammaln(size) - gammaln(x + 1.0)
            depth = np.where(
                mean <= 0.0,
                0.0,
                fixed - (size + x) * np.log1p(scaled) + xlogy(x, scaled),
            )

    k = bulk.counts_bb[bins]
    n = bulk.total_bb_RD[bins]
    choose = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)

    if not np.isfinite(bulk.taus):
        share = np.clip(p, DISPERSION_FLOOR, 1.0 - DISPERSION_FLOOR)
        allele = choose + k * np.log(share) + (n - k) * np.log1p(-share)
    else:
        a = np.maximum(p * bulk.taus, DISPERSION_FLOOR)
        b = np.maximum((1.0 - p) * bulk.taus, DISPERSION_FLOOR)
        allele = (
            choose
            + gammaln(k + a)
            + gammaln(n - k + b)
            - gammaln(n + a + b)
            - (gammaln(a) + gammaln(b) - gammaln(a + b))
        )

    return np.asarray(depth + allele)


def _parameters(
    copies: np.ndarray, purity: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """`(log mu, p)` of each pair, in a spot `purity` tumour and the rest normal.

    Depth `purity (A + B) / 2 + (1 - purity)`; allele share
    `(purity A + 1 - purity) / (purity (A + B) + 2 (1 - purity))`, 0.5 where
    there are no copies at all.
    """
    total = copies.sum(axis=1).astype(np.float64)
    depth = purity * total / 2.0 + (1.0 - purity)
    alleles = purity * total + 2.0 * (1.0 - purity)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (purity * copies[:, 0] + (1.0 - purity)) / alleles
        return np.log(depth), np.where(alleles > 0, share, 0.5)


def _with(bulk: Pseudobulk, alpha: float, tau: float) -> Pseudobulk:
    return Pseudobulk(
        bulk.counts_nb,
        bulk.base_nb_mean,
        bulk.counts_bb,
        bulk.total_bb_RD,
        bulk.normal_log_lambda,
        alpha,
        tau,
    )


def viterbi_oracle(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, n_obs)` emissions; the best path, restarted at each length.

    The NumPy recursion :func:`_viterbi` compiles, kept as its oracle (#512).
    """
    path = np.empty(log_emission.shape[1], dtype=np.int64)
    total = 0.0
    start = 0

    for length in np.asarray(lengths, dtype=np.int64):
        stop = start + int(length)
        delta = log_startprob + log_emission[:, start]
        back = np.empty((stop - start, log_transmat.shape[0]), dtype=np.int64)

        for t in range(start + 1, stop):
            scores = delta[:, None] + log_transmat
            back[t - start] = np.argmax(scores, axis=0)
            delta = scores[back[t - start], np.arange(scores.shape[1])]
            delta = delta + log_emission[:, t]

        path[stop - 1] = int(np.argmax(delta))
        total += float(np.max(delta))

        for t in range(stop - 1, start, -1):
            path[t - 1] = back[t - start, path[t]]

        start = stop

    return path, total


@lru_cache(maxsize=1)
def _viterbi_kernel() -> Callable[..., float]:
    """:func:`viterbi_oracle`'s recursion, compiled once (#512).

    The same additions in the same order, and the first maximum on a tie as
    `np.argmax` takes it, so the path and the score are the oracle's bitwise.
    """
    from numba import njit

    @njit(cache=True)
    def kernel(
        log_emission: np.ndarray,
        log_transmat: np.ndarray,
        log_startprob: np.ndarray,
        lengths: np.ndarray,
        path: np.ndarray,
    ) -> float:
        n_states = log_transmat.shape[0]
        total = 0.0
        start = 0
        delta = np.empty(n_states)
        moved = np.empty(n_states)

        for length in lengths:
            stop = start + length
            back = np.empty((stop - start, n_states), dtype=np.int64)

            for j in range(n_states):
                delta[j] = log_startprob[j] + log_emission[j, start]

            for t in range(start + 1, stop):
                for j in range(n_states):
                    best = 0
                    top = delta[0] + log_transmat[0, j]

                    for i in range(1, n_states):
                        score = delta[i] + log_transmat[i, j]
                        if score > top:
                            top = score
                            best = i

                    back[t - start, j] = best
                    moved[j] = top + log_emission[j, t]

                for j in range(n_states):
                    delta[j] = moved[j]

            last = 0
            for j in range(1, n_states):
                if delta[j] > delta[last]:
                    last = j

            path[stop - 1] = last
            total += delta[last]

            for t in range(stop - 1, start, -1):
                path[t - 1] = back[t - start, path[t]]

            start = stop

        return total

    return kernel


def _viterbi(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, n_obs)` emissions; the best path, restarted at each length.

    Compiled (:func:`_viterbi_kernel`); :func:`viterbi_oracle` is the NumPy
    recursion it reproduces bitwise.
    """
    path = np.empty(log_emission.shape[1], dtype=np.int64)
    total = _viterbi_kernel()(
        np.ascontiguousarray(log_emission, dtype=np.float64),
        np.ascontiguousarray(log_transmat, dtype=np.float64),
        np.ascontiguousarray(log_startprob, dtype=np.float64),
        np.asarray(lengths, dtype=np.int64),
        path,
    )
    return path, float(total)


@dataclass
class CopyFit:
    """Each clone's per-bin `(A, B)`, and what was fitted to reach them."""

    pairs: list[np.ndarray]
    """Per clone, `(n_obs, 2)`."""
    states: np.ndarray
    """`(n_states, 2)`: each state's pair."""
    paths: list[np.ndarray]
    shifts: np.ndarray
    purity: np.ndarray
    dispersion: float
    taus: float
    log_likelihood: float


def _prior(states: np.ndarray, parsimony: float) -> np.ndarray:
    """Each state's log-prior per bin: `-parsimony |A + B - 2|`."""
    return -parsimony * np.abs(states.sum(axis=1) - 2).astype(np.float64)


def _log_emissions(
    states: np.ndarray,
    shift: float,
    purity: float,
    bulk: Pseudobulk,
    parsimony: float,
) -> np.ndarray:
    """`(n_states, n_obs)` plus the prior, `-1e10` where a state cannot emit."""
    log_mu, p = _parameters(states, purity)
    bins = np.arange(bulk.counts_nb.size)
    emission = _emission((log_mu - shift)[:, None], p[:, None], bulk, bins)
    emission = np.where(np.isfinite(emission), emission, -1e10)
    return np.asarray(emission + _prior(states, parsimony)[:, None])


def _best_path(
    purity: float,
    shift: float,
    states: np.ndarray,
    bulk: Pseudobulk,
    chain: tuple[np.ndarray, np.ndarray, np.ndarray],
    parsimony: float,
) -> float:
    """Minus the clone's best path's log-likelihood (Viterbi's) at `purity`, `shift`."""
    log_transmat, log_startprob, lengths = chain
    emission = _log_emissions(states, shift, purity, bulk, parsimony)
    return -_viterbi(emission, log_transmat, log_startprob, lengths)[1]


def _start(
    states: np.ndarray,
    shift: float,
    bulk: Pseudobulk,
    chain: tuple[np.ndarray, np.ndarray, np.ndarray],
    parsimony: float,
    grid: tuple[float, ...] = PURITY_GRID,
    window: float = SHIFT_WINDOW,
) -> tuple[float, float]:
    """A tumour clone's `(purity, shift)` jointly, before any E-step.

    The fraction and the shift trade against each other, so fitting either
    alone from a wrong start settles wherever the first E-step left the
    paths: each fraction on :data:`PURITY_GRID` gets its own best shift
    within :data:`SHIFT_WINDOW`, and the best pair starts the EM.
    """
    from scipy.optimize import minimize_scalar

    found = []

    for fraction in grid:
        if window <= 0.0:
            value = _best_path(fraction, shift, states, bulk, chain, parsimony)
            found.append((value, fraction, shift))
            continue

        result = minimize_scalar(
            lambda s, f=fraction: _best_path(f, s, states, bulk, chain, parsimony),
            bounds=(shift - window, shift + window),
            method="bounded",
        )
        found.append((float(result.fun), fraction, float(result.x)))

    _, fraction, shift = min(found)
    return fraction, shift


def _monotone(
    objective: Callable[[float], float],
    current: float,
    bounds: tuple[float, float],
    grid: tuple[float, ...] = (),
) -> float:
    """The lowest of `current`, a bounded Brent search, and `grid`: never uphill.

    The fraction's objective is a best path, piecewise in the fraction and
    not convex, and a bounded search never scores its endpoints: on the
    critical instance it returned `0.20` at 11,424 against `1.0`'s 11,078
    (#371). Scoring the current value and the grid beside it keeps each
    M-step from raising the objective, and reaches fraction 1 exactly.
    """
    from scipy.optimize import minimize_scalar

    found = minimize_scalar(objective, bounds=bounds, method="bounded")
    candidates = [current, float(found.x), *grid]
    return min(candidates, key=objective)


def _on_path(
    shift: float, purity: float, states: np.ndarray, bulk: Pseudobulk, path: np.ndarray
) -> float:
    """The clone's log-likelihood along `path`."""
    log_mu, p = _parameters(states, purity)
    bins = np.arange(path.size)
    return float(np.sum(_emission(log_mu[path] - shift, p[path], bulk, bins)))


def _dispersions(
    alpha: float,
    tau: float,
    states: np.ndarray,
    paths: list[np.ndarray],
    bulks: list[Pseudobulk],
    shifts: np.ndarray,
    purity: np.ndarray,
) -> tuple[float, float]:
    """The shared `alpha`, then `tau`, maximizing the likelihood along the paths."""
    from scipy.optimize import minimize_scalar

    def total(a: float, t: float) -> float:
        return sum(
            _on_path(float(s), float(f), states, _with(b, a, t), z)
            for z, b, s, f in zip(paths, bulks, shifts, purity, strict=True)
        )

    alpha = float(
        np.exp(
            minimize_scalar(
                lambda x: -total(float(np.exp(x)), tau),
                bounds=ALPHA_BOUNDS,
                method="bounded",
            ).x
        )
    )
    tau = float(
        np.exp(
            minimize_scalar(
                lambda x: -total(alpha, float(np.exp(x))),
                bounds=TAU_BOUNDS,
                method="bounded",
            ).x
        )
    )
    return alpha, tau


def lattice_decode(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    *,
    normal_clone: int,
    max_total_copy: int,
    lengths: np.ndarray | None = None,
    stay: float = 1.0 - 1e-7,
    parsimony: float = PARSIMONY,
    fit_purity: bool = True,
    fit_shifts: bool = True,
    dispersion: Literal["fit", "held", "poisson"] = "fit",
    em: bool = True,
    iterations: int = 5,
    max_inner: int = 10,
) -> CopyFit:
    """Each clone's per-bin `(A, B)`: the default decode (module docstring).

    `clones` holds each clone's continuous path (read for its length),
    pseudobulk and shift, in the fit's order; `normal_clone` is held at
    shift 0 and fraction 1. Transitions are `stay` on the diagonal and the
    rest even.

    Start: each tumour clone's fraction (on :data:`PURITY_GRID`, or 1
    without `fit_purity`) and shift (within :data:`SHIFT_WINDOW`, or the
    continuous fit's without `fit_shifts`), jointly, by :func:`_start`. Then,
    with `em`, `iterations` times: each clone's Viterbi path (E-step); then
    its shift and fraction and the shared `alpha` and `tau`, until none moves
    or `max_inner` times (M-step). Without `em`, one Viterbi pass at the
    start. `dispersion`: `"fit"` in the M-step, `"held"` at the continuous
    fit's, `"poisson"` at the Poisson and binomial limits.

    The flags are the simplifications the #362 audit measured; the defaults
    are the decode it adopted.
    """
    states = candidates(max_total_copy)
    n = len(states)
    transmat = np.log(
        np.full((n, n), (1.0 - stay) / (n - 1))
        + np.eye(n) * (stay - (1.0 - stay) / (n - 1))
    )
    start = np.full(n, -np.log(n))
    bulks = [bulk for _, bulk, _ in clones]
    shifts = np.array([shift for _, _, shift in clones], dtype=np.float64)
    shifts[normal_clone] = 0.0
    purity = np.ones(len(clones))
    alpha, tau = (
        (0.0, np.inf)
        if dispersion == "poisson"
        else (bulks[0].dispersion, bulks[0].taus)
    )
    lengths = np.array([clones[0][0].size]) if lengths is None else lengths
    chain = (transmat, start, np.asarray(lengths))
    grid = PURITY_GRID if fit_purity else (1.0,)

    for i, bulk in enumerate(bulks):
        if i != normal_clone and (fit_purity or fit_shifts):
            purity[i], shifts[i] = _start(
                states,
                float(shifts[i]),
                _with(bulk, alpha, tau),
                chain,
                parsimony,
                grid,
                SHIFT_WINDOW if fit_shifts else 0.0,
            )

    def e_step() -> tuple[list[np.ndarray], float]:
        paths, total = [], 0.0

        for i, bulk in enumerate(bulks):
            emission = _log_emissions(
                states,
                float(shifts[i]),
                float(purity[i]),
                _with(bulk, alpha, tau),
                parsimony,
            )
            path, score = _viterbi(emission, transmat, start, chain[2])
            paths.append(path)
            total += score

        return paths, total

    paths, total = e_step()

    for _ in range(iterations if em else 0):
        for _ in range(max_inner):
            before = (shifts.copy(), purity.copy(), alpha, tau)

            for i, bulk in enumerate(bulks):
                if i == normal_clone:
                    continue

                fitted = _with(bulk, alpha, tau)

                if fit_shifts:

                    def off_path(
                        shift: float,
                        fraction: float = float(purity[i]),
                        fitted: Pseudobulk = fitted,
                        path: np.ndarray = paths[i],
                    ) -> float:
                        return -_on_path(shift, fraction, states, fitted, path)

                    shifts[i] = _monotone(
                        off_path,
                        float(shifts[i]),
                        (float(shifts[i]) - 3.0, float(shifts[i]) + 3.0),
                    )

                if fit_purity:

                    def best(
                        fraction: float,
                        shift: float = float(shifts[i]),
                        fitted: Pseudobulk = fitted,
                    ) -> float:
                        return _best_path(
                            fraction, shift, states, fitted, chain, parsimony
                        )

                    purity[i] = _monotone(
                        best, float(purity[i]), (0.05, 1.0), PURITY_GRID
                    )

            if dispersion == "fit":
                alpha, tau = _dispersions(
                    alpha, tau, states, paths, bulks, shifts, purity
                )

            if (
                np.allclose(shifts, before[0], atol=1e-4)
                and np.allclose(purity, before[1], atol=1e-4)
                and np.isclose(alpha, before[2], rtol=1e-3)
                and np.isclose(tau, before[3], rtol=1e-3)
            ):
                break

        paths, total = e_step()

    return CopyFit(
        [states[path] for path in paths],
        states,
        paths,
        shifts,
        purity,
        alpha,
        tau,
        total,
    )


def shared_decode(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    *,
    n_states: int,
    normal: int,
    max_total_copy: int,
) -> CopyFit:
    """Each continuous state's `(A, B)`, one pair shared by every clone.

    The continuous paths, shifts and dispersions are held, so the
    likelihood is a sum over states of terms each depending on one state's
    pair, and each state's argmax over the lattice solves the one-pair-per-
    state MILP exactly. `normal` is `(1, 1)`; a state no clone visits is too.
    """
    lattice = candidates(max_total_copy)
    log_mu, p = _parameters(lattice)
    states = np.ones((n_states, 2), dtype=np.int64)
    paths = [np.asarray(path, dtype=np.int64) for path, _, _ in clones]
    total = 0.0

    for k in np.unique(np.concatenate(paths)):
        state = int(k)
        scores = np.zeros(len(lattice))

        for path, (_, bulk, shift) in zip(paths, clones, strict=True):
            bins = np.flatnonzero(path == state)

            if bins.size:
                scores += np.array(
                    [
                        np.sum(_emission(log_mu[i] - shift, p[i], bulk, bins))
                        for i in range(len(lattice))
                    ]
                )

        best = (
            int(np.flatnonzero((lattice[:, 0] == 1) & (lattice[:, 1] == 1))[0])
            if state == normal
            else int(np.argmax(scores))
        )
        states[state] = lattice[best]
        total += float(scores[best])

    bulks = [bulk for _, bulk, _ in clones]
    return CopyFit(
        [states[path] for path in paths],
        states,
        paths,
        np.array([shift for _, _, shift in clones], dtype=np.float64),
        np.ones(len(clones)),
        bulks[0].dispersion,
        bulks[0].taus,
        total,
    )


def captured_clones() -> list[tuple[np.ndarray, Pseudobulk, float]] | None:
    """Every captured clone's path, pseudobulk and shift, in the fit's order."""
    fit = captured_fit()

    if fit is None:
        return None

    single_x, base, total, result = (
        fit.single_X,
        fit.single_base_nb_mean,
        fit.single_total_bb_RD,
        fit.res,
    )
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    log_mu = np.asarray(result["new_log_mu"], dtype=np.float64).reshape(-1)
    path = np.asarray(result["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1) % log_mu.size

    try:
        shifts = np.asarray(result["new_log_mu_shift"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError):
        shifts = np.zeros(path.shape[1])

    if shifts.size != path.shape[1]:
        shifts = np.zeros(path.shape[1])

    profile = base.sum(axis=1)
    alpha = float(np.asarray(result["new_alphas"]).reshape(-1)[0])
    tau = float(np.asarray(result["new_taus"]).reshape(-1)[0])
    rows = []

    for clone in range(path.shape[1]):
        spots = assignment == clone
        bulk = Pseudobulk(
            counts_nb=single_x[:, 0, spots].sum(axis=1),
            base_nb_mean=base[:, spots].sum(axis=1),
            counts_bb=single_x[:, 1, spots].sum(axis=1),
            total_bb_RD=total[:, spots].sum(axis=1),
            normal_log_lambda=np.log(profile / profile.sum()),
            dispersion=alpha,
            taus=tau,
        )
        rows.append((path[:, clone], bulk, float(shifts[clone])))

    return rows


_FITS: list[list[Any]] = []
"""The fits each open `capture()` block collects (`copy_errors.captured_fits`)."""


def captured_fit() -> Any:
    """The last `params="smp"` fit of the innermost `capture()`, or `None`."""
    return _FITS[-1][-1] if _FITS and _FITS[-1] else None


def captured_normal() -> int | None:
    """The captured fit's normal clone: the largest share of balanced bins (#389).

    The rule `run_core_inference` zeroes a clone's shift by
    (`core_inference.clone_shifts`), so the clone the decode holds at
    `(1, 1)`, shift 0 and fraction 1 is the one whose shift was zeroed.
    `argmin |shift|`, the rule it replaces, ties among every clone near
    diploid: on CalicoST easy and hard under `--sal` it named a tumour clone,
    held it at fraction 1 and shift 0, and decoded its LOH bins as `(1, 5)`.
    """
    from port.patch.hmm_nophasing.shifted_emission import NEUTRAL_BAF_TOLERANCE

    fit = captured_fit()

    if fit is None:
        return None

    result = fit.res
    p_binom = np.asarray(result["new_p_binom"], dtype=np.float64).reshape(-1)
    path = np.asarray(result["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1) % p_binom.size
    balanced = np.abs(p_binom - 0.5) <= NEUTRAL_BAF_TOLERANCE

    return int(np.argmax(balanced[path].mean(axis=0)))


def captured_chain() -> tuple[np.ndarray | None, float]:
    """The captured fit's `(lengths, stay)`: the chain `lattice_decode` runs on.

    `stay` is the mean of the fitted transition matrix's diagonal, as #370's
    measurements took it; `1 - 1e-7`, `lattice_decode`'s default, without a
    captured fit.
    """
    fit = captured_fit()

    if fit is None:
        return None, 1.0 - 1e-7

    result = fit.res
    lengths = fit.lengths

    try:
        transmat = np.asarray(result["new_log_transmat"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return lengths, 1.0 - 1e-7

    diagonal = np.diagonal(transmat.reshape(-1, *transmat.shape[-2:])[0])
    return lengths, float(np.exp(diagonal).mean())


@contextlib.contextmanager
def capture() -> Iterator[None]:
    """Keep the RDR+BAF fit's inputs and result, for the decoder that follows.

    `port.extensions.copy_errors.captured_fits` for the block; the decode
    reads the last `params="smp"` fit through :func:`captured_fit`.
    """
    from port.extensions.copy_errors import captured_fits

    with captured_fits() as kept:
        _FITS.append(kept)

        try:
            yield
        finally:
            _FITS.remove(kept)
