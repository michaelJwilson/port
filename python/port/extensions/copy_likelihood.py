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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import gammaln

__all__ = [
    "CopyFit",
    "Pseudobulk",
    "candidates",
    "capture",
    "captured_clones",
    "lattice_decode",
    "shared_decode",
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


@dataclass
class Pseudobulk:
    """One clone's summed counts and what the fit held fixed."""

    counts_nb: np.ndarray
    base_nb_mean: np.ndarray
    counts_bb: np.ndarray
    total_bb_rd: np.ndarray
    log_lambda: np.ndarray
    alpha: float
    tau: float


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
    """
    x = bulk.counts_nb[bins]
    exposure = bulk.base_nb_mean[bins]
    mean = exposure * np.exp(log_rate)

    with np.errstate(divide="ignore", invalid="ignore"):
        if bulk.alpha <= 0.0:
            depth = np.where(
                mean <= 0.0, 0.0, x * np.log(mean) - mean - gammaln(x + 1.0)
            )
        else:
            size = 1.0 / max(bulk.alpha, 1e-10)
            success = 1.0 / (1.0 + bulk.alpha * mean)
            depth = np.where(
                mean <= 0.0,
                0.0,
                gammaln(x + size)
                - gammaln(size)
                - gammaln(x + 1.0)
                + size * np.log(success)
                + x * np.log1p(-success),
            )

    k = bulk.counts_bb[bins]
    n = bulk.total_bb_rd[bins]
    choose = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)

    if not np.isfinite(bulk.tau):
        share = np.clip(p, 1e-10, 1.0 - 1e-10)
        allele = choose + k * np.log(share) + (n - k) * np.log1p(-share)
    else:
        a = np.maximum(p * bulk.tau, 1e-10)
        b = np.maximum((1.0 - p) * bulk.tau, 1e-10)
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
        bulk.total_bb_rd,
        bulk.log_lambda,
        alpha,
        tau,
    )


def _viterbi(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, n_obs)` emissions; the best path, restarted at each length."""
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
    alpha: float
    tau: float
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
    emission = np.stack(
        [_emission(log_mu[k] - shift, p[k], bulk, bins) for k in range(len(states))]
    )
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
    from scipy.optimize import minimize_scalar

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
        (0.0, np.inf) if dispersion == "poisson" else (bulks[0].alpha, bulks[0].tau)
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
                    shifts[i] = float(
                        minimize_scalar(
                            lambda s, i=i, fitted=fitted, path=paths[i]: -_on_path(
                                s, float(purity[i]), states, fitted, path
                            ),
                            bounds=(shifts[i] - 3.0, shifts[i] + 3.0),
                            method="bounded",
                        ).x
                    )

                if fit_purity:
                    purity[i] = float(
                        minimize_scalar(
                            lambda f, i=i, fitted=fitted: _best_path(
                                f, float(shifts[i]), states, fitted, chain, parsimony
                            ),
                            bounds=(0.05, 1.0),
                            method="bounded",
                        ).x
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
        bulks[0].alpha,
        bulks[0].tau,
        total,
    )


def captured_clones() -> list[tuple[np.ndarray, Pseudobulk, float]] | None:
    """Every captured clone's path, pseudobulk and shift, in the fit's order."""
    if not _CAPTURED:
        return None

    single_x, base, total, result = _CAPTURED[0]
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
            total_bb_rd=total[:, spots].sum(axis=1),
            log_lambda=np.log(profile / profile.sum()),
            alpha=alpha,
            tau=tau,
        )
        rows.append((path[:, clone], bulk, float(shifts[clone])))

    return rows


_CAPTURED: list[tuple[Any, Any, Any, Any]] = []
"""`(single_X, single_base_nb_mean, single_total_bb_RD, result)` of the last fit."""


@contextlib.contextmanager
def capture() -> Iterator[None]:
    """Keep the RDR+BAF fit's inputs and result, for the decoder that follows.

    Wraps `port`'s `run_core_inference`, the one the shift installs, as
    `tests/realizations.py` does, and keeps only the `params="smp"` call.
    """
    import port.patch.hmrf as patch

    original = patch.run_core_inference

    def keep(
        single_x: Any, lengths: Any, base: Any, total: Any, *rest: Any, **kw: Any
    ) -> Any:
        result = original(single_x, lengths, base, total, *rest, **kw)

        if kw.get("params") == "smp":
            _CAPTURED[:] = [
                (np.array(single_x), np.array(base), np.array(total), result)
            ]

        return result

    patch.run_core_inference = keep

    try:
        yield
    finally:
        patch.run_core_inference = original
        _CAPTURED.clear()
