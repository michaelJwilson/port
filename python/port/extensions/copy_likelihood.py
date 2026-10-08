"""Integer copies by the likelihood the HMM maximised (#327, #362).

Replaces `cnaster`'s L1 cost on fitted `(mu, p)` with :func:`lattice_decode`:
one HMM state per `(A, B)`, `A + B <= max_total_copy`, Viterbi per clone in an
EM fitting each clone's shift and tumour fraction `rho` and shared dispersions.
Depth `rho (A + B) / 2 + 1 - rho`, share `(rho A + 1 - rho) / (rho (A + B) + 2 (1 - rho))`,
log-prior `-parsimony |A + B - 2|` per bin. Alternatives: `port.sandbox.integer_decoding`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
from sal.opt.termination import Termination

from port.patch.emission import DISPERSION_FLOOR, bb_log_pmf, nb_log_pmf

__all__ = [
    "CopyFit",
    "Pseudobulk",
    "candidates",
    "capture",
    "captured_chain",
    "captured_clones",
    "captured_normal",
    "clones_of",
    "lattice_decode",
    "normal_of",
    "viterbi_oracle",
]

PARSIMONY = 0.5
"""Nats per bin per unit of `|A + B - 2|`: the prior :func:`lattice_decode` uses."""

ALPHA_BOUNDS = (np.log(1e-8), np.log(10.0))
"""Where `alpha` is searched, in logs: from Poisson to ten times overdispersed."""

TAU_BOUNDS = (0.0, np.log(1e8))
"""Where `tau` is searched, in logs: from binomial to a flat allele share."""

SHIFT_WINDOW = 0.35
"""How far the start moves a clone's shift: under `log 2`, since `(2A, 2B)` at `shift + log 2` is indistinguishable."""

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


def candidates(max_total_copy: int, max_allele_copy: int | None = None) -> np.ndarray:
    """Every `(A, B)` with `0 < A + B <= max_total_copy` and `A, B <= max_allele_copy`, `(n, 2)`, `A`-major."""
    allele = max_total_copy if max_allele_copy is None else max_allele_copy
    return np.array(
        [
            (a, b)
            for a in range(allele + 1)
            for b in range(allele + 1)
            if 0 < a + b <= max_total_copy
        ],
        dtype=np.int64,
    )


def pseudobulk_log_pmf(
    log_rate: np.ndarray, p: np.ndarray, bulk: Pseudobulk, bins: np.ndarray
) -> np.ndarray:
    """NB + BB log pmf per bin, via `port.patch.emission` (T- #776).

    `log_rate` and `p` broadcast against the bins (a leading state axis
    scores every state). At `tau = inf` the share is clipped to
    `[DISPERSION_FLOOR, 1 - DISPERSION_FLOOR]`, so a lost allele scores finitely.
    """
    x = bulk.counts_nb[bins]
    mean = bulk.base_nb_mean[bins] * np.exp(log_rate)
    depth = nb_log_pmf(x, bulk.dispersion, mean)

    k = bulk.counts_bb[bins]
    n = bulk.total_bb_RD[bins]
    share = (
        np.clip(p, DISPERSION_FLOOR, 1.0 - DISPERSION_FLOOR)
        if not np.isfinite(bulk.taus)
        else p
    )
    allele = bb_log_pmf(k, n, share, bulk.taus)

    return np.asarray(depth + allele)


def pair_rate_and_share(
    copies: np.ndarray, purity: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """`(log mu, p)` of each pair at tumour fraction `purity`; `p` is 0.5 with no copies at all."""
    total = copies.sum(axis=1).astype(np.float64)
    depth = purity * total / 2.0 + (1.0 - purity)
    alleles = purity * total + 2.0 * (1.0 - purity)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (purity * copies[:, 0] + (1.0 - purity)) / alleles
        return np.log(depth), np.where(alleles > 0, share, 0.5)


def with_dispersions(bulk: Pseudobulk, dispersion: float, taus: float) -> Pseudobulk:
    """`bulk` at another NB `dispersion` and beta-binomial concentration `taus`, its counts unchanged."""
    return Pseudobulk(
        bulk.counts_nb,
        bulk.base_nb_mean,
        bulk.counts_bb,
        bulk.total_bb_RD,
        bulk.normal_log_lambda,
        dispersion,
        taus,
    )


def viterbi_oracle(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, n_obs)` emissions; the best path, restarted at each length.

    The NumPy recursion :func:`_viterbi`'s sal kernel reproduces, kept as its oracle (#512, T- #632).
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


def _viterbi(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, n_obs)` emissions; the best path, restarted at each length.

    sal's `likelihood.ragged.viterbi` (sal #1138, T- #632), bitwise :func:`viterbi_oracle`.
    """
    from sal.likelihood.ragged import viterbi
    from sal.ragged import Ragged

    density = np.ascontiguousarray(np.asarray(log_emission, dtype=np.float64).T)
    sizes = np.asarray(lengths, dtype=np.int64)
    decoded = viterbi(
        Ragged(density, tuple(int(n) for n in sizes)),
        np.asarray(log_startprob, dtype=np.float64),
        np.asarray(log_transmat, dtype=np.float64),
    )

    total = 0.0
    for score in np.asarray(decoded.log_joint):
        total += float(score)
    return np.asarray(decoded.path, dtype=np.int64), total


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
    termination: Termination = field(kw_only=True)
    """Whether the last EM iteration left paths and parameters unchanged (T- #617)."""


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
    log_mu, p = pair_rate_and_share(states, purity)
    bins = np.arange(bulk.counts_nb.size)
    emission = pseudobulk_log_pmf((log_mu - shift)[:, None], p[:, None], bulk, bins)
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
    """A tumour clone's `(purity, shift)` jointly: the best shift within `window` per fraction on `grid`."""
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
    """The lowest of `current`, a bounded Brent search, and `grid`: never uphill (#371)."""
    from scipy.optimize import minimize_scalar

    found = minimize_scalar(objective, bounds=bounds, method="bounded")
    candidates = [current, float(found.x), *grid]
    return min(candidates, key=objective)


def _on_path(
    shift: float, purity: float, states: np.ndarray, bulk: Pseudobulk, path: np.ndarray
) -> float:
    """The clone's log-likelihood along `path`."""
    log_mu, p = pair_rate_and_share(states, purity)
    bins = np.arange(path.size)
    return float(np.sum(pseudobulk_log_pmf(log_mu[path] - shift, p[path], bulk, bins)))


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
            _on_path(float(s), float(f), states, with_dispersions(b, a, t), z)
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
    max_allele_copy: int | None = None,
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
    """Each clone's per-bin `(A, B)` (module docstring).

    `clones`: each clone's continuous path, pseudobulk and shift; `normal_clone`
    is held at shift 0 and fraction 1. Transitions: `stay` on the diagonal.
    Starts by :func:`_start`, then with `em` runs `iterations` of Viterbi
    E-step and an M-step (shift, fraction, `alpha`, `tau`) of up to
    `max_inner` rounds. `dispersion`: `"fit"`, `"held"`, or `"poisson"`. The
    flags are #362's simplifications; the defaults are the adopted decode.
    """
    states = candidates(max_total_copy, max_allele_copy)
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
                with_dispersions(bulk, alpha, tau),
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
                with_dispersions(bulk, alpha, tau),
                parsimony,
            )
            path, score = _viterbi(emission, transmat, start, chain[2])
            paths.append(path)
            total += score

        return paths, total

    paths, total = e_step()
    done, settled = 0, False

    for _ in range(iterations if em else 0):
        before_paths = [path.copy() for path in paths]
        stable = False

        for inner in range(max_inner):
            before = (shifts.copy(), purity.copy(), alpha, tau)

            for i, bulk in enumerate(bulks):
                if i == normal_clone:
                    continue

                fitted = with_dispersions(bulk, alpha, tau)

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
                stable = inner == 0
                break

        paths, total = e_step()
        done += 1
        settled = stable and all(
            np.array_equal(old, new)
            for old, new in zip(before_paths, paths, strict=True)
        )
        # NB no early exit on `settled`, so the decode stays bitwise.

    return CopyFit(
        [states[path] for path in paths],
        states,
        paths,
        shifts,
        purity,
        alpha,
        tau,
        total,
        termination=Termination.after(done, converged=settled),
    )


def captured_clones() -> list[tuple[np.ndarray, Pseudobulk, float]] | None:
    """Every captured clone's path, pseudobulk and shift, in the fit's order."""
    fit = captured_fit()

    if fit is None:
        return None

    return clones_of(fit)


class Captured(NamedTuple):
    """What `run_core_inference` was given, and what it returned."""

    single_X: np.ndarray
    lengths: np.ndarray
    single_base_nb_mean: np.ndarray
    single_total_bb_RD: np.ndarray
    res: Any


@contextlib.contextmanager
def captured_fits() -> Iterator[list[Captured]]:
    """Every `params="smp"` fit `port`'s `run_core_inference` returns in the block (#517).

    Enter **before** `patched`, which then installs the wrapper.
    """
    import port.patch.hmrf as patch

    kept: list[Captured] = []
    original = patch.run_core_inference

    def keep(
        single_X: Any,
        lengths: Any,
        base: Any,
        total: Any,
        *rest: Any,
        **kw: Any,
    ) -> Any:
        result = original(single_X, lengths, base, total, *rest, **kw)

        if kw.get("params") == "smp":
            kept.append(
                Captured(
                    np.array(single_X, dtype=np.float64),
                    np.asarray(lengths, dtype=np.int64),
                    np.array(base, dtype=np.float64),
                    np.array(total, dtype=np.float64),
                    result,
                )
            )

        return result

    patch.run_core_inference = keep

    try:
        yield kept
    finally:
        patch.run_core_inference = original


def clones_of(captured: Any) -> list[tuple[np.ndarray, Pseudobulk, float]]:
    """:func:`captured_clones` of a given fit (`Captured`)."""
    fit = captured
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
"""The fits each open `capture()` block collects (`captured_fits`)."""


def captured_fit() -> Any:
    """The last `params="smp"` fit of the innermost `capture()`, or `None`."""
    return _FITS[-1][-1] if _FITS and _FITS[-1] else None


def captured_normal() -> int | None:
    """The captured fit's normal clone: the largest share of balanced bins (#389).

    The same rule `core_inference.clone_shifts` zeroes a shift by.
    """
    fit = captured_fit()

    if fit is None:
        return None

    return normal_of(fit)


def normal_of(captured: Any) -> int:
    """:func:`captured_normal` of a given fit (`Captured`)."""
    from port.patch.hmm_nophasing.shifted_emission import normal_clone

    result = captured.res
    p_binom = np.asarray(result["new_p_binom"], dtype=np.float64).reshape(-1)
    path = np.asarray(result["pred_cnv"], dtype=np.int64)
    return normal_clone(p_binom, path.reshape(path.shape[0], -1) % p_binom.size)[0]


def captured_chain() -> tuple[np.ndarray | None, float]:
    """The captured fit's `(lengths, stay)`, `stay` the mean fitted diagonal (#370), else `1 - 1e-7`."""
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
    """Keep the RDR+BAF fit's inputs and result for :func:`captured_fit`."""
    with captured_fits() as kept:
        _FITS.append(kept)

        try:
            yield
        finally:
            _FITS.remove(kept)
