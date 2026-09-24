"""Integer copies by the likelihood the HMM maximised (#327, #362).

`cnaster` decodes a clone's integer copies with an L1 cost on its fitted
`(mu, p)` (`integer_copy.py`). Here they are fitted by the pseudobulk NB/BB
likelihood itself, on the same counts, the same normal baseline and each
clone's `logmu_shift`. One entry point, :func:`fit_copies`; what it does is
set by a :class:`Scheme`'s flags:

- `states`: `"fit"` keeps the continuous fit's states and chooses each one's
  `(A, B)`, shared by every clone, in the M-step; `"lattice"` has one state
  per `(A, B)` with `A + B <= max_total_copy`, so the pairs are fixed and the
  path chooses among them;
- `temperatures`: one EM iteration each. The E-step is forward-backward with
  log emissions and transitions at `1 / T`, so as `T -> 0` each bin's
  responsibility concentrates on one state, the Viterbi path's; `T = 0` is
  Viterbi itself. Empty: no E-step, the continuous fit's paths held;
- `poisson`: the dispersions fixed at the Poisson and binomial limits
  (`alpha = 0`, `tau = inf`) rather than fitted;
- `fit_purity`: each tumour clone's spots are a fraction `rho` tumour and the
  rest normal. Depth `rho (A + B) / 2 + 1 - rho`, allele share
  `(rho A + 1 - rho) / (rho (A + B) + 2 (1 - rho))`, `rho` fitted to the
  marginal likelihood (the best path's, for `T = 0`). The normal clone's
  `rho` is 1.

The M-step fits, in order, each clone's `rho` (with `fit_purity`) and shift
(the normal clone's held at 0), the shared dispersions `alpha` and `tau`,
and, for `"fit"` states, each state's pair. With paths and everything else
held, the likelihood is a sum over states of terms each depending on one
state's pair, so the one-pair-per-state MILP separates and each state's
argmax solves it exactly; `distinct` adds that no two states share a pair,
an assignment problem (Hungarian). Both constraint sets are totally
unimodular, so no branching is needed. The normal state is `(1, 1)`.

:data:`SHARED` is the pipeline's decode (`port.patch.integer_copy`);
:data:`VITERBI` and :data:`TEMPERED` are the two EMs over every pair.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = [
    "SHARED",
    "TEMPERED",
    "VITERBI",
    "CopyFit",
    "Pseudobulk",
    "Scheme",
    "candidates",
    "capture",
    "captured_clones",
    "fit_copies",
]


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


@dataclass(frozen=True)
class Scheme:
    """How :func:`fit_copies` fits; see the module docstring."""

    states: Literal["fit", "lattice"] = "fit"
    temperatures: tuple[float, ...] = ()
    fit_purity: bool = False
    distinct: bool = False
    poisson: bool = False


SHARED = Scheme()
"""The continuous fit's states and paths, each state's pair shared by every clone."""

VITERBI = Scheme(states="lattice", temperatures=(0.0,) * 5, fit_purity=True)
"""One state per pair, hard EM: Viterbi E-steps, `rho` to the best path."""

TEMPERED = Scheme(
    states="lattice", temperatures=(1.0, 0.5, 0.25, 0.1, 0.05), fit_purity=True
)
"""One state per pair, responsibilities tempered towards one state per bin."""


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


def _forward_backward(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """`(n_states, n_obs)` responsibilities with everything scaled by `1 / T`.

    At `T = 1` the HMM's posteriors; as `T -> 0` they concentrate on the
    Viterbi path, one state per bin.
    """
    inverse = 1.0 / temperature
    emission = log_emission * inverse
    transmat = log_transmat * inverse
    start_prob = log_startprob * inverse
    gamma = np.empty_like(emission)
    begin = 0

    for length in np.asarray(lengths, dtype=np.int64):
        stop = begin + int(length)
        n = stop - begin
        forward = np.empty((n, emission.shape[0]))
        backward = np.zeros((n, emission.shape[0]))
        forward[0] = start_prob + emission[:, begin]

        for t in range(1, n):
            forward[t] = (
                logsumexp(forward[t - 1][:, None] + transmat, axis=0)
                + emission[:, begin + t]
            )

        for t in range(n - 2, -1, -1):
            backward[t] = logsumexp(
                transmat + (emission[:, begin + t + 1] + backward[t + 1])[None, :],
                axis=1,
            )

        joint = forward + backward
        gamma[:, begin:stop] = np.exp(joint - logsumexp(joint, axis=1)[:, None]).T
        begin = stop

    return gamma


def _weighted(
    log_mu: np.ndarray,
    p: np.ndarray,
    shift: float,
    bulk: Pseudobulk,
    gamma: np.ndarray,
    states: np.ndarray,
) -> float:
    """`sum_g sum_k gamma_kg log f(x_g | k)` over the states carrying weight."""
    bins = np.arange(gamma.shape[1])
    return float(
        sum(
            np.sum(gamma[k] * _emission(log_mu[k] - shift, p[k], bulk, bins))
            for k in states
        )
    )


def _negative_marginal(
    purity: float,
    lattice: np.ndarray,
    shift: float,
    bulk: Pseudobulk,
    log_transmat: np.ndarray | None,
    log_startprob: np.ndarray | None,
    lengths: np.ndarray,
    best: bool = False,
) -> float:
    """Minus the clone's forward log-likelihood, every path summed, at `purity`.

    With `best`, the best path's alone (Viterbi's), for the hard scheme.

    The tumour fraction is shared by every bin, so it is fitted to the
    marginal rather than to responsibilities computed at the old fraction,
    which pin it: a bin placed at `(1, 5)` under a pure fit only loses
    likelihood as the fraction falls, though `(0, 1)` would then fit it.
    """
    if log_transmat is None or log_startprob is None:
        msg = "a tumour fraction is fitted to the marginal, which needs transitions"
        raise ValueError(msg)

    log_mu, p = _parameters(lattice, purity)
    bins = np.arange(bulk.counts_nb.size)
    emission = np.stack(
        [_emission(log_mu[k] - shift, p[k], bulk, bins) for k in range(len(lattice))]
    )
    emission = np.where(np.isfinite(emission), emission, -1e10)
    total = 0.0
    begin = 0

    for length in np.asarray(lengths, dtype=np.int64):
        stop = begin + int(length)
        forward = log_startprob + emission[:, begin]

        for t in range(begin + 1, stop):
            step = forward[:, None] + log_transmat
            reduced = step.max(axis=0) if best else logsumexp(step, axis=0)
            forward = reduced + emission[:, t]

        total += float(forward.max() if best else logsumexp(forward))
        begin = stop

    return -total


def _negative_shift_weighted(
    shift: float,
    log_mu: np.ndarray,
    p: np.ndarray,
    bulk: Pseudobulk,
    gamma: np.ndarray,
    states: np.ndarray,
) -> float:
    return -_weighted(log_mu, p, shift, bulk, gamma, states)


def _negative_dispersion_weighted(
    log_value: float,
    which: str,
    other: float,
    bulks: list[Pseudobulk],
    lattice: np.ndarray,
    purity: np.ndarray,
    shifts: np.ndarray,
    gammas: list[np.ndarray],
    floor: float,
) -> float:
    value = float(np.exp(log_value))
    alpha, tau = (value, other) if which == "alpha" else (other, value)
    total = 0.0

    for i, bulk in enumerate(bulks):
        log_mu, p = _parameters(lattice, float(purity[i]))
        states = np.flatnonzero(gammas[i].max(axis=1) > floor)
        total += _weighted(
            log_mu, p, float(shifts[i]), _with(bulk, alpha, tau), gammas[i], states
        )

    return -total


def _shared_pairs(
    paths: list[np.ndarray],
    bulks: list[Pseudobulk],
    shifts: np.ndarray,
    purity: np.ndarray,
    *,
    n_states: int,
    lattice: np.ndarray,
    normal: int,
    distinct: bool,
) -> np.ndarray:
    """Each state's pair, shared by every clone, paths and the rest held."""
    rates = [_parameters(lattice, float(f)) for f in purity]
    one = (lattice[:, 0] == 1) & (lattice[:, 1] == 1)
    copies = np.ones((n_states, 2), dtype=np.int64)
    table: dict[int, np.ndarray] = {}

    for k in np.unique(np.concatenate(paths)):
        state = int(k)

        if state == normal:
            continue

        scores = np.zeros(len(lattice))

        for path, bulk, shift, (log_mu, p) in zip(
            paths, bulks, shifts, rates, strict=True
        ):
            bins = np.flatnonzero(path == state)

            if bins.size:
                scores += np.array(
                    [
                        np.sum(_emission(log_mu[i] - shift, p[i], bulk, bins))
                        for i in range(len(lattice))
                    ]
                )

        table[state] = scores
        copies[state] = lattice[int(np.argmax(scores))]

    if distinct and table:
        from scipy.optimize import linear_sum_assignment

        order = sorted(table)
        matrix = np.array([np.where(one, -np.inf, table[k]) for k in order])
        rows, columns = linear_sum_assignment(
            np.where(np.isfinite(matrix), -matrix, 1e300)
        )

        for row, column in zip(rows, columns, strict=True):
            copies[order[row]] = lattice[column]

    return copies


def _log_emissions(
    states: np.ndarray, shift: float, purity: float, bulk: Pseudobulk
) -> np.ndarray:
    """`(n_states, n_obs)`, `-1e10` where a state cannot emit the counts."""
    log_mu, p = _parameters(states, purity)
    bins = np.arange(bulk.counts_nb.size)
    emission = np.stack(
        [_emission(log_mu[k] - shift, p[k], bulk, bins) for k in range(len(states))]
    )
    return np.where(np.isfinite(emission), emission, -1e10)


PURITY_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
"""Where :func:`_profile_start` looks for a clone's tumour fraction."""


def _profile_start(
    states: np.ndarray,
    shift: float,
    bulk: Pseudobulk,
    log_transmat: np.ndarray | None,
    log_startprob: np.ndarray | None,
    lengths: np.ndarray,
    *,
    best: bool,
    fix_shift: bool,
) -> tuple[float, float]:
    """A clone's `(purity, shift)` jointly, before any E-step.

    The fraction and the shift trade against each other -- a lower fraction
    at a lower shift reads as the same depth -- so fitting either alone from
    a wrong start settles wherever the first E-step left the paths. So each
    fraction on :data:`PURITY_GRID` gets its own best shift against the
    marginal (every path summed; the best path's with `best`), and the best
    pair starts the EM. On a pure planted clone the marginal is maximal at 1
    by 1,000 nats over 0.4 (`tests/test_integer_em.py`).
    """
    from scipy.optimize import minimize_scalar

    found: list[tuple[float, float, float]] = []

    for fraction in PURITY_GRID:
        if fix_shift:
            value = _negative_marginal(
                fraction,
                states,
                shift,
                bulk,
                log_transmat,
                log_startprob,
                lengths,
                best,
            )
            found.append((value, fraction, shift))
            continue

        result = minimize_scalar(
            lambda s, f=fraction: _negative_marginal(
                f, states, s, bulk, log_transmat, log_startprob, lengths, best
            ),
            bounds=(shift - 1.5, shift + 1.5),
            method="bounded",
        )
        found.append((float(result.fun), fraction, float(result.x)))

    _, fraction, shift = min(found)
    return fraction, shift


def fit_copies(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    scheme: Scheme = SHARED,
    *,
    n_states: int,
    normal: int,
    normal_clone: int,
    max_total_copy: int,
    log_transmat: np.ndarray | None = None,
    log_startprob: np.ndarray | None = None,
    lengths: np.ndarray | None = None,
    stay: float = 1.0 - 1e-6,
    zero_normal: bool = True,
    floor: float = 1e-6,
    max_inner: int = 10,
) -> CopyFit:
    """Each clone's integer copies under `scheme` (see the module docstring).

    `clones` holds each clone's continuous path, pseudobulk and shift, in
    the fit's order; `normal` is the continuous fit's normal state and
    `normal_clone` the normal clone. `log_transmat`, `log_startprob` and
    `lengths` are the fit's, for an E-step over its states; over the lattice
    the transitions are `stay` on the diagonal, the rest even. Each M-step
    iterates its blocks -- shifts, then tumour fractions, then dispersions,
    then pairs -- until none moves, at most `max_inner` times.
    """
    lattice = candidates(max_total_copy)
    paths = [np.asarray(path, dtype=np.int64) for path, _, _ in clones]
    bulks = [bulk for _, bulk, _ in clones]
    shifts = np.array([shift for _, _, shift in clones], dtype=np.float64)
    purity = np.ones(len(clones))
    alpha, tau = (0.0, np.inf) if scheme.poisson else (bulks[0].alpha, bulks[0].tau)
    lengths = np.array([paths[0].size]) if lengths is None else np.asarray(lengths)

    if scheme.states == "fit":
        states = _shared_pairs(
            paths,
            [_with(b, alpha, tau) for b in bulks],
            shifts,
            purity,
            n_states=n_states,
            lattice=lattice,
            normal=normal,
            distinct=scheme.distinct,
        )
        states[normal] = (1, 1)
        transmat, start = log_transmat, log_startprob
        gammas: list[np.ndarray] | None = [np.eye(n_states)[:, path] for path in paths]
    else:
        states = lattice
        n = len(lattice)
        transmat = np.log(
            np.full((n, n), (1.0 - stay) / (n - 1))
            + np.eye(n) * (stay - (1.0 - stay) / (n - 1))
        )
        start = np.full(n, -np.log(n))
        gammas = None

    msg = "an E-step over the fit's states needs its log_transmat and log_startprob"

    if (scheme.temperatures or scheme.fit_purity) and (
        transmat is None or start is None
    ):
        raise ValueError(msg)

    fit = _Blocks(
        scheme,
        bulks,
        lattice,
        n_states,
        normal,
        normal_clone,
        zero_normal,
        floor,
        max_inner,
        transmat,
        start,
        lengths,
    )

    if scheme.fit_purity:
        for i, bulk in enumerate(bulks):
            if i != normal_clone:
                purity[i], shifts[i] = _profile_start(
                    states,
                    float(shifts[i]),
                    _with(bulk, alpha, tau),
                    transmat,
                    start,
                    lengths,
                    best=bool(scheme.temperatures) and scheme.temperatures[0] == 0.0,
                    fix_shift=zero_normal and i == normal_clone,
                )

    if gammas is not None and scheme.temperatures:
        states, alpha, tau = fit.m_step(
            gammas, states, shifts, purity, alpha, tau, scheme.temperatures[0] == 0.0
        )

    for temperature in scheme.temperatures:
        if transmat is None or start is None:  # pragma: no cover -- refused above
            raise ValueError(msg)

        gammas = []

        for i, bulk in enumerate(bulks):
            emission = _log_emissions(
                states, float(shifts[i]), float(purity[i]), _with(bulk, alpha, tau)
            )

            if temperature == 0.0:
                path, _ = _viterbi(emission, transmat, start, lengths)
                hard = np.zeros_like(emission)
                hard[path, np.arange(path.size)] = 1.0
                gammas.append(hard)
            else:
                gammas.append(
                    _forward_backward(emission, transmat, start, lengths, temperature)
                )

        paths = [np.argmax(gamma, axis=0) for gamma in gammas]
        states, alpha, tau = fit.m_step(
            gammas, states, shifts, purity, alpha, tau, temperature == 0.0
        )

    total = 0.0

    for path, bulk, shift, f in zip(paths, bulks, shifts, purity, strict=True):
        log_mu, p = _parameters(states, float(f))
        total += float(
            np.sum(
                _emission(
                    log_mu[path] - shift,
                    p[path],
                    _with(bulk, alpha, tau),
                    np.arange(path.size),
                )
            )
        )

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


@dataclass
class _Blocks:
    """The M-step's blocks, and what they hold fixed."""

    scheme: Scheme
    bulks: list[Pseudobulk]
    lattice: np.ndarray
    n_states: int
    normal: int
    normal_clone: int
    zero_normal: bool
    floor: float
    max_inner: int
    transmat: np.ndarray | None
    start: np.ndarray | None
    lengths: np.ndarray

    def m_step(
        self,
        gammas: list[np.ndarray],
        states: np.ndarray,
        shifts: np.ndarray,
        purity: np.ndarray,
        alpha: float,
        tau: float,
        best: bool,
    ) -> tuple[np.ndarray, float, float]:
        """Shifts, fractions, dispersions and pairs, in turn, until none moves.

        `shifts` and `purity` are updated in place.
        """
        from scipy.optimize import minimize_scalar

        paths = [np.argmax(gamma, axis=0) for gamma in gammas]

        for _ in range(self.max_inner):
            before = (states.copy(), shifts.copy(), purity.copy())

            for i, bulk in enumerate(self.bulks):
                fitted = _with(bulk, alpha, tau)
                weighted = np.flatnonzero(gammas[i].max(axis=1) > self.floor)

                if self.zero_normal and i == self.normal_clone:
                    shifts[i] = 0.0
                else:
                    log_mu, p = _parameters(states, float(purity[i]))
                    shifts[i] = float(
                        minimize_scalar(
                            _negative_shift_weighted,
                            bounds=(shifts[i] - 3.0, shifts[i] + 3.0),
                            args=(log_mu, p, fitted, gammas[i], weighted),
                            method="bounded",
                        ).x
                    )

                if self.scheme.fit_purity and i != self.normal_clone:
                    purity[i] = float(
                        minimize_scalar(
                            _negative_marginal,
                            bounds=(0.05, 1.0),
                            args=(
                                states,
                                float(shifts[i]),
                                fitted,
                                self.transmat,
                                self.start,
                                self.lengths,
                                best,
                            ),
                            method="bounded",
                        ).x
                    )

            if not self.scheme.poisson:
                common = (self.bulks, states, purity, shifts, gammas, self.floor)
                alpha = float(
                    np.exp(
                        minimize_scalar(
                            _negative_dispersion_weighted,
                            bounds=(np.log(alpha) - 5.0, np.log(alpha) + 5.0),
                            args=("alpha", tau, *common),
                            method="bounded",
                        ).x
                    )
                )
                tau = float(
                    np.exp(
                        minimize_scalar(
                            _negative_dispersion_weighted,
                            bounds=(np.log(tau) - 5.0, np.log(tau) + 5.0),
                            args=("tau", alpha, *common),
                            method="bounded",
                        ).x
                    )
                )

            if self.scheme.states == "fit":
                states = _shared_pairs(
                    paths,
                    [_with(b, alpha, tau) for b in self.bulks],
                    shifts,
                    purity,
                    n_states=self.n_states,
                    lattice=self.lattice,
                    normal=self.normal,
                    distinct=self.scheme.distinct,
                )
                states[self.normal] = (1, 1)

            if (
                np.array_equal(states, before[0])
                and np.allclose(shifts, before[1], atol=1e-4)
                and np.allclose(purity, before[2], atol=1e-4)
            ):
                break

        return states, alpha, tau


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
