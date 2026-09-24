r"""Each clone's pseudobulk as a mixture of the clones and a diploid normal (#380).

`cnaster`'s clone loop fits the HMM to each clone's pseudobulk as labelled,
then scores every spot against every clone's decoded path. When the labels
are impure -- a start that splits a clone, spots admixed from another, or
normal cells in tumour spots -- each pseudobulk is a blend, the path fitted
to it describes the blend, and the spot scores reinforce it. This fits the
blend instead:

.. math::
    \mu^{\rm mix}_{ib} = \sum_j W_{ij}\,\mu_{s_j(b)}, \qquad
    p^{\rm mix}_{ib} = \frac{\sum_j W_{ij}\,\mu_{s_j(b)}\,p_{s_j(b)}}
                            {\mu^{\rm mix}_{ib}},

over `K + 1` pure profiles: each clone's path :math:`s_j` over the integer
pairs `(A, B)`, `mu = (A + B) / 2` and `p = A / (A + B)`, and a last,
**diploid** column fixed at `(1, 1)`. `W` is `K x (K + 1)` and
row-stochastic. Integer pairs are the point: a uniform normal admixture is
absorbed into continuous fitted states, and cannot be into integer ones, so
it has to show in `W`; the diploid column gives it a known profile to load
on, and no clone has to be named normal. Depth mixes linearly because reads
add; the allele share is a ratio of mixed allele counts. Each pseudobulk is
scored by one NB and one BB per bin at the mixed parameters -- a mixture of
means, not of likelihoods -- library-normalized per observed clone as
`port`'s shifted emission is (#276).

The fit is coordinate ascent: each clone's pure path by Viterbi against the
pseudobulks that load on it, with the library normalizer held while its
candidates are scored; then each row of `W` by exponentiated gradient (the
one-sided Sinkhorn step) with at most half of it off the diagonal. The
objective carries a parsimony prior of `PARSIMONY` nats per `|A + B - 2|`
per bin and an entropy penalty `ENTROPY_WEIGHT * H(row)` per row, so a pure
clone stays pure; a step is kept only if it does not lower it. One fit per
value in `ADMIXTURE_STARTS`, the best kept: a pure pair and its admixture
only fit together.

The penalty costs some accuracy in `W`, measured on the unit fixtures: a
pure sample keeps 0.015 to 0.021 off the diagonal, and a planted 0.08
normal fraction reads 0.058 on an LOH clone. Spot recovery on the
simulated samples ties the BIC alternative in `port.sandbox.admixture`
to 0.002 in ARI; the penalty is the simpler rule.

:func:`clone_mixture` installs it between the HMM fit and spot assignment,
which then scores spots against each clone's pure path at its own diploid
weight. The design study behind these choices, and the variants it set
aside, are in `port.sandbox.admixture`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln

__all__ = [
    "FITS",
    "MixtureFit",
    "clone_mixture",
    "fit_mixture",
    "lattice",
    "mixed_parameters",
    "pseudobulks",
    "score",
    "scoring_states",
]

EPS = 1e-10
"""`cnaster`'s floor on the beta-binomial's shape parameters."""

MIN_GAIN = 1e-3
"""Nats a path step has to gain to be kept."""

ENTROPY_WEIGHT = 1.0
"""Nats of penalty per nat of a row's entropy."""

SUPPORT = 0.01
"""An off-diagonal weight below this is dropped after each row step."""

MAX_MIXING = 0.5
"""The largest share of a pseudobulk off the diagonal: an observed clone is
still at least half itself, so the one-to-one pairing holds."""

ADMIXTURE_STARTS = (0.0, 0.05, 0.1, 0.2, 0.3)
"""Initial diploid weight of every row, one fit per value, best kept."""

PARSIMONY = 0.5
"""Nats per bin per unit of `|A + B - 2|` on a path, as `lattice_decode`."""

Bulks = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""`(rdr, baf, total, base)` per clone and bin, each `(K, bins)`."""


@dataclass(frozen=True)
class MixtureFit:
    """The fitted weights and pure paths, and the objective either side."""

    weights: np.ndarray
    paths: np.ndarray
    start: float
    end: float
    sweeps: int


FITS: list[MixtureFit] = []
"""Every fit :func:`clone_mixture` made in the run, in order, for audit."""


def pseudobulks(
    single_x: np.ndarray,
    base: np.ndarray,
    total: np.ndarray,
    assignment: np.ndarray,
    n_clones: int,
) -> Bulks:
    """Per-clone sums over spots: `(rdr, baf, total, base)`, each `(K, bins)`."""
    one_hot = np.zeros((single_x.shape[2], n_clones))
    one_hot[np.arange(single_x.shape[2]), assignment] = 1.0

    return (
        (single_x[:, 0, :] @ one_hot).T,
        (single_x[:, 1, :] @ one_hot).T,
        (total @ one_hot).T,
        (base @ one_hot).T,
    )


def mixed_parameters(
    mu: np.ndarray, p: np.ndarray, paths: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(mu_mix, p_mix)`, each `(K, bins)`, for state rates `mu` and shares `p`."""
    depth = mu[paths]
    mixed = weights @ depth
    alleles = weights @ (depth * p[paths])

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(mixed > 0, alleles / mixed, 0.5)

    return mixed, share


def _nb(k: np.ndarray, mean: np.ndarray, alpha: float) -> np.ndarray:
    """`cnaster`'s `nbinom_logpmf_numba`, with its zero for a non-positive mean."""
    r = 1.0 / max(alpha, 1.0e-10)
    positive = mean > 0.0
    safe = np.where(positive, mean, 1.0)
    p = 1.0 / (1.0 + alpha * safe)
    out = (
        gammaln(k + r)
        - gammaln(r)
        - gammaln(k + 1.0)
        + r * np.log(p)
        + k * np.log1p(-p)
    )
    return np.asarray(np.where(positive, out, 0.0), dtype=np.float64)


def _bb(k: np.ndarray, n: np.ndarray, p: np.ndarray, tau: float) -> np.ndarray:
    """`cnaster`'s `betabinom_logpmf_numba` at share `p` and concentration `tau`."""
    a = np.maximum(p * tau, EPS)
    b = np.maximum((1.0 - p) * tau, EPS)
    valid = (n >= 0) & (k >= 0) & (k <= n)
    out = (
        gammaln(n + 1.0)
        - gammaln(k + 1.0)
        - gammaln(n - k + 1.0)
        + gammaln(k + a)
        + gammaln(n - k + b)
        - gammaln(n + a + b)
        - gammaln(a)
        - gammaln(b)
        + gammaln(a + b)
    )
    return np.asarray(np.where(valid, out, 0.0), dtype=np.float64)


def library(base: np.ndarray, mu_mix: np.ndarray) -> np.ndarray:
    """Each clone's library normalizer, `sum_b lambda_b mu_b`, `(K, 1)`."""
    weight = base / np.maximum(base.sum(axis=1, keepdims=True), EPS)
    return np.asarray(np.sum(weight * mu_mix, axis=1, keepdims=True), dtype=np.float64)


def score(
    bulks: Bulks,
    mu_mix: np.ndarray,
    p_mix: np.ndarray,
    alpha: float,
    tau: float,
    normalizer: np.ndarray | None = None,
) -> np.ndarray:
    """Log-likelihood per observed clone and bin, `(K, bins)`.

    `normalizer`, `(K, 1)`, holds each clone's library normalizer fixed
    rather than taking it from `mu_mix`: scoring one state at every bin
    otherwise makes the path constant, and a constant path's depth cancels
    against its own normalizer.
    """
    rdr, baf, total, base = bulks
    if normalizer is None:
        normalizer = library(base, mu_mix)
    mean = base * mu_mix / np.maximum(normalizer, EPS)

    scored: np.ndarray = _nb(rdr, mean, alpha) + _bb(baf, total, p_mix, tau)
    return scored


def _entropy(row: np.ndarray) -> float:
    positive = row[row > 0]
    return float(-np.sum(positive * np.log(positive)))


def _viterbi(scores: np.ndarray, log_transmat: np.ndarray) -> np.ndarray:
    """The best path through `(bins, states)` scores under one transition matrix."""
    n_bins, n_states = scores.shape
    back = np.zeros((n_bins, n_states), dtype=np.int64)
    best = scores[0].copy()

    for b in range(1, n_bins):
        step = best[:, None] + log_transmat
        back[b] = np.argmax(step, axis=0)
        best = step[back[b], np.arange(n_states)] + scores[b]

    path = np.empty(n_bins, dtype=np.int64)
    path[-1] = int(np.argmax(best))

    for b in range(n_bins - 1, 0, -1):
        path[b - 1] = back[b, path[b]]

    return path


def _row(
    i: int,
    weights: np.ndarray,
    bulks: Bulks,
    mu: np.ndarray,
    p: np.ndarray,
    paths: np.ndarray,
    alpha: float,
    tau: float,
    steps: int = 60,
) -> np.ndarray:
    """Row `i` of `W`, by exponentiated gradient on its penalized likelihood.

    Each step is `w <- w exp(-eta grad)` renormalized -- the one-sided
    Sinkhorn step -- then scaled so at most :data:`MAX_MIXING` is off the
    diagonal, with `eta` halved until the objective does not rise. The
    descent starts from the current row softened towards an even spread,
    since a multiplicative step cannot move a zero; weights under
    :data:`SUPPORT` are then dropped, and the row is kept only if it is no
    worse than the current one.
    """
    one: Bulks = (
        bulks[0][i : i + 1],
        bulks[1][i : i + 1],
        bulks[2][i : i + 1],
        bulks[3][i : i + 1],
    )
    depth = mu[paths]
    allele = depth * p[paths]
    k = weights.shape[1]
    others = np.arange(k) != i

    def bound(row: np.ndarray) -> np.ndarray:
        row = row / row.sum()
        off = 1.0 - row[i]
        if off > MAX_MIXING:
            row = row * (MAX_MIXING / off)
            row[i] = 1.0 - MAX_MIXING
        return row

    def objective(row: np.ndarray) -> float:
        mixed = row @ depth
        share = np.where(mixed > 0, (row @ allele) / np.maximum(mixed, EPS), 0.5)
        nll = -float(score(one, mixed[None, :], share[None, :], alpha, tau).sum())
        return nll + ENTROPY_WEIGHT * _entropy(row)

    def gradient(row: np.ndarray, value: float) -> np.ndarray:
        step = 1e-6
        return np.array(
            [(objective(row + step * np.eye(k)[j]) - value) / step for j in range(k)]
        )

    current = bound(weights[i].copy())
    before = objective(current)
    spread = np.where(others, MAX_MIXING / (k - 1), 1.0 - MAX_MIXING)
    row = bound(0.8 * current + 0.2 * spread)
    value = objective(row)
    eta = 1.0 / max(np.abs(gradient(row, value)).max(), 1e-12)

    for _ in range(steps):
        grad = gradient(row, value)
        grad -= grad @ row

        while eta > 1e-12:
            trial = bound(row * np.exp(-np.clip(eta * grad, -30.0, 30.0)))
            trial_value = objective(trial)
            if trial_value <= value:
                break
            eta *= 0.5
        else:
            break

        gain = value - trial_value
        row, value = trial, trial_value
        if gain < 1e-6:
            break
        eta *= 1.5

    row = bound(np.where(others & (row < SUPPORT), 0.0, row))

    return np.asarray(row if objective(row) <= before else current, dtype=np.float64)


def fit_mixture(
    bulks: Bulks,
    mu: np.ndarray,
    p: np.ndarray,
    alpha: float,
    tau: float,
    paths: np.ndarray,
    log_transmat: np.ndarray,
    *,
    log_prior: np.ndarray | None = None,
    diploid: int | None = None,
    admixture: float = 0.0,
    sweeps: int = 5,
    tol: float = 1e-3,
) -> MixtureFit:
    """Coordinate ascent on `(W, paths)`; the objective never falls.

    `mu`, `p` are the states' rate and share, `(states,)`; `paths` the
    starting paths, `(K, bins)`, or `(K + 1, bins)` with a last column no
    pseudobulk observes. Path `diploid` is never refitted and has no row.
    `admixture` starts every row with that much weight on it. `log_prior`,
    per state, enters both the path scores and the objective.
    """
    k = paths.shape[0]
    rows = bulks[0].shape[0]
    weights = np.eye(rows, k)

    if diploid is not None and admixture > 0.0:
        for i in range(rows):
            weights[i, i] = 1.0 - admixture
            weights[i, diploid] = admixture

    paths = paths.copy()
    n_states = mu.size
    prior = np.zeros(n_states) if log_prior is None else log_prior

    def objective(w: np.ndarray, s: np.ndarray) -> float:
        mixed, share = mixed_parameters(mu, p, s, w)
        fit = float(score(bulks, mixed, share, alpha, tau).sum() + prior[s].sum())
        return fit - ENTROPY_WEIGHT * sum(_entropy(row) for row in w)

    start = current = objective(weights, paths)
    done = 0

    for done in range(1, sweeps + 1):  # noqa: B007
        before = current

        for j in range(k):
            if j == diploid:
                continue

            loads = np.flatnonzero(weights[:, j] > 1e-6)
            sub: Bulks = (
                bulks[0][loads],
                bulks[1][loads],
                bulks[2][loads],
                bulks[3][loads],
            )
            # NB the library normalizer is held at the current paths: scoring
            #    one state at every bin makes the trial path constant, and a
            #    constant path's depth cancels against its own normalizer.
            held = library(bulks[3], mixed_parameters(mu, p, paths, weights)[0])
            candidate = np.empty((paths.shape[1], n_states))

            for s in range(n_states):
                trial = paths.copy()
                trial[j] = s
                mixed, share = mixed_parameters(mu, p, trial, weights)
                candidate[:, s] = (
                    score(sub, mixed[loads], share[loads], alpha, tau, held[loads]).sum(
                        axis=0
                    )
                    + prior[s]
                )

            trial = paths.copy()
            trial[j] = _viterbi(candidate, log_transmat)
            value = objective(weights, trial)

            # NB better by `MIN_GAIN` only: states differing only in `mu`
            #    score alike on a constant path, and a path is not relabelled
            #    for a rounding error.
            if value > current + MIN_GAIN:
                paths, current = trial, value

        for i in range(rows):
            trial_w = weights.copy()
            trial_w[i] = _row(i, weights, bulks, mu, p, paths, alpha, tau)
            value = objective(trial_w, paths)

            if value >= current:
                weights, current = trial_w, value

        if current - before < tol:
            break

    return MixtureFit(weights, paths, start, current, done)


def lattice(max_total_copy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(pairs, mu, p)` over every `(A, B)` with `0 < A + B <= max_total_copy`."""
    from port.extensions.copy_likelihood import candidates

    pairs = candidates(max_total_copy)
    total = pairs.sum(axis=1).astype(np.float64)
    return pairs, total / 2.0, pairs[:, 0] / total


def scoring_states(
    fit: MixtureFit, mu: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(log_mu, p, pred)` of the profiles spots are scored against.

    Candidate clone `j`'s profile is its pure path at its own diploid weight,
    the last column of `W`; its weights on other clones are dropped, since
    contamination by other clones is what the assignment undoes. One state
    per distinct `(clone, pair)`.
    """
    k, n_bins = fit.weights.shape[0], fit.paths.shape[1]
    rates: list[float] = []
    shares: list[float] = []
    index: dict[tuple[int, int], int] = {}
    pred = np.empty((k, n_bins), dtype=np.int64)

    for j in range(k):
        normal = float(fit.weights[j, -1])

        for b, state in enumerate(fit.paths[j]):
            key = (j, int(state))
            if key not in index:
                depth = normal + (1.0 - normal) * mu[state]
                allele = 0.5 * normal + (1.0 - normal) * mu[state] * p[state]
                index[key] = len(rates)
                rates.append(depth)
                shares.append(allele / depth if depth > 0 else 0.5)
            pred[j, b] = index[key]

    return np.log(np.maximum(rates, EPS)), np.asarray(shares), pred


def _max_total_copy() -> int:
    from cnaster.config import get_global_config

    try:
        value = get_global_config().int_copy_num.max_total_copy
    except AttributeError:
        value = None

    return 6 if value is None else int(value)


def _fitted(res: Any) -> tuple[float, float, float]:
    """`(alpha, tau, stay)`: the dispersions and mean self-transition cnaster fitted."""
    alpha = float(np.asarray(res["new_alphas"]).ravel()[0])
    tau = float(np.asarray(res["new_taus"]).ravel()[0])
    transmat = np.asarray(res["new_log_transmat"], dtype=np.float64)

    if transmat.ndim == 3:
        transmat = transmat[:, :, 0]

    return alpha, tau, float(np.exp(np.diag(transmat)).mean())


@contextlib.contextmanager
def clone_mixture(sweeps: int = 5) -> Iterator[list[MixtureFit]]:
    """Score spots against each clone's pure, admixed path, not its fitted one.

    Wraps whatever `cnaster.hmrf.pipeline_clone_assignment` is bound to on
    entry -- `port`'s swap, or `--sal`'s -- so it is entered **after** the
    swaps. Only the unphased HMM is handled: its `p` is the allele share the
    mixture formula mixes. Anything else passes through unchanged.
    """
    from cnaster import hmrf

    original = hmrf.pipeline_clone_assignment
    FITS.clear()

    def assign(
        single_x: np.ndarray,
        base: np.ndarray,
        total: np.ndarray,
        res: Any,
        pred: np.ndarray,
        adjacency: Any,
        previous: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        hmmclass = kwargs.get("hmmclass")
        n_bins = single_x.shape[0]

        if (
            getattr(hmmclass, "__name__", "") == "hmm_nophasing"
            and np.ndim(pred) == 1
            and pred.size % n_bins == 0
            and pred.size // n_bins > 1
        ):
            k = pred.size // n_bins
            alpha, tau, stay = _fitted(res)
            bulks = pseudobulks(single_x, base, total, np.asarray(previous), k)
            pairs, mu, p = lattice(_max_total_copy())
            n = mu.size
            transmat = np.log(
                np.full((n, n), (1.0 - stay) / (n - 1))
                + np.eye(n) * (stay - (1.0 - stay) / (n - 1))
            )
            neutral = int(np.flatnonzero((pairs == 1).all(axis=1))[0])
            start = np.full((k + 1, n_bins), neutral, dtype=np.int64)
            prior = -PARSIMONY * np.abs(pairs.sum(axis=1) - 2).astype(np.float64)
            fitted = max(
                (
                    fit_mixture(
                        bulks,
                        mu,
                        p,
                        alpha,
                        tau,
                        start,
                        transmat,
                        log_prior=prior,
                        diploid=k,
                        admixture=a,
                        sweeps=sweeps,
                    )
                    for a in ADMIXTURE_STARTS
                ),
                key=lambda fit: fit.end,
            )
            FITS.append(fitted)

            log_mu, shares, decoded = scoring_states(fitted, mu, p)
            res = res.copy(deep=True)
            res.unlock()
            res["new_log_mu"] = log_mu[:, None]
            res["new_p_binom"] = shares[:, None]
            res["new_alphas"] = np.full((log_mu.size, 1), alpha)
            res["new_taus"] = np.full((log_mu.size, 1), tau)
            pred = decoded.reshape(-1)

        return original(
            single_x, base, total, res, pred, adjacency, previous, *args, **kwargs
        )

    hmrf.pipeline_clone_assignment = assign

    try:
        yield FITS
    finally:
        hmrf.pipeline_clone_assignment = original
