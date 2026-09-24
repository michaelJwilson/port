r"""Each clone's pseudobulk as a mixture of the K model clones, inside the run (#380).

`cnaster`'s clone loop fits the HMM to each clone's pseudobulk as labelled,
then scores every spot against every clone's decoded path. When the labels
are impure -- a start that splits a clone, or spots admixed from another --
each pseudobulk is a blend, the path fitted to it describes the blend, and
the spot scores reinforce it. This fits the blend instead:

.. math::
    \mu^{\rm mix}_{ib} = \sum_j W_{ij}\,\mu_{s_j(b)}, \qquad
    p^{\rm mix}_{ib} = \frac{\sum_j W_{ij}\,\mu_{s_j(b)}\,p_{s_j(b)}}
                            {\mu^{\rm mix}_{ib}},

with :math:`s_j` model clone j's **pure** state path over bins, over
`cnaster`'s own fitted states, and :math:`W` row-stochastic, K x K, one
observed clone per model clone. Depth mixes linearly because reads add;
the allele share is a ratio of mixed allele counts, each clone contributing
in proportion to its depth. The pseudobulk is scored by **one** NB and one
BB per bin at the mixed parameters -- a mixture of means, not of
likelihoods, since a pseudobulk sums many spots rather than being drawn from
one clone.

**Depth is library-normalized per observed clone**, as `port`'s shifted
emission normalizes it (#276): mean `base_ib mu_ib / sum_b' lambda_b' mu_ib'`
with `lambda = base_i / sum base_i`. The normalizer is recomputed from the
mixed profile, so no fitted shift is needed.

The fit is coordinate ascent from `W = I` and the fitted paths, which is
`cnaster`'s own answer: each model clone's path by Viterbi against every
observed clone it loads on, then each row of `W` on the simplex. A step is
kept only if the joint log-likelihood does not fall, so the result is never
worse than `W = I` by the model's own measure; a row's new off-diagonal
weight is kept only if it gains half a log-bin count in nats per weight
(BIC), so a pure clone stays pure.

**Mixing is capped:** each row's off-diagonal mass is at most `cap`, a
linear constraint that keeps the feasible set convex. `cap` is fixed (0.2
is at most a fifth of a pseudobulk from other clones) or annealed linearly
over one inference stage's outer iterations, e.g. 0.1 to 0.5, so clones
still forming may borrow little and formed ones more. Uncapped, it is 0.5. :func:`clone_mixture` installs
it between the HMM fit and spot assignment, which then scores spots against
the pure paths.
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
    "annealed",
    "clone_mixture",
    "fit_mixture",
    "mixed_parameters",
    "pseudobulks",
    "score",
]

EPS = 1e-10
"""`cnaster`'s floor on the beta-binomial's shape parameters."""

MIN_GAIN = 1e-3
"""Nats a path step has to gain to be kept."""

MIN_WEIGHT = 1e-4
"""A mixing weight below this is zero."""

Bulks = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""`(rdr, baf, total, base)` per clone and bin, each `(K, bins)`."""


@dataclass(frozen=True)
class MixtureFit:
    """The fitted weights and pure paths, and the log-likelihoods either side."""

    weights: np.ndarray
    paths: np.ndarray
    start: float
    end: float
    sweeps: int
    cap: float | None = None


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


def score(
    bulks: Bulks,
    mu_mix: np.ndarray,
    p_mix: np.ndarray,
    alpha: float,
    tau: float,
) -> np.ndarray:
    """Log-likelihood per observed clone and bin, `(K, bins)`."""
    rdr, baf, total, base = bulks
    weight = base / np.maximum(base.sum(axis=1, keepdims=True), EPS)
    normalizer = np.sum(weight * mu_mix, axis=1, keepdims=True)
    mean = base * mu_mix / np.maximum(normalizer, EPS)

    scored: np.ndarray = _nb(rdr, mean, alpha) + _bb(baf, total, p_mix, tau)
    return scored


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


SUPPORT = 0.01
"""An off-diagonal weight below this is dropped before the BIC test."""

DEFAULT_CAP = 0.5
"""The largest share of a pseudobulk from other clones, uncapped: at 0.5 an
observed clone is still at least half itself, so the one-to-one pairing holds."""


def _project(row: np.ndarray, i: int, limit: float) -> np.ndarray:
    """The KL projection onto `W_ii >= 1 - limit`: the others share `limit`."""
    outside = 1.0 - row[i]
    if outside <= limit:
        return np.asarray(row, dtype=np.float64)

    projected = np.asarray(row * (limit / outside), dtype=np.float64)
    projected[i] = 1.0 - limit
    return projected


def _row(
    i: int,
    weights: np.ndarray,
    bulks: Bulks,
    mu: np.ndarray,
    p: np.ndarray,
    paths: np.ndarray,
    alpha: float,
    tau: float,
    cap: float | None = None,
    steps: int = 60,
) -> np.ndarray:
    """Row `i` of `W` by mirror descent in KL geometry, under a mixing cap.

    The feasible set is the simplex with `sum_{j != i} W_ij <= cap`, a
    polytope. Each step is exponentiated gradient, `w <- w exp(-eta grad)`
    renormalized -- the one-sided Sinkhorn projection -- then the KL
    projection onto the cap, with `eta` halved until the likelihood does not
    fall. The start is strictly positive, since a multiplicative step cannot
    move a zero. Weights under `SUPPORT` are then dropped, and the row is kept
    only if it gains half a log-bin count in nats per remaining contaminant
    (BIC) over the current row, so a pure clone stays pure.
    """
    one: Bulks = (
        bulks[0][i : i + 1],
        bulks[1][i : i + 1],
        bulks[2][i : i + 1],
        bulks[3][i : i + 1],
    )
    depth = mu[paths]
    allele = depth * p[paths]
    k = weights.shape[0]
    limit = min(DEFAULT_CAP if cap is None else cap, DEFAULT_CAP)
    penalty = 0.5 * np.log(paths.shape[1])

    def nll(row: np.ndarray) -> float:
        mixed = row @ depth
        share = np.where(mixed > 0, (row @ allele) / np.maximum(mixed, EPS), 0.5)
        return -float(score(one, mixed[None, :], share[None, :], alpha, tau).sum())

    def gradient(row: np.ndarray, value: float) -> np.ndarray:
        step = 1e-6
        grad = np.empty(k)
        for j in range(k):
            bumped = row.copy()
            bumped[j] += step
            grad[j] = (nll(bumped) - value) / step
        return grad

    current = _project(weights[i].copy(), i, limit)
    before = nll(current)

    if k == 1 or limit <= 0.0:
        return np.asarray(current, dtype=np.float64)

    spread = np.full(k, limit / (k - 1))
    spread[i] = 1.0 - limit
    row = _project(0.8 * current + 0.2 * spread, i, limit)
    value = nll(row)
    eta = 1.0 / max(np.abs(gradient(row, value)).max(), 1e-12)

    for _ in range(steps):
        grad = gradient(row, value)
        grad -= grad @ row
        accepted = False

        while eta > 1e-12:
            trial = row * np.exp(-np.clip(eta * grad, -30.0, 30.0))
            trial = _project(trial / trial.sum(), i, limit)
            trial_value = nll(trial)
            if trial_value <= value:
                accepted = True
                break
            eta *= 0.5

        if not accepted or value - trial_value < 1e-6:
            if accepted:
                row, value = trial, trial_value
            break

        row, value = trial, trial_value
        eta *= 1.5

    row = np.where((np.arange(k) != i) & (row < SUPPORT), 0.0, row)
    row = _project(row / row.sum(), i, limit)
    after = nll(row)
    added = np.count_nonzero(np.delete(row, i)) - np.count_nonzero(
        np.delete(current, i) > 0
    )

    if after + penalty * max(added, 0) <= before:
        return np.asarray(row, dtype=np.float64)

    return np.asarray(current, dtype=np.float64)


def fit_mixture(
    bulks: Bulks,
    mu: np.ndarray,
    p: np.ndarray,
    alpha: float,
    tau: float,
    paths: np.ndarray,
    log_transmat: np.ndarray,
    sweeps: int = 5,
    tol: float = 1e-3,
    cap: float | None = None,
) -> MixtureFit:
    """Coordinate ascent on `(W, paths)` from `(I, paths)`; never downhill.

    `mu`, `p` are the fitted states' rate and share, `(states,)`; `paths` the
    fitted per-clone paths, `(K, bins)`.
    """
    k = paths.shape[0]
    weights = np.eye(k)
    paths = paths.copy()
    n_states = mu.size

    def total(w: np.ndarray, s: np.ndarray) -> float:
        mixed, share = mixed_parameters(mu, p, s, w)
        return float(score(bulks, mixed, share, alpha, tau).sum())

    start = current = total(weights, paths)
    done = 0

    for done in range(1, sweeps + 1):  # noqa: B007
        before = current

        for j in range(k):
            loads = np.flatnonzero(weights[:, j] > 1e-6)
            candidate = np.empty((paths.shape[1], n_states))

            for s in range(n_states):
                trial = paths.copy()
                trial[j] = s
                mixed, share = mixed_parameters(mu, p, trial, weights)
                sub: Bulks = (
                    bulks[0][loads],
                    bulks[1][loads],
                    bulks[2][loads],
                    bulks[3][loads],
                )
                candidate[:, s] = score(
                    sub,
                    mixed[loads],
                    share[loads],
                    alpha,
                    tau,
                ).sum(axis=0)

            trial = paths.copy()
            trial[j] = _viterbi(candidate, log_transmat)
            value = total(weights, trial)

            # NB better by `MIN_GAIN` only: a constant path has no scale under
            #    the per-clone library normalization, so states differing only
            #    in `mu` score alike up to the weight another row puts on it,
            #    and a path is not relabelled for a rounding error of that.
            if value > current + MIN_GAIN:
                paths, current = trial, value

        for i in range(k):
            trial_w = weights.copy()
            trial_w[i] = _row(i, weights, bulks, mu, p, paths, alpha, tau, cap)
            value = total(trial_w, paths)

            if value >= current:
                weights, current = trial_w, value

        if current - before < tol:
            break

    return MixtureFit(weights, paths, start, current, done, cap)


def _states(res: Any) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    mu = np.exp(np.asarray(res["new_log_mu"], dtype=np.float64)[:, 0])
    p = np.asarray(res["new_p_binom"], dtype=np.float64)[:, 0]
    alpha = float(np.asarray(res["new_alphas"]).ravel()[0])
    tau = float(np.asarray(res["new_taus"]).ravel()[0])
    transmat = np.asarray(res["new_log_transmat"], dtype=np.float64)

    if transmat.ndim == 3:
        transmat = transmat[:, :, 0]

    return mu, p, alpha, tau, transmat


def annealed(start: float, end: float, iteration: int, iterations: int) -> float:
    """The cap at outer `iteration` of `iterations`, linear from `start` to `end`."""
    if iterations <= 0:
        return end

    return start + (end - start) * min(iteration / iterations, 1.0)


def _outer_iterations() -> int:
    from cnaster.config import get_global_config

    try:
        return int(get_global_config().hmrf.max_iter_outer)
    except (AttributeError, TypeError, ValueError):
        return 1


@contextlib.contextmanager
def clone_mixture(
    sweeps: int = 5,
    cap: float | None = None,
    anneal: tuple[float, float] | None = None,
) -> Iterator[list[MixtureFit]]:
    """Score spots against each clone's pure path rather than its fitted one.

    Wraps whatever `cnaster.hmrf.pipeline_clone_assignment` is bound to on
    entry -- `port`'s swap, or `--sal`'s -- so it is entered **after** the
    swaps. Only the unphased HMM is handled: its `p` is the allele share the
    mixture formula mixes. Anything else passes through unchanged.
    """
    from cnaster import hmrf

    original = hmrf.pipeline_clone_assignment
    FITS.clear()
    stage: dict[str, int] = {"id": -1, "iteration": 0}

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

            # NB one inference stage passes the same count array on every
            #    outer iteration; a new array is a new stage, and the anneal
            #    restarts with it.
            if stage["id"] != id(single_x):
                stage["id"], stage["iteration"] = id(single_x), 0
            else:
                stage["iteration"] += 1

            limit = cap
            if anneal is not None:
                limit = annealed(*anneal, stage["iteration"], _outer_iterations())

            mu, p, alpha, tau, transmat = _states(res)
            bulks = pseudobulks(single_x, base, total, np.asarray(previous), k)
            fitted = fit_mixture(
                bulks,
                mu,
                p,
                alpha,
                tau,
                np.asarray(pred).reshape(k, n_bins),
                transmat,
                sweeps=sweeps,
                cap=limit,
            )
            FITS.append(fitted)
            pred = fitted.paths.reshape(-1)

        return original(
            single_x, base, total, res, pred, adjacency, previous, *args, **kwargs
        )

    hmrf.pipeline_clone_assignment = assign

    try:
        yield FITS
    finally:
        hmrf.pipeline_clone_assignment = original
