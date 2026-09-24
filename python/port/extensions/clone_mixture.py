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
    "lattice",
    "mixed_parameters",
    "normal_clone",
    "pseudobulks",
    "score",
    "scoring_states",
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


ROW_MODE = "bic"
"""How a row of `W` is kept sparse: `bic`, nested support with a BIC cost per
contaminant; or `entropy`, a penalty `ENTROPY_WEIGHT * H(row)` on the
objective and nothing else."""

ENTROPY_WEIGHT = 5.0
"""Nats per nat of row entropy, in `entropy` mode."""

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

    if ROW_MODE == "entropy":
        plain = nll

        def nll(row: np.ndarray) -> float:
            return plain(row) + ENTROPY_WEIGHT * _entropy(row)

    before = nll(current)

    if k == 1 or limit <= 0.0:
        return np.asarray(current, dtype=np.float64)

    def descend(row: np.ndarray) -> tuple[np.ndarray, float]:
        """Exponentiated gradient from `row`; its zeros stay zero."""
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

            if not accepted:
                break

            gain = value - trial_value
            row, value = trial, trial_value
            if gain < 1e-6:
                break
            eta *= 1.5

        row = np.where((np.arange(k) != i) & (row < SUPPORT), 0.0, row)
        row = _project(row / row.sum(), i, limit)
        return row, nll(row)

    if ROW_MODE == "entropy":
        spread = np.full(k, limit / (k - 1))
        spread[i] = 1.0 - limit
        full, value = descend(_project(0.8 * current + 0.2 * spread, i, limit))
        best = full if value <= before else current
        return np.asarray(best, dtype=np.float64)

    # NB nested: the row on its current support first, which needs no
    #    evidence beyond not falling; then on every clone, which replaces it
    #    only if it pays the BIC cost of each contaminant it adds.
    kept, kept_value = current, before
    if np.count_nonzero(np.delete(current, i)) > 0:
        restricted, value = descend(current.copy())
        if value <= before:
            kept, kept_value = restricted, value

    spread = np.full(k, limit / (k - 1))
    spread[i] = 1.0 - limit
    full, value = descend(_project(0.8 * kept + 0.2 * spread, i, limit))
    added = np.count_nonzero(np.delete(full, i)) - np.count_nonzero(np.delete(kept, i))

    if value + penalty * max(added, 0) <= kept_value:
        return np.asarray(full, dtype=np.float64)

    return np.asarray(kept, dtype=np.float64)


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
    log_prior: np.ndarray | None = None,
    fixed: int | None = None,
    admixture: tuple[float, ...] = (0.0,),
) -> MixtureFit:
    """Coordinate ascent on `(W, paths)` from `(I, paths)`; never downhill.

    `mu`, `p` are the states' rate and share, `(states,)`; `paths` the
    starting per-clone paths, `(K, bins)`. `log_prior`, per state and bin,
    enters both the path scores and the objective. Clone `fixed` -- the
    normal clone -- keeps its path and a pure row. Several `admixture`
    values are several starts, each tumour row that much normal, and the
    best objective is kept: a pure pair and its admixture only fit together,
    so one start from `W = I` can settle on the wrong pair.
    """
    if len(admixture) > 1:
        fits = [
            fit_mixture(
                bulks,
                mu,
                p,
                alpha,
                tau,
                paths,
                log_transmat,
                sweeps,
                tol,
                cap,
                log_prior,
                fixed,
                (a,),
            )
            for a in admixture
        ]
        return max(fits, key=lambda fit: fit.end)

    k = paths.shape[0]
    weights = np.eye(k)
    if fixed is not None and admixture[0] > 0.0:
        # NB a start with every tumour row `a` normal: the first path sweep
        #    then scores pure pairs under an admixed model, so a pair and the
        #    admixture that only fit together can be found together.
        for i in range(k):
            if i != fixed:
                weights[i, i] = 1.0 - admixture[0]
                weights[i, fixed] = admixture[0]
    paths = paths.copy()
    n_states = mu.size
    prior = np.zeros(n_states) if log_prior is None else log_prior

    def total(w: np.ndarray, s: np.ndarray) -> float:
        mixed, share = mixed_parameters(mu, p, s, w)
        value = float(score(bulks, mixed, share, alpha, tau).sum() + prior[s].sum())
        if ROW_MODE == "entropy":
            value -= ENTROPY_WEIGHT * sum(_entropy(row) for row in w)
        return value

    start = current = total(weights, paths)
    done = 0

    for done in range(1, sweeps + 1):  # noqa: B007
        before = current

        for j in range(k):
            if j == fixed:
                continue

            loads = np.flatnonzero(weights[:, j] > 1e-6)
            candidate = np.empty((paths.shape[1], n_states))
            held = library(bulks[3], mixed_parameters(mu, p, paths, weights)[0])

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
                candidate[:, s] = (
                    score(
                        sub,
                        mixed[loads],
                        share[loads],
                        alpha,
                        tau,
                        held[loads],
                    ).sum(axis=0)
                    + prior[s]
                )

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
            if i == fixed:
                continue

            trial_w = weights.copy()
            trial_w[i] = _row(i, weights, bulks, mu, p, paths, alpha, tau, cap)
            value = total(trial_w, paths)

            if value >= current:
                weights, current = trial_w, value

        if current - before < tol:
            break

    return MixtureFit(weights, paths, start, current, done, cap)


ADMIXTURE_STARTS = (0.0, 0.05, 0.1, 0.2, 0.3)
"""Initial normal weight of every tumour row, one fit per value, best kept."""

PARSIMONY = 0.5
"""Nats per bin per unit of `|A + B - 2|` on a lattice path, as `lattice_decode`."""


def lattice(max_total_copy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(pairs, mu, p)` over every `(A, B)` with `0 < A + B <= max_total_copy`.

    `mu = (A + B) / 2` against a diploid normal, `p = A / (A + B)`: integer
    profiles, so a uniform normal admixture cannot be absorbed into the
    states -- a pure LOH has `p = 0`, and 8 per cent normal moves it to 0.08
    only through `W`.
    """
    from port.extensions.copy_likelihood import candidates

    pairs = candidates(max_total_copy)
    total = pairs.sum(axis=1).astype(np.float64)
    return pairs, total / 2.0, pairs[:, 0] / total


def _max_total_copy() -> int:
    from cnaster.config import get_global_config

    try:
        value = get_global_config().int_copy_num.max_total_copy
    except AttributeError:
        value = None

    return 6 if value is None else int(value)


def normal_clone(mu: np.ndarray, p: np.ndarray, paths: np.ndarray) -> int:
    """The clone with the largest share of bins at `mu = 1`, `p = 1/2`."""
    neutral = (np.abs(mu - 1.0) < 0.1) & (np.abs(p - 0.5) < 0.05)
    return int(np.argmax(neutral[paths].mean(axis=1)))


def scoring_states(
    fit: MixtureFit, mu: np.ndarray, p: np.ndarray, normal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(log_mu, p, pred)` of the profiles spots are scored against.

    Candidate clone `j`'s profile keeps its own normal admixture,
    `W_j,normal`, and drops its weights on other tumour clones: admixture
    is the spot's, contamination by other clones is what the assignment
    undoes. One state per distinct `(clone, pair)`.
    """
    k, n_bins = fit.paths.shape
    rates: list[float] = []
    shares: list[float] = []
    index: dict[tuple[int, int], int] = {}
    pred = np.empty((k, n_bins), dtype=np.int64)

    for j in range(k):
        admixed = 0.0 if j == normal else float(fit.weights[j, normal])

        for b, state in enumerate(fit.paths[j]):
            key = (j, int(state))
            if key not in index:
                depth = admixed * 1.0 + (1.0 - admixed) * mu[state]
                allele = admixed * 0.5 + (1.0 - admixed) * mu[state] * p[state]
                index[key] = len(rates)
                rates.append(depth)
                shares.append(allele / depth if depth > 0 else 0.5)
            pred[j, b] = index[key]

    return np.log(np.maximum(rates, EPS)), np.asarray(shares), pred


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
    space: str = "lattice",
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
            fitted_paths = np.asarray(pred).reshape(k, n_bins)
            normal = normal_clone(mu, p, fitted_paths)

            if space == "lattice":
                pairs, mu, p = lattice(_max_total_copy())
                stay = float(np.exp(np.diag(transmat)).mean())
                n = mu.size
                transmat = np.log(
                    np.full((n, n), (1.0 - stay) / (n - 1))
                    + np.eye(n) * (stay - (1.0 - stay) / (n - 1))
                )
                neutral = int(np.flatnonzero((pairs == 1).all(axis=1))[0])
                start = np.full((k, n_bins), neutral, dtype=np.int64)
                prior = -PARSIMONY * np.abs(pairs.sum(axis=1) - 2).astype(np.float64)
            else:
                start, prior = fitted_paths, None

            fitted = fit_mixture(
                bulks,
                mu,
                p,
                alpha,
                tau,
                start,
                transmat,
                sweeps=sweeps,
                cap=limit,
                log_prior=prior,
                fixed=normal,
                admixture=ADMIXTURE_STARTS if space == "lattice" else (0.0,),
            )
            FITS.append(fitted)

            if space == "lattice":
                log_mu, shares, decoded = scoring_states(fitted, mu, p, normal)
                res = res.copy(deep=True)
                res.unlock()
                alphas = np.asarray(res["new_alphas"], dtype=np.float64)
                taus = np.asarray(res["new_taus"], dtype=np.float64)
                res["new_log_mu"] = log_mu[:, None]
                res["new_p_binom"] = shares[:, None]
                res["new_alphas"] = np.full((log_mu.size, 1), alphas.ravel()[0])
                res["new_taus"] = np.full((log_mu.size, 1), taus.ravel()[0])
                pred = decoded.reshape(-1)
            else:
                pred = fitted.paths.reshape(-1)

        return original(
            single_x, base, total, res, pred, adjacency, previous, *args, **kwargs
        )

    hmrf.pipeline_clone_assignment = assign

    try:
        yield FITS
    finally:
        hmrf.pipeline_clone_assignment = original
