"""Several HMM initializers, one referee, and the best of them.

**#229 stages 3 and 4.** `cnaster` ships two initializers and picks one by a
default argument (`hmrf.py:425`, `phasing.py:48`); `snakes_and_ladders` carries
a third in `opt/emission_mixture`. Nothing compared them, because nothing could:
they report likelihoods in different spaces.

## One yardstick, and it is not the fitter's own

A `GaussianMixture`'s likelihood is Gaussian over standardized log-RDR; an
emission-family mixture's is negative binomial times beta-binomial over raw
counts. **Those are not comparable numbers**, so ranking backends by "their"
likelihood ranks the spaces rather than the fits.

`referee_score` is the single yardstick: the mixture log-likelihood under
`cnaster`'s **own** emission density, at uniform weights, over the
observations as given. Both `cnaster` initializers already compute something
like it privately -- `gmm_init`'s step 7 posterior ranking and
`cna_mixture_init`'s `best_solution_lnlike` -- and neither returns it.

## Selecting inflates the winner

`select` takes the maximum over candidates, which is in-sample maximization,
so the winning score is biased upward by construction. It reports how many it
chose from for that reason, and **which initializer is better is #230**, which
judges on recovery of planted truth rather than on the score selection
maximized.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d

MIRRORS: tuple[str, ...] = ("cnaster.hmm_initialize",)
"""`hmm_initialize`'s two initializers, scored on one referee.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

__all__ = [
    "Candidate",
    "Selection",
    "cnaster_gmm_backend",
    "referee_score",
    "sal_emission_backend",
    "select",
]

DEFAULT_ALPHA = 0.1
DEFAULT_TAU = 1_000.0
"""`cna_mixture_init`'s starting dispersion and concentration.

`gmm_init` returns `None` for both, so a candidate from it has no shape of
its own and these stand in. Reproduced from `hmm_initialize.py:88` rather
than chosen, because a different pair would change the referee score and make
the backends incomparable for a reason that has nothing to do with the fits.
"""


@dataclass(frozen=True)
class Candidate:
    """One initializer's answer, in `cnaster`'s parameters.

    Every backend returns this shape whatever space it fitted in, which is
    what makes one referee possible.
    """

    backend: str
    log_mu: np.ndarray
    p_binom: np.ndarray
    alphas: np.ndarray
    taus: np.ndarray
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return int(np.asarray(self.log_mu).reshape(-1).size)


@dataclass(frozen=True)
class Selection:
    """The chosen candidate, and what it was chosen from."""

    best: Candidate
    score: float
    scores: dict[str, float]

    @property
    def n_candidates(self) -> int:
        return len(self.scores)

    def __str__(self) -> str:
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        rows = "  ".join(f"{name}={value:.6g}" for name, value in ranked)

        return (
            f"selected {self.best.backend} at {self.score:.6g} "
            f"(max over n={self.n_candidates}, biased upward; #230 decides "
            f"which is better) | {rows}"
        )


def referee_score(
    candidate: Candidate,
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
) -> float:
    """The mixture log-likelihood under `cnaster`'s own emission density.

    Uniform weights, so this scores *where the states are* rather than how
    the mass is split -- which is what an initializer decides and all it
    decides. `logsumexp` over states per observation, summed.

    The same function scores every backend, which is the whole point: a
    candidate cannot win by being fitted in a space where numbers are larger.
    """
    totals = np.asarray(X[:, 0, 0] if X.ndim == 3 else X[:, 0], dtype=np.float64)
    successes = np.asarray(X[:, 1, 0] if X.ndim == 3 else X[:, 1], dtype=np.float64)

    exposure = np.asarray(base_nb_mean, dtype=np.float64).reshape(-1)
    trials = np.asarray(total_bb_RD, dtype=np.float64).reshape(-1)

    log_mu = np.asarray(candidate.log_mu, dtype=np.float64).reshape(-1)
    p_binom = np.asarray(candidate.p_binom, dtype=np.float64).reshape(-1)
    alphas = np.asarray(candidate.alphas, dtype=np.float64).reshape(-1)
    taus = np.asarray(candidate.taus, dtype=np.float64).reshape(-1)

    n_states = log_mu.size
    n_obs = totals.size
    per_state = np.zeros((n_states, n_obs))

    for state in range(n_states):
        rdr = np.zeros(n_obs)
        baf = np.zeros(n_obs)

        _nb_logpmf_1d(
            totals, exposure, float(np.exp(log_mu[state])), float(alphas[state]), rdr
        )
        _bb_logpmf_1d(successes, trials, float(p_binom[state]), float(taus[state]), baf)

        per_state[state] = rdr + baf

    # NB a non-finite entry is a state the data refuses, not a fit to abort:
    #    `cnaster` writes -1e6 for the same reason (`hmm_initialize.py:123`).
    per_state = np.nan_to_num(per_state, nan=-1e6, posinf=1e6, neginf=-1e6)

    highest = per_state.max(axis=0)
    mixture = highest + np.log(np.exp(per_state - highest).sum(axis=0) / n_states)

    return float(mixture.sum())


def select(
    candidates: Sequence[Candidate],
    scorer: Callable[[Candidate], float],
) -> Selection:
    """The highest-scoring candidate under one referee.

    Refuses an empty set rather than returning `None`: a run with no
    initializer is a configuration error, and discovering it at the first use
    of the parameters is worse than discovering it here.
    """
    if not candidates:
        msg = "no candidates to select from"
        raise ValueError(msg)

    scores = {candidate.backend: scorer(candidate) for candidate in candidates}
    best = max(candidates, key=lambda candidate: scores[candidate.backend])

    return Selection(best=best, score=scores[best.backend], scores=scores)


def cnaster_gmm_backend(*arguments: Any, seed: int = 0, **keywords: Any) -> Candidate:
    """`cnaster.hmm_initialize.gmm_init`, as a candidate.

    It returns `None` for the dispersion and concentration, so the defaults
    above stand in -- stated rather than silent, because the referee reads
    them.
    """
    from cnaster.hmm_initialize import gmm_init

    log_mu, p_binom, alphas, taus = gmm_init(*arguments, random_state=seed, **keywords)

    n_states = np.asarray(log_mu).reshape(-1).size

    return Candidate(
        backend="cnaster_gmm",
        log_mu=np.asarray(log_mu).reshape(-1),
        p_binom=np.asarray(p_binom).reshape(-1),
        alphas=np.full(n_states, DEFAULT_ALPHA)
        if alphas is None
        else np.asarray(alphas).reshape(-1),
        taus=np.full(n_states, DEFAULT_TAU)
        if taus is None
        else np.asarray(taus).reshape(-1),
    )


def sal_emission_backend(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    n_states: int,
    *,
    seed: int = 0,
    max_iterations: int = 200,
) -> Candidate:
    """`snakes_and_ladders.opt.emission_mixture`, as a candidate.

    Fits the mixture **in the family the data came from**, so there is no log,
    no standardization and no inverse -- #229's steps 2, 3 and 11 do not exist
    on this path.

    Regime-limited by construction: `count_pair_family`'s `constant_covariate`
    raises where the exposure or trial count varies across bins, because
    upstream carries one of each per state (#57, #65). That refusal is the
    honest behaviour, not a gap to paper over.
    """
    import torch
    from snakes_and_ladders.emissions import CountPairEmission
    from snakes_and_ladders.opt.emission_mixture import (
        CountPairSeeding,
        expectation_maximization,
        plus_plus_start,
    )

    from port.patch.emission_family import constant_covariate

    exposure = constant_covariate(np.asarray(base_nb_mean), "base_nb_mean")
    trials = constant_covariate(np.asarray(total_bb_RD), "total_bb_RD")

    totals = np.asarray(X[:, 0, 0] if X.ndim == 3 else X[:, 0], dtype=np.float64)
    successes = np.asarray(X[:, 1, 0] if X.ndim == 3 else X[:, 1], dtype=np.float64)
    observations = np.column_stack([totals, successes])

    rng = np.random.default_rng(seed)
    seeding = CountPairSeeding(
        dispersion=1.0 / DEFAULT_ALPHA,
        concentration=DEFAULT_TAU,
        joint=False,
        trials=trials,
    )
    components = plus_plus_start(observations, n_states, seeding, rng)
    weights = torch.full((n_states,), 1.0 / n_states, dtype=torch.float64)

    fit = expectation_maximization(
        observations, weights, components, max_iterations=max_iterations
    )

    # NB back into `cnaster`'s parameters, the inverse of
    #    `emission_family.count_pair_family`, checked as a round trip.
    #
    #    Upstream names three of the four directly: `rate` is the
    #    beta-binomial's mean, which is `p_binom`; `concentration` is
    #    `alpha + beta`, which is `tau`; and the negative binomial is its own
    #    object under `total`, carrying `dispersion = 1 / alphas`. Only the
    #    exposure has to be divided back out, because upstream folds it into
    #    the mean and `cnaster` keeps it separate -- which is the covariate
    #    gap (#57, #65) showing up as one line of arithmetic.
    fitted = fit.components

    assert isinstance(fitted, CountPairEmission), (
        f"expected a CountPairEmission, got {type(fitted).__name__}"
    )

    depth = np.asarray(fitted.total.mean, dtype=np.float64).reshape(-1)
    dispersion = np.asarray(fitted.total.dispersion, dtype=np.float64).reshape(-1)
    rate = np.asarray(fitted.rate, dtype=np.float64).reshape(-1)
    concentration = np.asarray(fitted.concentration, dtype=np.float64).reshape(-1)

    return Candidate(
        backend="sal_emission",
        log_mu=np.log(np.maximum(depth, 1e-12) / exposure),
        p_binom=rate,
        alphas=1.0 / np.maximum(dispersion, 1e-12),
        taus=concentration,
        detail={
            "log_likelihood": fit.log_likelihood,
            "iterations": fit.iterations,
            "at_boundary": fit.at_boundary,
        },
    )
