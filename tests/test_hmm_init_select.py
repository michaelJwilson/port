"""Initializer backends scored by one referee, and the selection over them (#229 stages 3, 4; #230).

`patch` tests check agreement; the sal recovery tests are `oracle` against planted
states.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.emission_family import count_pair_family
from port.sandbox.patch.hmm_initialize.backends import (
    DEFAULT_ALPHA,
    DEFAULT_TAU,
    Candidate,
    cnaster_gmm_backend,
    referee_score,
    sal_emission_backend,
    select,
)

EXPOSURE, TRIALS = 40.0, 60.0
N_OBS = 400


@pytest.fixture
def drawn() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts from one known state, at constant exposure and trials."""
    rng = np.random.default_rng(3)
    totals = rng.poisson(EXPOSURE * np.exp(0.0), size=N_OBS).astype(float)
    successes = rng.binomial(int(TRIALS), 0.32, size=N_OBS).astype(float)

    X = np.stack([totals, successes], axis=1).reshape(N_OBS, 2, 1)

    return X, np.full(N_OBS, EXPOSURE), np.full(N_OBS, TRIALS)


def _candidate(name: str, log_mu: list[float], p_binom: list[float]) -> Candidate:
    n = len(log_mu)

    return Candidate(
        backend=name,
        log_mu=np.array(log_mu),
        p_binom=np.array(p_binom),
        alphas=np.full(n, DEFAULT_ALPHA),
        taus=np.full(n, DEFAULT_TAU),
    )


@pytest.mark.patch
def test_the_referee_prefers_the_state_the_data_came_from(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The referee ranks the generating state first."""
    X, exposure, trials = drawn

    truth = _candidate("truth", [0.0], [0.32])
    wrong = _candidate("wrong", [1.2], [0.05])

    assert referee_score(truth, X, exposure, trials) > referee_score(
        wrong, X, exposure, trials
    )


@pytest.mark.patch
def test_the_referee_is_indifferent_to_the_space_a_backend_fitted_in(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Candidates with identical parameters score identically, whatever backend fitted them."""
    X, exposure, trials = drawn

    one = _candidate("gaussian_space", [0.0], [0.32])
    two = _candidate("count_space", [0.0], [0.32])

    assert referee_score(one, X, exposure, trials) == referee_score(
        two, X, exposure, trials
    )


@pytest.mark.patch
def test_select_takes_the_best_and_reports_what_it_chose_from(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """`select` takes the best and reports how many it chose from."""
    X, exposure, trials = drawn

    candidates = [
        _candidate("wrong", [1.2], [0.05]),
        _candidate("truth", [0.0], [0.32]),
        _candidate("also_wrong", [-1.5], [0.9]),
    ]
    chosen = select(candidates, lambda c: referee_score(c, X, exposure, trials))

    assert chosen.best.backend == "truth"
    assert chosen.n_candidates == 3
    assert set(chosen.scores) == {"wrong", "truth", "also_wrong"}

    rendered = str(chosen)

    assert "biased upward" in rendered
    assert "n=3" in rendered
    assert "#230" in rendered


@pytest.mark.patch
def test_selecting_from_nothing_is_refused() -> None:
    """Selecting from no initializers is refused."""
    with pytest.raises(ValueError, match="no candidates"):
        select([], lambda _c: 0.0)


@pytest.mark.patch
def test_the_parameter_map_round_trips(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """`count_pair_family` and the backend's inverse round-trip the parameters."""
    log_mu = np.array([-0.3, 0.0, 0.45])
    alphas = np.array([0.08, 0.12, 0.05])
    p_binom = np.array([0.50, 0.32, 0.18])
    taus = np.array([30.0, 22.0, 45.0])

    family = count_pair_family(
        log_mu, alphas, p_binom, taus, exposure=EXPOSURE, trials=TRIALS
    )

    back_log_mu = np.log(np.asarray(family.total.mean).reshape(-1) / EXPOSURE)
    back_alphas = 1.0 / np.asarray(family.total.dispersion).reshape(-1)
    back_p = np.asarray(family.rate).reshape(-1)
    back_taus = np.asarray(family.concentration).reshape(-1)

    assert np.allclose(back_log_mu, log_mu, rtol=0, atol=1e-12)
    assert np.allclose(back_alphas, alphas, rtol=0, atol=1e-12)
    assert np.allclose(back_p, p_binom, rtol=0, atol=1e-12)
    assert np.allclose(back_taus, taus, rtol=0, atol=1e-12)


@pytest.mark.oracle
def test_the_sal_backend_recovers_the_planted_states(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Upstream's mixture recovers the planted state in `cnaster`'s parameters."""
    X, exposure, trials = drawn

    candidate = sal_emission_backend(X, exposure, trials, n_states=2, seed=5)

    assert candidate.backend == "sal_emission"
    assert candidate.n_states == 2
    assert candidate.detail["iterations"] >= 1

    # NB data come from one state, so at least one component must sit on it.
    assert np.min(np.abs(candidate.log_mu - 0.0)) < 0.25, (
        f"log_mu {candidate.log_mu} misses the planted 0.0"
    )
    assert np.min(np.abs(candidate.p_binom - 0.32)) < 0.1, (
        f"p_binom {candidate.p_binom} misses the planted 0.32"
    )


@pytest.mark.oracle
def test_the_sal_backend_beats_a_deliberately_wrong_start(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The fit scores higher than a deliberately wrong start under `cnaster`'s density."""
    X, exposure, trials = drawn

    fitted = sal_emission_backend(X, exposure, trials, n_states=2, seed=5)
    wrong = _candidate("wrong", [1.5, -1.5], [0.9, 0.05])

    assert referee_score(fitted, X, exposure, trials) > referee_score(
        wrong, X, exposure, trials
    )


@pytest.mark.oracle
def test_sals_density_with_a_covariate_is_cnasters() -> None:
    """sal's covariate density equals `cnaster`'s NB and BB bin for bin, within 1e-10 (sal #1083)."""
    import torch
    from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
    from sal.emissions import CountPairEmission

    rng = np.random.default_rng(0)
    n = 400
    log_mu = np.array([-0.3, 0.0, 0.5])
    alphas = np.array([0.05, 0.1, 0.2])
    p_binom = np.array([0.5, 0.2, 0.35])
    taus = np.array([30.0, 40.0, 50.0])
    exposure = rng.uniform(20, 200, n)
    trials = rng.integers(0, 60, n).astype(float)
    totals = rng.poisson(exposure).astype(float)
    successes = np.minimum(rng.poisson(trials * 0.4), trials).astype(float)

    family = CountPairEmission(
        dispersion=1 / alphas,
        mean=np.exp(log_mu),
        alpha=p_binom * taus,
        beta=(1 - p_binom) * taus,
        trials=np.full(3, 10.0),
        joint=False,
    )
    ours = np.asarray(
        family.log_density(
            torch.as_tensor(np.column_stack([totals, successes])),
            covariate=torch.as_tensor(np.column_stack([exposure, trials])),
        )
    ).T

    theirs = np.zeros((3, n))
    for state in range(3):
        nb, bb = np.zeros(n), np.zeros(n)
        _nb_logpmf_1d(totals, exposure, float(np.exp(log_mu[state])), alphas[state], nb)
        _bb_logpmf_1d(successes, trials, p_binom[state], taus[state], bb)
        theirs[state] = nb + bb

    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-10)


def _varying() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three planted states at 3,000 bins, each bin its own exposure and trials."""
    rng = np.random.default_rng(1)
    n = 3000
    log_mu = np.log([0.8, 1.0, 1.6])
    p_binom = np.array([0.5, 0.2, 0.35])
    alpha, tau = 0.05, 40.0
    state = rng.integers(0, 3, n)
    exposure = rng.uniform(20, 200, n)
    trials = rng.integers(5, 60, n).astype(float)
    mean = exposure * np.exp(log_mu[state])
    r = 1 / alpha
    totals = rng.negative_binomial(r, r / (r + mean)).astype(float)
    rate = rng.beta(p_binom[state] * tau, (1 - p_binom[state]) * tau)
    successes = rng.binomial(trials.astype(int), rate).astype(float)
    X = np.stack([totals, successes], axis=1)[:, :, None]
    return X, exposure[:, None], trials[:, None], log_mu, p_binom


@pytest.mark.oracle
@pytest.mark.cnaster
def test_the_sal_backend_outscores_cnasters_gmm_under_cnasters_density(
    cnaster_config: None,
) -> None:
    """On varying exposure, sal's fit outscores `gmm_init` under `cnaster`'s density."""
    X, exposure, trials, _, _ = _varying()
    n_obs = X.shape[0]

    sal = sal_emission_backend(X, exposure, trials, n_states=3, seed=1)
    gmm = cnaster_gmm_backend(
        3,
        X,
        exposure,
        trials,
        "smp",
        np.array([n_obs]),
        np.log(np.full((3, 3), 1 / 3)),
        np.log(np.full(n_obs, 1e-3)),
        seed=1,
    )

    assert referee_score(sal, X, exposure, trials) > referee_score(
        gmm, X, exposure, trials
    )


@pytest.mark.end2end
def test_the_sal_backend_recovers_states_under_a_varying_exposure() -> None:
    """Recovers planted states under varying exposure: `log_mu` to 0.1, `p` to 0.05 (#236)."""
    X, exposure, trials, log_mu, p_binom = _varying()

    candidate = sal_emission_backend(X, exposure, trials, n_states=3, seed=1)
    order = np.argsort(candidate.log_mu)

    np.testing.assert_allclose(candidate.log_mu[order], log_mu, atol=0.1)
    np.testing.assert_allclose(candidate.p_binom[order], p_binom, atol=0.05)
