"""Several initializers, one referee, and the best of them.

**#229 stages 3 and 4.** `cnaster` picks an initializer by a default argument
and has never compared it to anything. These pin the machinery that makes a
comparison possible: one score every backend is judged by, and a selection
that reports what it chose from.

The markers split on what each test judges. The round trip and the referee's
properties are `patch` -- they say two implementations agree, not that either
is right. `test_the_sal_backend_recovers_the_planted_states` is `oracle`: the
referee is `snakes_and_ladders`, and the claim is about where the states are.

**Which backend is better is #230**, which judges on recovery of planted truth
under a matched budget. Nothing here decides that, and `Selection.__str__`
says so on every line it prints.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.extensions.emission_family import count_pair_family
from port.patch.hmm_initialize.backends import (
    DEFAULT_ALPHA,
    DEFAULT_TAU,
    Candidate,
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
    """The yardstick has to rank the truth first, or ranking by it selects nothing."""
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
    """Two candidates with identical parameters score identically.

    Trivial to state and the whole point: a backend cannot win by reporting
    a bigger number in its own units, because its own number is never read.
    """
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
    """Selection is in-sample maximization, so the count travels with the score."""
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
    """A run with no initializer is a configuration error, caught here."""
    with pytest.raises(ValueError, match="no candidates"):
        select([], lambda _c: 0.0)


@pytest.mark.patch
def test_the_parameter_map_round_trips(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """`count_pair_family` and the backend's inverse undo each other.

    The backend reads `rate`, `concentration` and `total.dispersion` back out
    of a fitted family. If those are not the inverse of what
    `count_pair_family` puts in, every `sal_emission` candidate is scored on
    parameters that are not the ones it fitted -- silently, because the
    referee would still return a number.
    """
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
    """Upstream's mixture, fitted in the family, judged against the truth.

    No log, no standardization, no inverse: #229's steps 2, 3 and 11 do not
    exist on this path. What is checked is that the fit lands where the data
    were drawn from, in `cnaster`'s parameters.
    """
    X, exposure, trials = drawn

    candidate = sal_emission_backend(X, exposure, trials, n_states=2, seed=5)

    assert candidate.backend == "sal_emission"
    assert candidate.n_states == 2
    assert candidate.detail["iterations"] >= 1

    # NB the data come from one state, so at least one component must sit on
    #    it. The other is free to go anywhere the likelihood allows.
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
    """The fit is worth something, measured on the shared referee.

    `CLAUDE.md` forbids a test that only asserts something ran. This is the
    cheapest claim that is not that: the fitted parameters score higher under
    `cnaster`'s own density than a start that is plainly wrong.
    """
    X, exposure, trials = drawn

    fitted = sal_emission_backend(X, exposure, trials, n_states=2, seed=5)
    wrong = _candidate("wrong", [1.5, -1.5], [0.9, 0.05])

    assert referee_score(fitted, X, exposure, trials) > referee_score(
        wrong, X, exposure, trials
    )


@pytest.mark.warning
def test_the_backend_refuses_a_varying_exposure(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """#236, and the reason this backend is regime-limited.

    `cnaster` divides the exposure out of the mean and leaves it in the
    variance; upstream's family has no per-observation exposure at all. So
    outside the constant regime there is no correspondence, and the backend
    raises rather than fitting a model the data did not come from.

    `warning` rather than `bug`: refusing is the defensible behaviour. What
    is suspicious is that the regime excludes real data, which is #57 and
    #65.
    """
    from port.extensions.emission_family import CovariateNotConstant

    X, _, trials = drawn
    varying = np.linspace(EXPOSURE, 3.0 * EXPOSURE, N_OBS)

    with pytest.raises(CovariateNotConstant, match="base_nb_mean varies"):
        sal_emission_backend(X, varying, trials, n_states=2, seed=5)
