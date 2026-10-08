"""`cnaster`'s beta-binomial M step `Weighted_BetaBinom_mix`, refereed by
`snakes_and_ladders` (#24).

Fits three states at constant trials; the live caller (`normal_baf_bin_filter`) fits
one,
covered separately. The HMM's joint M step has no upstream counterpart and is not
compared.
"""

import numpy as np
import pytest

from tests.adapters import (
    cnaster_beta_binomial_m_step,
    cnaster_beta_binomial_objective,
    upstream_beta_binomial_m_step,
)
from tests.fixtures import BetaBinomialChains, beta_binomial_chains, planted_posterior

SOLVER_AGREEMENT = 1e-3
"""Relative tolerance between the two M steps' `(alpha, beta)` at `CONVERGED_EM_FTOL`: 20x
the measured 5e-5.
"""

RECOVERY_TOLERANCE = 0.12
"""Relative tolerance on recovering the planted `(alpha, beta)`: sampling error, set by the
concentration.
"""


@pytest.fixture
def chains() -> BetaBinomialChains:
    """The default draw. Three states, 2,400 observations at 40 trials."""
    return beta_binomial_chains()


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_m_step_agrees_with_upstream(chains: BetaBinomialChains) -> None:
    """Both M steps reach the same `(alpha, beta)` from a planted smoothed posterior, to
    `SOLVER_AGREEMENT`.
    """
    posterior = planted_posterior(chains, smoothing=0.1)

    upstream = upstream_beta_binomial_m_step(chains, posterior)
    cnaster = cnaster_beta_binomial_m_step(chains, posterior)

    assert upstream.converged, "upstream M step did not settle"
    assert cnaster.converged, "cnaster M step did not settle"

    np.testing.assert_allclose(cnaster.alpha, upstream.alpha, rtol=SOLVER_AGREEMENT)
    np.testing.assert_allclose(cnaster.beta, upstream.beta, rtol=SOLVER_AGREEMENT)


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_m_step_agrees_on_the_success_probability_more_tightly(
    chains: BetaBinomialChains,
) -> None:
    """The two agree on `p` an order tighter than on the concentration."""
    posterior = planted_posterior(chains, smoothing=0.1)

    upstream = upstream_beta_binomial_m_step(chains, posterior)
    cnaster = cnaster_beta_binomial_m_step(chains, posterior)

    location = np.max(
        np.abs(cnaster.success_probability - upstream.success_probability)
        / upstream.success_probability
    )
    concentration = np.max(
        np.abs(cnaster.concentration - upstream.concentration) / upstream.concentration
    )

    assert location < 1e-4, f"success probabilities differ by {location:.3e}"
    assert concentration < SOLVER_AGREEMENT, (
        f"concentrations differ by {concentration:.3e}"
    )
    assert location < concentration, (
        "the location is supposed to be the better determined coordinate; "
        f"got {location:.3e} against {concentration:.3e}"
    )


@pytest.mark.end2end
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_m_step_recovers_the_planted_family(chains: BetaBinomialChains) -> None:
    """At the planted posterior (`smoothing = 0`) both recover the planted family, to
    `RECOVERY_TOLERANCE`.
    """
    posterior = planted_posterior(chains, smoothing=0.0)

    upstream = upstream_beta_binomial_m_step(chains, posterior)
    cnaster = cnaster_beta_binomial_m_step(chains, posterior)

    for name, fitted in (("upstream", upstream), ("cnaster", cnaster)):
        np.testing.assert_allclose(
            fitted.alpha,
            chains.alpha,
            rtol=RECOVERY_TOLERANCE,
            err_msg=f"{name} did not recover the planted alpha",
        )
        np.testing.assert_allclose(
            fitted.beta,
            chains.beta,
            rtol=RECOVERY_TOLERANCE,
            err_msg=f"{name} did not recover the planted beta",
        )


@pytest.mark.analytic
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_m_step_does_not_increase_its_own_objective(chains: BetaBinomialChains) -> None:
    """The fit does not increase `cnaster`'s own `nloglikeobs` from the planted parameters."""
    posterior = planted_posterior(chains, smoothing=0.1)

    at_truth = cnaster_beta_binomial_objective(
        chains, posterior, chains.alpha, chains.beta
    )
    fitted = cnaster_beta_binomial_m_step(chains, posterior)
    at_fit = cnaster_beta_binomial_objective(
        chains, posterior, fitted.alpha, fitted.beta
    )

    assert at_fit <= at_truth, (
        f"the M step raised its own negative log-likelihood, "
        f"{at_truth:.6e} -> {at_fit:.6e}"
    )


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_the_two_dispersion_branches_coincide_at_one_state() -> None:
    """At `K = 1` (the live shape) both `shared_dispersion` branches reach upstream's
    answer.
    """
    single = beta_binomial_chains(n_states=1, success_probability=np.array([0.35]))
    posterior = planted_posterior(single, smoothing=0.0)

    per_state = cnaster_beta_binomial_m_step(single, posterior, shared_dispersion=False)
    shared = cnaster_beta_binomial_m_step(single, posterior, shared_dispersion=True)
    upstream = upstream_beta_binomial_m_step(single, posterior)

    np.testing.assert_allclose(per_state.alpha, shared.alpha, rtol=SOLVER_AGREEMENT)
    np.testing.assert_allclose(per_state.beta, shared.beta, rtol=SOLVER_AGREEMENT)
    np.testing.assert_allclose(shared.alpha, upstream.alpha, rtol=SOLVER_AGREEMENT)
    np.testing.assert_allclose(shared.beta, upstream.beta, rtol=SOLVER_AGREEMENT)


@pytest.mark.smoke
def test_the_design_carries_the_posterior_to_the_right_state(
    chains: BetaBinomialChains,
) -> None:
    """Every design row's weight is the posterior for the observation and state its one-hot
    `exog` names.
    """
    posterior = planted_posterior(chains, smoothing=0.1)
    observations = np.asarray(chains.dataset.observations).reshape(-1)

    from tests.adapters import cnaster_beta_binomial_design

    endog, exog, weights, exposure = cnaster_beta_binomial_design(chains, posterior)

    expected_rows = observations.size * chains.n_states
    assert endog.shape == (expected_rows,)

    states = np.argmax(exog, axis=1)
    rows = np.arange(expected_rows)

    np.testing.assert_array_equal(endog, observations[rows // chains.n_states])
    np.testing.assert_array_equal(states, rows % chains.n_states)
    np.testing.assert_allclose(
        weights,
        posterior.reshape(-1, chains.n_states)[rows // chains.n_states, states],
    )
    np.testing.assert_array_equal(
        exposure, np.full(expected_rows, float(chains.trials))
    )


@pytest.mark.oracle
@pytest.mark.xfail(
    strict=True,
    reason=(
        "issue #30: get_em_solver_params hands L-BFGS-B a *relative* ftol, so "
        "the effective absolute criterion scales with the objective's "
        "magnitude and the solve stops several nats short of its maximum "
        "while reporting that it converged"
    ),
)
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_m_step_agrees_with_upstream_at_cnaster_s_own_settings(
    chains: BetaBinomialChains,
) -> None:
    """Agreement with upstream at `cnaster`'s shipped `em_ftol = 1e-6`; strict xfail until
    #30 is fixed.
    """
    posterior = planted_posterior(chains, smoothing=0.1)

    upstream = upstream_beta_binomial_m_step(chains, posterior)
    cnaster = cnaster_beta_binomial_m_step(chains, posterior)

    assert cnaster.converged, (
        "the solve reported failure; this test is about a solve that reports "
        "success without reaching the maximum"
    )

    np.testing.assert_allclose(cnaster.alpha, upstream.alpha, rtol=SOLVER_AGREEMENT)
    np.testing.assert_allclose(cnaster.beta, upstream.beta, rtol=SOLVER_AGREEMENT)


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_the_shipped_solver_options_carry_a_key_scipy_rejects(
    chains: BetaBinomialChains,
) -> None:
    """`get_em_solver_params` passes `disp`, which `scipy`'s `L-BFGS-B` rejects with
    `OptimizeWarning`.
    """
    import warnings

    posterior = planted_posterior(chains, smoothing=0.1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cnaster_beta_binomial_m_step(chains, posterior)

    rejected = [w for w in caught if "Unknown solver options" in str(w.message)]
    assert rejected, (
        "expected scipy to reject an option cnaster sends; if the option list "
        "was fixed, delete this test"
    )
    assert "disp" in str(rejected[0].message), (
        f"a different option is being rejected now: {rejected[0].message}"
    )
