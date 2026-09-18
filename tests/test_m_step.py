"""`cnaster`'s emission M step, refereed by `snakes_and_ladders`.

Issue #24. The M step is the one part of the fit nothing checked: the
lattice is pinned by #15 and #17, the emission by #9's consistency tests,
and between them sits a re-estimation that both implementations perform by
*optimization* rather than in closed form. Two optimizers landing on the
same point is the strong statement here, because there is no third place a
shared bug could hide -- they do not share a solver, a parameterization, an
objective evaluation, or a starting point.

**What is under test, and what is not.** `cnaster`'s live beta-binomial M
step is `hmm_emission.Weighted_BetaBinom_mix`, reached from the
`run_cnaster` console script through `normal_spot.normal_baf_bin_filter`
under the `partial` alias `Weighted_BetaBinom`. It is not the HMM's M step:
that is `hmm_nophasing._run_optimization_pipeline`, a single joint
`scipy.optimize.minimize` over the packed parameter vector, and it has no
upstream counterpart of the same shape -- upstream re-estimates each family
separately from a posterior. So this module refereed the estimator that
*does* correspond, and the pipeline is left to a ticket rather than
compared against something it is not.

The live caller fits **one** state. These tests fit three, which exercises
`Weighted_BetaBinom_mix`'s multi-state branches; at one state its
`shared_dispersion` and per-state branches are the same parameter, so the
branch a multi-state fit takes is shipped code with no live caller today.
That is a difference in the problem, stated here as `CLAUDE.md` requires:
it makes these tests a stronger check than the live path needs, not a
weaker one, and the one-state case is covered separately below.

**The regime.** Constant trials, for the reason `adapters` gives for
constant exposure: `cnaster` carries one exposure per observation and the
upstream family one per state, so only a constant is the same problem on
both sides. Every comparison here is inside it.
"""

import numpy as np
import pytest

from tests.adapters import (
    cnaster_beta_binomial_m_step,
    cnaster_beta_binomial_objective,
    upstream_beta_binomial_m_step,
)
from tests.fixtures import (
    BetaBinomialChains,
    beta_binomial_chains,
    planted_posterior,
)

SOLVER_AGREEMENT = 1e-3
"""Relative tolerance between the two M steps' `(alpha, beta)`.

Not a float64 tolerance, and it should not be read as one. Both sides stop
on a convergence criterion rather than at a root -- `cnaster` runs
`L-BFGS-B` and upstream alternating bisection -- so they differ by the
looser of the two criteria, and a flat-topped likelihood turns a small
difference in objective into a larger one in parameter.

Measured rather than assumed. At `CONVERGED_EM_FTOL`, where both solves
reach their maximum, the observed relative difference at the default fixture
is `5e-5`; this floor is twenty times the gap it has to admit. It is the
number to tighten if either criterion tightens, and the number that would
have to loosen for a real disagreement to hide.

Every test using it takes `cnaster_converged_config` rather than
`cnaster_config`. At `cnaster`'s shipped `em_ftol` the two differ by twenty
per cent, which is not a tolerance to widen but the subject of
`test_the_shipped_solver_criterion_stops_short_of_the_maximum`.
"""

RECOVERY_TOLERANCE = 0.12
"""Relative tolerance on recovering the planted `(alpha, beta)`.

Sampling error, not solver error, and far larger. `alpha + beta = 16` at 40
trials over 2,400 observations is a concentration the data identify only
loosely -- the beta-binomial's variance is near-quadratic in the
concentration, so the likelihood is flat in it. Widened by that flatness
rather than by trial: at this fixture the success probabilities recover to
under `2e-2` and the concentrations to `1e-1`, and the shared floor is set
by the second.

A recovery test is not what refereed the estimator -- `SOLVER_AGREEMENT`
does that. This one refutes a different failure: two implementations that
agree with each other on the wrong answer.
"""


@pytest.fixture
def chains() -> BetaBinomialChains:
    """The default draw. Three states, 2,400 observations at 40 trials."""
    return beta_binomial_chains()


@pytest.mark.oracle
@pytest.mark.upstream_oracle
@pytest.mark.critical
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_m_step_agrees_with_upstream(chains: BetaBinomialChains) -> None:
    """Both M steps land on the same `(alpha, beta)` from the same posterior.

    The referee test. The posterior is planted and smoothed, so it is an
    input rather than an inference and a defect in the lattice cannot reach
    this assertion. Smoothed rather than one-hot so every state carries
    weight at every observation: against a one-hot posterior a per-state
    solve that ignored its weights entirely would still pass, because the
    weights would be exactly the partition it would have used anyway.

    What would have to be wrong for this to fail: either implementation
    maximising a different function, parameterizing the beta differently,
    or associating a weight with the wrong state.
    """
    posterior = planted_posterior(chains, smoothing=0.1)

    upstream = upstream_beta_binomial_m_step(chains, posterior)
    cnaster = cnaster_beta_binomial_m_step(chains, posterior)

    assert upstream.converged, "upstream M step did not settle"
    assert cnaster.converged, "cnaster M step did not settle"

    np.testing.assert_allclose(cnaster.alpha, upstream.alpha, rtol=SOLVER_AGREEMENT)
    np.testing.assert_allclose(cnaster.beta, upstream.beta, rtol=SOLVER_AGREEMENT)


@pytest.mark.oracle
@pytest.mark.upstream_oracle
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_m_step_agrees_on_the_success_probability_more_tightly(
    chains: BetaBinomialChains,
) -> None:
    """The two agree on `p` an order tighter than on the concentration.

    Not a restatement of the test above. `(alpha, beta)` mixes a location
    the data pin down with a concentration they barely do, and a single
    tolerance over both is set by the worse of the two. Splitting them says
    *which* coordinate the agreement is in, and pins the ordering: if the
    concentration ever agreed as tightly as the location, the fixture would
    have stopped being flat in it and `RECOVERY_TOLERANCE` above would be
    wrong.
    """
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
@pytest.mark.planted
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_m_step_recovers_the_planted_family(chains: BetaBinomialChains) -> None:
    """At the truth's own posterior, both recover what drew the data.

    The complete-data maximum likelihood: `smoothing = 0` makes the
    posterior the indicator of the planted states, which is the one case
    where the M step's answer is comparable with the fixture's truth rather
    than only with the other implementation's.

    Both sides are asserted, not just `cnaster`'s. An agreement test and a
    recovery test fail differently, and running the referee through the same
    assertion is what distinguishes "`cnaster` is wrong" from "the fixture
    is not identified".
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


@pytest.mark.oracle
@pytest.mark.property
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_m_step_does_not_increase_its_own_objective(chains: BetaBinomialChains) -> None:
    """An M step does not leave the objective worse than it found it.

    The invariant, and the only claim here that holds whichever
    implementation is right. Evaluated through `cnaster`'s own
    `nloglikeobs`, so it is made against the function that was minimised
    rather than against a restatement of it -- a restatement would turn a
    defect in the objective into a passing test.

    The planted parameters are the starting point in the sense that matters:
    the fit is free to reach any point it likes, and the claim is only that
    where it stops is no worse than where the data came from.
    """
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
@pytest.mark.upstream_oracle
@pytest.mark.critical
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
def test_the_two_dispersion_branches_coincide_at_one_state() -> None:
    """At `K = 1`, `shared_dispersion` is the same parameter either way.

    This is the shape the live caller uses:
    `normal_spot.normal_baf_bin_filter` builds the model with
    `exog = ones(n)`, so `num_states` is one. The two branches of
    `Weighted_BetaBinom_mix.nloglikeobs` then describe one problem, and this
    pins that they also *solve* it identically -- which is what licenses the
    rest of this module to use the per-state branch while claiming to say
    something about the live path.

    Upstream referees the common answer, so a failure separates "the
    branches disagree" from "both branches are wrong".
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


@pytest.mark.infra
@pytest.mark.analytic
def test_the_design_carries_the_posterior_to_the_right_state(
    chains: BetaBinomialChains,
) -> None:
    """The flattening is a relabelling, not a computation.

    `cnaster` takes the M step as a regression over `(observation, state)`
    rows and upstream takes it as an array with a trailing state axis. The
    row order is what makes them the same problem, and getting it wrong
    permutes the states rather than raising: the fit would converge, to the
    wrong assignment, and `test_m_step_agrees_with_upstream` would report a
    disagreement without saying where it came from.

    So the correspondence is pinned directly: every row's weight is the
    posterior for the observation and state its one-hot `exog` names.
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
@pytest.mark.upstream_oracle
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
    """The referee test again, at the solver settings `cnaster` ships.

    Deliberately identical to `test_m_step_agrees_with_upstream` in every
    respect but one: it takes `cnaster_config` where that takes
    `cnaster_converged_config`. Same draw, same posterior, same adapter,
    same tolerance. So the pair isolates the cause to the single value that
    differs between them -- `em_ftol`, `1e-6` against `1e-9` -- and nothing
    else in this module can be the reason one passes and the other does not.

    `strict` because a fix should turn this red. When `cnaster` reaches its
    maximum at its own settings there is nothing left to mark, and this test
    becomes the second copy of one above it and should be deleted rather
    than unmarked.

    What is failing, measured: at the default fixture the shipped criterion
    ends the solve eleven iterations in, 7.5 nats below the maximum, at an
    `alpha` a fifth away from it, with `converged` `True`. Issue #30 carries
    the sizes and the fix.
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
@pytest.mark.exact
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
def test_the_shipped_solver_options_carry_a_key_scipy_rejects(
    chains: BetaBinomialChains,
) -> None:
    """`get_em_solver_params` returns `disp` for `L-BFGS-B`, which ignores it.

    A smaller finding, pinned because it is silent in production and
    misleading in a log. `scipy` 1.18's `_minimize_lbfgsb` does not take
    `disp` in `options`, so `_check_unknown_options` raises an
    `OptimizeWarning` and drops it -- every live `Weighted_BetaBinom_mix.fit`
    on the `run_cnaster` path emits one, and the setting never takes effect.

    `fit` supplies `disp` from its own `kwargs` as well, so removing it from
    `get_em_solver_params` alone would not silence this.

    The referee here is `scipy` rather than `snakes_and_ladders`: what is
    being checked is that the option `cnaster` sends is one the solver it
    sends it to accepts.
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
