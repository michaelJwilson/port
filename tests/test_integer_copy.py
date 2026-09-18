"""`port.integer_copy`, the decoding `integer_copy_numbers.tex` specifies.

Issues #25 and #6. The paper defines a **one-to-many** map from a fitted copy
state into the set of integer pairs consistent with its credible region;
`cnaster` returns one pair from a hill climb over an ad-hoc weighted L1
objective, and `snakes_and_ladders` has no counterpart at all. This module is
`port`'s implementation and these are its referees.

**Three referees, and the coverage one is the referee.** The lattice is pinned
against `cnaster`'s own construction, the arithmetic against the paper's
formulae, and the credible set against a Monte Carlo of its own nominal rate.
The last is the only one that can catch a region that is the right shape and
the wrong size, which is the failure a decoding would otherwise ship with.

**What is deliberately not tested here.** Whether a *fitted* `mubar` is
de-biased. `decode_copy_state` cannot check it -- the constraint is over a
clone's profile, which the function never sees -- so the module states the
dependence and `test_the_decoding_moves_under_a_scale_error` measures what
getting it wrong costs. Issue #5 owns the fix.
"""

import numpy as np
import pytest
from port.integer_copy import (
    CHANNELS,
    acn_lattice,
    acn_observables,
    debias_rdr,
    decode_copy_state,
    success_probability_variance,
)
from scipy.stats import chi2

PLANTED = (2, 1)
"""A single-copy gain with one allele lost: `mubar = 1.5`, `p = 1/3`.

Chosen because its neighbours on the lattice are close in both coordinates --
`(3, 1)` and `(2, 2)` sit `0.5` and `0` away in `mubar` -- so a region that is
even slightly too wide picks them up. A balanced diploid `(1, 1)` would be
easier and would prove less.
"""

TIGHT = np.diag([0.02**2, 0.01**2])
"""A covariance small enough that the decoding is unambiguous."""


@pytest.mark.cnaster
@pytest.mark.subject
def test_the_lattice_is_the_one_cnaster_searches() -> None:
    """Pinned against `cnaster`'s construction, not against `get_ordered_acn`.

    `hill_climbing_integer_copynumber_oneclone` builds its `candidates` from
    `max_allele_copy` and `max_total_copy` at every call. `get_ordered_acn()`
    returns a hard-coded sixteen-entry tuple that the hill climbers never
    use, and the two disagree: the list contains `(6, 0)`, whose major allele
    exceeds the `max_allele_copy = 5` it is documented beside.

    So this rebuilds `candidates` the way `cnaster` does and compares. If a
    future `cnaster` changes its bounds this fails, which is the point -- a
    comparison on a different lattice measures the lattice.
    """
    max_allele_copy, max_total_copy = 5, 6
    cnaster_candidates = {
        (first, second)
        for first in range(max_allele_copy + 1)
        for second in range(max_allele_copy + 1)
        if not (first == 0 and second == 0)
        if first + second <= max_total_copy
    }

    assert set(acn_lattice()) == cnaster_candidates
    assert len(acn_lattice()) == 25

    from cnaster.integer_copy import get_ordered_acn

    assert (6, 0) in get_ordered_acn()
    assert (6, 0) not in cnaster_candidates


@pytest.mark.cnaster
@pytest.mark.subject
def test_the_minor_allele_is_the_numerator() -> None:
    """`p = B / (A + B)`, which is `emission.tex`; `cnaster` uses the other one.

    `get_acn_baf_rdr` returns `acn[:, 0] / total`, the *major* allele, so the
    two conventions are complements. Phasing makes the labelling
    non-identifiable, so neither is wrong -- but a comparison that mixes them
    is off by `1 - p`, and at an LOH state that is the difference between
    `0` and `1`.

    Pinned at an asymmetric pair, since a balanced one cannot tell them
    apart.
    """
    observables = acn_observables([(3, 1)])

    assert observables[0, 0] == pytest.approx(2.0)
    assert observables[0, 1] == pytest.approx(0.25)

    from cnaster.integer_copy import get_acn_baf_rdr

    cnaster_baf, cnaster_rdr = get_acn_baf_rdr([(3, 1)])

    assert cnaster_rdr[0] == pytest.approx(observables[0, 0])
    assert cnaster_baf[0] == pytest.approx(1.0 - observables[0, 1])


@pytest.mark.end2end
@pytest.mark.planted
def test_decoding_recovers_the_planted_copies() -> None:
    """At the truth's own observables and a tight covariance, one pair survives."""
    mean = acn_observables([PLANTED])[0]
    result = decode_copy_state(mean, TIGHT)

    assert result.best == PLANTED
    assert result.consistent == (PLANTED,)
    assert result.identified
    assert result.consistent_with_data
    assert result.distance == pytest.approx(0.0)


@pytest.mark.infra
@pytest.mark.analytic
def test_the_credible_set_covers_at_its_nominal_rate() -> None:
    """The referee: 95 per cent of draws put the truth in the 95 per cent set.

    A region can be the right shape and the wrong size, and nothing else here
    would catch it -- every other test would pass against a region scaled by
    two. So this draws the fitted `(mubar, p)` from the sampling distribution
    the covariance asserts, decodes each draw, and counts how often the
    planted pair is in the set.

    The claim is asymptotic and the check is Monte Carlo, so it is stated
    with its own uncertainty: 4,000 draws give a binomial standard error of
    `0.0035` on a rate of `0.95`, and the tolerance below is four of those.
    Asserting equality would be a flaky test asserting a false precision.

    The measured rate is **below** nominal, and that is expected rather than
    tolerated: the set is a *discrete* subset of a continuous region, so a
    draw whose region contains no lattice point at all contributes a miss.
    The one-sided reading is what the assertion uses.
    """
    truth = acn_observables([PLANTED])[0]
    covariance = np.diag([0.25**2, 0.08**2])

    rng = np.random.default_rng(11)
    draws = rng.multivariate_normal(truth, covariance, size=4_000)

    hits = sum(
        PLANTED in decode_copy_state(draw, covariance).consistent for draw in draws
    )
    rate = hits / draws.shape[0]

    assert 0.93 <= rate <= 0.96, f"coverage {rate:.4f} is not near the nominal 0.95"


@pytest.mark.infra
@pytest.mark.analytic
def test_a_wider_covariance_admits_more() -> None:
    """The set grows monotonically with the uncertainty, and never shrinks.

    An invariant of the construction: scaling the covariance up scales every
    Mahalanobis distance down, so no point can leave. A decoding that lost a
    pair as the error bars widened would be reporting something other than a
    credible region.
    """
    mean = acn_observables([PLANTED])[0]

    previous: set[tuple[int, int]] = set()
    sizes = []
    for scale in (0.02, 0.05, 0.1, 0.25, 0.5):
        consistent = set(
            decode_copy_state(mean, np.diag([scale**2, (scale / 2.0) ** 2])).consistent
        )
        assert previous <= consistent, f"a pair left the set at scale {scale}"
        previous = consistent
        sizes.append(len(consistent))

    assert sizes[0] == 1
    assert sizes[-1] > sizes[0]


@pytest.mark.infra
@pytest.mark.analytic
def test_no_integer_pair_explains_an_impossible_fit() -> None:
    """A mean far from every lattice point returns an empty set, not a winner.

    The case `cnaster`'s argmin cannot report. `hill_climbing_*` returns its
    best candidate whatever the residual, so an eight-sigma miss and an exact
    match are the same output. Here `best` is still populated -- a caller may
    want the least-bad point -- but `consistent` is empty and
    `consistent_with_data` is False, which is the flag that says so.
    """
    impossible = np.array([1.5, 0.5])
    result = decode_copy_state(impossible, np.diag([1e-4**2, 1e-4**2]))

    assert result.consistent == ()
    assert not result.consistent_with_data
    assert not result.identified
    assert result.best in acn_lattice()
    assert result.distance > result.threshold


@pytest.mark.infra
@pytest.mark.analytic
def test_the_threshold_is_the_chi_square_quantile() -> None:
    """Two degrees of freedom, not one.

    The region is joint over `(mubar, p)`, so a per-parameter `1.96 sigma`
    box would be both the wrong shape and the wrong size -- `chi2.ppf(0.95, 2)`
    is `5.99` against `3.84` for one degree of freedom, so the box is too
    small in the joint sense and the error is in the direction that drops
    true pairs.
    """
    result = decode_copy_state(acn_observables([PLANTED])[0], TIGHT)

    assert CHANNELS == 2
    assert result.threshold == pytest.approx(float(chi2.ppf(0.95, 2)))
    assert result.threshold == pytest.approx(5.9914645, abs=1e-6)


@pytest.mark.infra
@pytest.mark.analytic
def test_it_refuses_a_covariance_that_identifies_nothing() -> None:
    """A singular covariance raises rather than returning the whole lattice.

    The refusal `snakes_and_ladders`' `parameter_covariance` makes for the
    same reason: a flat direction is a parameter the data did not determine,
    and "every integer is possible" reported as a measurement is worse than
    an error.
    """
    mean = acn_observables([PLANTED])[0]

    with pytest.raises(ValueError, match="positive definite"):
        decode_copy_state(mean, np.diag([0.01**2, 0.0]))

    with pytest.raises(ValueError, match="symmetric"):
        decode_copy_state(mean, np.array([[1e-4, 1e-5], [2e-5, 1e-4]]))

    with pytest.raises(ValueError, match="level"):
        decode_copy_state(mean, TIGHT, level=1.0)


@pytest.mark.infra
@pytest.mark.analytic
def test_debias_satisfies_the_paper_s_constraint() -> None:
    """`sum_g lambda_g mubar_g / sum_g lambda_g == 1` after rescaling.

    `emission.tex`'s constraint, which `cnaster` computes in
    `compute_logmu_shifts` and applies nowhere (#5). Checked at a
    non-uniform weight, since a uniform one would pass for any function that
    divides by a mean.
    """
    rng = np.random.default_rng(3)
    weights = rng.gamma(2.0, 1.0, size=64)
    raw = rng.lognormal(0.7, 0.4, size=64)

    debiased = debias_rdr(raw, weights)

    assert float(np.dot(weights, debiased) / weights.sum()) == pytest.approx(1.0)
    assert np.all(debiased > 0.0)

    already = debias_rdr(debiased, weights)
    np.testing.assert_allclose(already, debiased, rtol=1e-12)


@pytest.mark.end2end
@pytest.mark.planted
def test_a_scale_error_empties_the_set_without_moving_the_argmin() -> None:
    """The missing de-biasing corrupts the confidence, not the answer.

    `mubar = (A + B) / 2` is a statement about an *absolute* scale. A fitted
    `mu` carries an unknown per-clone factor until `compute_logmu_shifts` is
    applied, and `cnaster` never applies it (#5), so the input to any
    decoding is wrong by that factor.

    Measured, at `(4, 2)` with `sigma_mubar = 0.08`:

    | scale error | `best` | `distance` | set size |
    | ---: | --- | ---: | ---: |
    | 0 % | `(4, 2)` | 0.00 | 1 |
    | 2 % | `(4, 2)` | 0.56 | 1 |
    | 5 % | `(4, 2)` | 3.52 | 1 |
    | 8 % | `(4, 2)` | 9.00 | **0** |
    | 20 % | `(4, 2)` | 56.25 | **0** |

    **The argmin never moves.** The lattice spacing in `mubar` is `0.5` and
    the nearest competitor stays further away than the bias, so the closest
    point is the right one at every error tried. What collapses is the
    credible set: past eight per cent no integer pair is inside the region at
    all.

    That is the argument for the paper's one-to-many map in one table.
    `cnaster` reports the argmin and nothing else, so at a twenty per cent
    scale error it returns `(4, 2)` -- the correct pair -- with no indication
    that the fit is fifty-six chi-square units from explaining it. The
    decoding here returns the same pair and an empty set, and
    `consistent_with_data` is the flag that says the answer is not one.

    Not a defect in this module: the input's, and the reason #5 blocks #25.
    """
    high = (4, 2)
    truth = acn_observables([high])[0]
    covariance = np.diag([0.08**2, 0.03**2])

    distances = []
    for error in (0.0, 0.02, 0.05, 0.08, 0.20):
        biased = truth.copy()
        biased[0] *= 1.0 + error
        result = decode_copy_state(biased, covariance)

        assert result.best == high, (
            f"the argmin moved at a {error:.0%} scale error; the point of this "
            "test is that it does not, so the fixture has changed"
        )
        distances.append(result.distance)

        if error <= 0.05:
            assert result.consistent_with_data
            assert result.consistent == (high,)
        else:
            assert not result.consistent_with_data
            assert result.consistent == ()

    assert distances == sorted(distances), "the residual is not monotone in the bias"
    assert distances[-1] > 50.0


@pytest.mark.infra
@pytest.mark.upstream
def test_the_covariance_comes_from_upstream() -> None:
    """End to end: fit with `snakes_and_ladders`, propagate, decode.

    The route `CLAUDE.md` asks for -- upstream already carries the
    observed-information stack, so `port` calls it rather than writing one.
    `jax` is not needed and is not a dependency.

    Two things this pins beyond "it runs".

    **The fit must be at a maximum.** `parameter_covariance` refuses
    otherwise, and it refused the planted parameters of this very fixture
    during development: the truth is not the MLE of a finite sample, and the
    observed information there is not positive definite. So the test fits
    first and takes the covariance at `result.theta`.

    **The size is not arbitrary.** At 1,200 observations the information's
    eigenvalue ratio falls below `parameter_covariance`'s `rcond` and the
    call refuses; at 4,800 it does not. That is a real statement about what
    this model needs to identify its parameters, and it is why the fixture
    here is the larger one. #6 owns the negative binomial half.
    """
    from snakes_and_ladders.opt.fit import fit, parameter_covariance
    from snakes_and_ladders.opt.hmm import BetaBinomialHmmObjective

    from tests.fixtures import beta_binomial_chains

    fixture = beta_binomial_chains(n_states=3, sequence_length=600, n_sequences=8)
    observations = np.asarray(fixture.dataset.observations)

    objective = BetaBinomialHmmObjective(
        observations, fixture.n_states, np.full(fixture.n_states, float(fixture.trials))
    )
    start = objective.theta_from_truth(
        fixture.dataset.initial,
        fixture.dataset.transition,
        fixture.alpha,
        fixture.beta,
    )
    # NB the tolerance is stated rather than left to upstream's 1e-8 default,
    #    which sits inside this objective's noise floor. At the optimum the
    #    relative gradient is dominated by double-precision cancellation, and
    #    where that floor lands is the platform's: this machine reaches
    #    3.52e-9 in 4 iterations, a GitHub runner plateaus at 1.13e-8 through
    #    all 400. Both find the *same* maximum -- value 16851.46390216068 on
    #    both, agreeing to eleven significant figures -- so the default was
    #    pinning which side of the noise the arithmetic fell on, not whether
    #    the fit converged. 1e-7 is above the floor and still four orders
    #    inside the curvature `parameter_covariance` needs.
    fitted = fit(objective, start, max_iterations=400, gradient_tolerance=1e-7)
    assert fitted.converged

    covariance = np.asarray(parameter_covariance(objective, fitted.theta).detach())
    emissions = objective.emissions(fitted.theta)
    alpha = np.asarray(emissions.alpha.detach(), dtype=np.float64)
    beta = np.asarray(emissions.beta.detach(), dtype=np.float64)

    # NB the packed vector is (log_initial, log_transition, alpha, beta), so
    #    the per-state pair sits at these two offsets.
    n_states = fixture.n_states
    first_alpha = covariance.shape[0] - 2 * n_states

    for state in range(n_states):
        index_alpha = first_alpha + state
        index_beta = first_alpha + n_states + state
        block = covariance[np.ix_([index_alpha, index_beta], [index_alpha, index_beta])]

        variance_p = success_probability_variance(
            float(alpha[state]), float(beta[state]), block
        )
        assert variance_p > 0.0

        mean = np.array(
            [1.0, float(alpha[state] / (alpha[state] + beta[state]))], dtype=np.float64
        )
        result = decode_copy_state(mean, np.diag([0.05**2, variance_p]))

        assert result.best in acn_lattice()
        assert result.threshold == pytest.approx(float(chi2.ppf(0.95, 2)))
