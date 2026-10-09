"""`port.extensions.integer_copy`, the decoding `integer_copy_numbers.tex` specifies (#25, #6).

Referees: `cnaster`'s lattice, the paper's formulae, and Monte Carlo coverage at the
nominal rate.
"""

import numpy as np
import pytest
from cnaster.integer_copy import get_acn_baf_rdr, get_ordered_acn
from port.extensions.integer_copy import (
    CHANNELS,
    acn_lattice,
    acn_observables,
    debias_rdr,
    decode_copy_state,
    success_probability_variance,
)
from sal.emissions import BetaBinomialEmission
from sal.opt.fit import fit, parameter_covariance
from sal.opt.hmm import EmissionHmmObjective
from scipy.stats import chi2

from tests.fixtures import beta_binomial_chains

PLANTED = (2, 1)
"""A single-copy gain with one allele lost: `mubar = 1.5`, `p = 1/3`, near its lattice neighbours."""

TIGHT = np.diag([0.02**2, 0.01**2])
"""A covariance small enough that the decoding is unambiguous."""


@pytest.mark.snapshot
def test_the_lattice_is_the_one_cnaster_searches() -> None:
    """The lattice equals `cnaster`'s rebuilt `candidates`, not `get_ordered_acn`."""
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

    assert (6, 0) in get_ordered_acn()
    assert (6, 0) not in cnaster_candidates


@pytest.mark.snapshot
def test_the_minor_allele_is_the_numerator() -> None:
    """`p = B / (A + B)` per `emission.tex`, the complement of `cnaster`'s major-allele `p`."""
    observables = acn_observables([(3, 1)])

    assert observables[0, 0] == pytest.approx(2.0)
    assert observables[0, 1] == pytest.approx(0.25)

    cnaster_baf, cnaster_rdr = get_acn_baf_rdr([(3, 1)])

    assert cnaster_rdr[0] == pytest.approx(observables[0, 0])
    assert cnaster_baf[0] == pytest.approx(1.0 - observables[0, 1])


@pytest.mark.end2end
def test_decoding_recovers_the_planted_copies() -> None:
    """At the truth's own observables and a tight covariance, one pair survives."""
    mean = acn_observables([PLANTED])[0]
    result = decode_copy_state(mean, TIGHT)

    assert result.best == PLANTED
    assert result.consistent == (PLANTED,)
    assert result.identified
    assert result.consistent_with_data
    assert result.distance == pytest.approx(0.0)


@pytest.mark.analytic
def test_the_credible_set_covers_at_its_nominal_rate() -> None:
    """The 95% set covers the truth at a rate in [0.93, 0.96] over 4,000 Monte Carlo draws."""
    truth = acn_observables([PLANTED])[0]
    covariance = np.diag([0.25**2, 0.08**2])

    rng = np.random.default_rng(11)
    draws = rng.multivariate_normal(truth, covariance, size=4_000)

    hits = sum(
        PLANTED in decode_copy_state(draw, covariance).consistent for draw in draws
    )
    rate = hits / draws.shape[0]

    assert 0.93 <= rate <= 0.96, f"coverage {rate:.4f} is not near the nominal 0.95"


@pytest.mark.analytic
def test_a_wider_covariance_admits_more() -> None:
    """The set grows monotonically as the covariance widens."""
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


@pytest.mark.smoke
def test_no_integer_pair_explains_an_impossible_fit() -> None:
    """A mean far from every lattice point gives an empty set and `consistent_with_data` False."""
    impossible = np.array([1.5, 0.5])
    result = decode_copy_state(impossible, np.diag([1e-4**2, 1e-4**2]))

    assert result.consistent == ()
    assert not result.consistent_with_data
    assert not result.identified
    assert result.best in acn_lattice()
    assert result.distance > result.threshold


@pytest.mark.oracle
def test_the_threshold_is_the_chi_square_quantile() -> None:
    """The threshold is `chi2.ppf(0.95, 2)`, two degrees of freedom."""
    result = decode_copy_state(acn_observables([PLANTED])[0], TIGHT)

    assert CHANNELS == 2
    assert result.threshold == pytest.approx(float(chi2.ppf(0.95, 2)))
    assert result.threshold == pytest.approx(5.9914645, abs=1e-6)


@pytest.mark.smoke
def test_it_refuses_a_covariance_that_identifies_nothing() -> None:
    """A singular covariance is refused."""
    mean = acn_observables([PLANTED])[0]

    with pytest.raises(ValueError, match="positive definite"):
        decode_copy_state(mean, np.diag([0.01**2, 0.0]))

    with pytest.raises(ValueError, match="symmetric"):
        decode_copy_state(mean, np.array([[1e-4, 1e-5], [2e-5, 1e-4]]))

    with pytest.raises(ValueError, match="level"):
        decode_copy_state(mean, TIGHT, level=1.0)


@pytest.mark.analytic
def test_debias_satisfies_the_paper_s_constraint() -> None:
    """`debias` satisfies `emission.tex`'s constraint at non-uniform weights (#5)."""
    rng = np.random.default_rng(3)
    weights = rng.gamma(2.0, 1.0, size=64)
    raw = rng.lognormal(0.7, 0.4, size=64)

    debiased = debias_rdr(raw, weights)

    assert float(np.dot(weights, debiased) / weights.sum()) == pytest.approx(1.0)
    assert np.all(debiased > 0.0)

    already = debias_rdr(debiased, weights)
    np.testing.assert_allclose(already, debiased, rtol=1e-12)


@pytest.mark.end2end
def test_a_scale_error_empties_the_set_without_moving_the_argmin() -> None:
    """A scale error empties the credible set without moving the argmin (#5 blocks #25)."""
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


@pytest.mark.smoke
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_the_covariance_comes_from_upstream() -> None:
    """Covariance from `snakes_and_ladders` at a fitted maximum (4,800 observations), decoded (#6)."""

    fixture = beta_binomial_chains(n_states=3, sequence_length=600, n_sequences=8)
    observations = np.asarray(fixture.dataset.observations)

    # NB sal's one HMM objective over any family (sal #1189, #671).
    family = BetaBinomialEmission(
        np.full(fixture.n_states, float(fixture.trials)), fixture.alpha, fixture.beta
    )
    objective = EmissionHmmObjective(observations, family)
    start = objective.theta_from_truth(
        fixture.dataset.initial,
        fixture.dataset.transition,
        alpha=fixture.alpha,
        beta=fixture.beta,
    )
    # NB 1e-7 rather than upstream's 1e-8 default, which sits inside this objective's
    # noise floor on some platforms.
    fitted = fit(objective, start, max_iterations=400, tolerance=1e-7)
    assert fitted.converged

    covariance = np.asarray(parameter_covariance(objective, fitted.theta).detach())
    components = objective.components(fitted.theta)
    assert isinstance(components, BetaBinomialEmission)
    alpha = np.asarray(components.alpha.detach(), dtype=np.float64)
    beta = np.asarray(components.beta.detach(), dtype=np.float64)

    # NB packed as (log_initial, log_transition, alpha, beta).
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
