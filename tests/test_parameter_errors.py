"""Parameter errors from the fitted objective and the shift's Jacobian (#287).

Referees: closed-form Gaussian, `jax` gradients, Monte Carlo of the map, and planted
integer copies.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
from port.extensions.integer_copy import decode_copy_state
from port.patch.hmm_nophasing import shifts
from port.qa.jax_hmm import emission, marginal_negative_log_likelihood, shifted_rates
from port.qa.parameter_errors import (
    copy_state_covariance,
    parameter_errors,
    shift_jacobian,
    shift_weights,
    shifted_covariance,
)
from scipy.optimize import minimize


@pytest.mark.analytic
def test_the_covariance_of_a_gaussian_is_its_variance() -> None:
    """Covariance of `-log N(x; mu, S)` is `S` within 1e-10 (closed form)."""

    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    precision = np.linalg.inv(covariance)

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        centred = theta - jnp.asarray([1.0, -2.0])

        return 0.5 * centred @ jnp.asarray(precision) @ centred

    errors = parameter_errors(objective, np.array([1.0, -2.0]))

    np.testing.assert_allclose(errors.covariance, covariance, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(
        errors.standard_errors, np.sqrt(np.diag(covariance)), rtol=1e-10, atol=0.0
    )


def _flat(theta: jnp.ndarray) -> jnp.ndarray:
    """Curved in the sum, flat in the difference: an unfixed gauge."""
    return 0.5 * (theta[0] + theta[1]) ** 2


def _saddle(theta: jnp.ndarray) -> jnp.ndarray:
    """Indefinite information."""
    return 0.5 * theta[0] ** 2 - 0.5 * theta[1] ** 2


@pytest.mark.analytic
@pytest.mark.parametrize("objective", [_flat, _saddle], ids=["flat", "saddle"])
def test_a_flat_direction_or_a_saddle_is_refused(objective: Any) -> None:
    """A flat direction raises rather than inverting to a ~1e16 variance; a saddle is refused."""
    with pytest.raises(ValueError, match="not identifiable"):
        parameter_errors(objective, np.array([0.0, 0.0]))


def _weights_case(
    n_states: int = 4, n_segments: int = 20, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)

    return (
        generator.normal(0.0, 0.4, size=n_states),
        generator.integers(0, n_states, size=n_segments).astype(np.int64),
        np.log(generator.random(n_segments) / n_segments),
    )


@pytest.mark.analytic
def test_the_shift_weights_sum_to_one() -> None:
    """Shift weights are a distribution over one clone's segments, gathered by state."""

    log_mus, copy_states, normal_log_lambda = _weights_case()

    weights = shift_weights(log_mus, copy_states, normal_log_lambda)

    assert weights.shape == log_mus.shape

    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.oracle
def test_the_shift_weights_are_the_jax_derivative() -> None:
    """`w` matches `jax`'s gradient of `log Z` within 1e-12 and sums to one."""

    log_mus, copy_states, normal_log_lambda = _weights_case()

    def log_normalizer(rates: jnp.ndarray) -> jnp.ndarray:
        return jsp.logsumexp(rates[copy_states] + jnp.asarray(normal_log_lambda))

    theirs = np.asarray(jax.grad(log_normalizer)(jnp.asarray(log_mus)))
    ours = shift_weights(log_mus, copy_states, normal_log_lambda)

    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-12)


@pytest.mark.analytic
def test_the_propagated_covariance_is_singular_with_null_direction_w() -> None:
    """`J S J'` has rank `n - 1` with null vector `w`, not `1`."""

    log_mus, copy_states, normal_log_lambda = _weights_case()
    n_states = log_mus.size

    weights = shift_weights(log_mus, copy_states, normal_log_lambda)
    constant = np.ones(n_states)

    np.testing.assert_allclose(
        shift_jacobian(weights) @ constant, np.zeros(n_states), rtol=0.0, atol=1e-12
    )

    generator = np.random.default_rng(5)
    root = generator.normal(size=(n_states, n_states))
    covariance = root @ root.T + n_states * np.eye(n_states)

    shifted = shifted_covariance(covariance, weights)

    # NB full rank before, rank `n - 1` after.
    assert np.linalg.matrix_rank(covariance) == n_states
    assert np.linalg.matrix_rank(shifted, tol=1e-10) == n_states - 1

    # NB `1` is asserted not null, which is what makes `w` null a statement.
    np.testing.assert_allclose(weights @ shifted @ weights, 0.0, rtol=0.0, atol=1e-10)

    assert constant @ shifted @ constant > 1e-6

    np.testing.assert_allclose(shifted, shifted.T, rtol=0.0, atol=1e-12)

    assert np.linalg.eigvalsh(shifted).min() > -1e-10


@pytest.mark.oracle
def test_the_propagated_covariance_is_the_delta_method() -> None:
    """`J S J'` matches a 2,000-draw Monte Carlo of the debiasing within 5e-2 of the max."""

    log_mus, copy_states, normal_log_lambda = _weights_case(n_states=3, n_segments=12)
    n_states = log_mus.size
    lengths = np.array([normal_log_lambda.size], dtype=np.int64)

    generator = np.random.default_rng(11)
    root = 0.05 * generator.normal(size=(n_states, n_states))
    covariance = root @ root.T + 0.01 * np.eye(n_states)

    weights = shift_weights(log_mus, copy_states, normal_log_lambda)
    predicted = shifted_covariance(covariance, weights)

    draws = generator.multivariate_normal(log_mus, covariance, size=2_000)
    debiased = np.stack(
        [
            np.asarray(shifted_rates(draw, copy_states, normal_log_lambda, lengths))[0]
            for draw in draws
        ]
    )

    realized = np.cov(debiased, rowvar=False)

    scale = np.abs(predicted).max()

    np.testing.assert_allclose(realized, predicted, rtol=0.0, atol=5e-2 * scale)


@pytest.mark.smoke
def test_a_copy_state_block_is_what_decode_copy_state_reads() -> None:
    """The `(2, 2)` `(mubar, p)` block with zero off-diagonal, as `decode_copy_state` reads."""

    rates = np.diag([0.04, 0.09, 0.16])
    alleles = np.diag([0.0025, 0.01, 0.0225])

    block = copy_state_covariance(rates, alleles, 1)

    assert block.shape == (2, 2)
    assert block[0, 0] == pytest.approx(0.09)
    assert block[1, 1] == pytest.approx(0.01)
    assert block[0, 1] == 0.0
    assert block[1, 0] == 0.0


DECODED = ((1, 1), (2, 1), (1, 0))
"""Diploid, single-allele gain and deletion; equal gain/deletion mass makes the mean `mubar` one."""

LIBRARY = 1.7
"""Per-clone library factor planted in the counts, removed only by the shift."""


def _planted_genome(seed: int = 23) -> dict[str, np.ndarray]:
    """One clone, 600 bins in blocks of 50 from `DECODED`; `p_binom` is the major fraction."""
    generator = np.random.default_rng(seed)

    blocks = np.array([0, 1, 0, 2, 0, 1, 2, 0, 1, 0, 2, 0])
    states = np.repeat(blocks, 50).astype(np.int64)
    n_obs = states.size

    weights = generator.uniform(0.5, 1.5, n_obs)
    weights[states == 2] *= weights[states == 1].sum() / weights[states == 2].sum()
    weights /= weights.sum()

    copies = np.asarray(DECODED, dtype=np.float64)
    mubar = copies.sum(axis=1) / 2.0
    p_binom = copies[:, 0] / copies.sum(axis=1)

    alpha, tau = 0.04, 40.0
    exposure = 6.0e4 * weights
    mean = exposure * LIBRARY * mubar[states]

    counts_nb = generator.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))
    trials = generator.integers(20, 50, n_obs)
    # NB the deletion's fraction is drawn as 1/2 and replaced by the model's 1.
    interior = np.where(states == 2, 0.5, p_binom[states])
    majors = generator.beta(interior * tau, (1.0 - interior) * tau)
    counts_bb = generator.binomial(trials, np.where(states == 2, 1.0, majors))

    return {
        "states": states,
        "normal_log_lambda": np.log(weights),
        "counts_nb": counts_nb.astype(np.float64),
        "base_nb_mean": exposure,
        "counts_bb": counts_bb.astype(np.float64),
        "total_bb_RD": trials.astype(np.float64),
        "mubar": mubar,
    }


@pytest.mark.end2end
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_a_fit_s_errors_decode_the_planted_integer_copies() -> None:
    """Fit, debias, propagate and decode recovers the planted pairs within stated tolerances (#287)."""

    genome = _planted_genome()
    n_states = len(DECODED)
    lengths = np.array([genome["states"].size], dtype=np.int64)

    log_startprob = np.full(n_states, -np.log(n_states))
    stay = 0.99
    log_transmat = np.log(
        np.full((n_states, n_states), (1.0 - stay) / (n_states - 1))
        + (stay - (1.0 - stay) / (n_states - 1)) * np.eye(n_states)
    )

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        majors = jax.nn.sigmoid(theta[3:5])
        log_emission = emission(
            theta[0:3],
            jnp.full(n_states, jnp.exp(theta[5])),
            jnp.concatenate([majors, jnp.ones(1)]),
            jnp.full(n_states, jnp.exp(theta[6])),
            genome["counts_nb"],
            genome["base_nb_mean"],
            genome["counts_bb"],
            genome["total_bb_RD"],
        )

        return marginal_negative_log_likelihood(
            log_emission, log_startprob, log_transmat, lengths
        )

    # NB started 0.2 off in every rate and 0.3 in every logit, not at a neutral point.
    truth = np.log(LIBRARY * genome["mubar"])
    start = np.concatenate([truth + 0.2, [0.3, 0.9], [np.log(0.1), np.log(20.0)]])

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(theta))

        return float(value), np.asarray(gradient, dtype=np.float64)

    fit = minimize(
        scipy_objective, start, jac=True, method="BFGS", options={"gtol": 1e-6}
    )

    errors = parameter_errors(objective, fit.x)

    # NB the Newton decrement `g' S g` measures the remaining step in chi-square units.
    gradient = scipy_objective(fit.x)[1]
    decrement = float(gradient @ errors.covariance @ gradient)

    assert decrement < 1e-6, f"not at an optimum: Newton decrement {decrement:.2e}"

    # NB the zeroed rate-allele off-diagonal holds only approximately; asserted.
    scales = np.sqrt(np.diag(errors.covariance))
    correlation = errors.covariance / np.outer(scales, scales)

    assert np.abs(correlation[0:3, 3:5]).max() < 0.1

    log_mus = fit.x[0:3]
    states = genome["states"]
    lambdas = genome["normal_log_lambda"]

    debiased = log_mus - shifts(log_mus, states, lambdas, lengths)[0]
    rate_covariance = shifted_covariance(
        errors.covariance[0:3, 0:3], shift_weights(log_mus, states, lambdas)
    )

    mubar = np.exp(debiased)
    mubar_covariance = rate_covariance * np.outer(mubar, mubar)

    majors = 1.0 / (1.0 + np.exp(-fit.x[3:5]))
    slope = majors * (1.0 - majors)
    allele_covariance = errors.covariance[3:5, 3:5] * np.outer(slope, slope)

    standardized = (mubar - genome["mubar"]) / np.sqrt(np.diag(mubar_covariance))

    assert np.abs(standardized).max() < 3.0, f"mubar residuals {standardized}"

    for state, planted in enumerate(DECODED[:2]):
        block = copy_state_covariance(mubar_covariance, allele_covariance, state)
        result = decode_copy_state([mubar[state], 1.0 - majors[state]], block)

        assert result.best == planted, f"state {state}: {result.best} != {planted}"
        assert result.consistent == (planted,), f"state {state}: {result.consistent}"
        assert result.consistent_with_data

        # NB control: the raw rate, library factor included, must fail to decode.
        raw = decode_copy_state([np.exp(log_mus[state]), 1.0 - majors[state]], block)

        assert planted not in raw.consistent, f"state {state} decoded undebiased"

    # NB the deletion's allele fraction is not fitted, so its rate is judged alone.
    deletion = DECODED[2]
    residual = (mubar[2] - sum(deletion) / 2.0) ** 2 / mubar_covariance[2, 2]

    assert residual < 3.0**2, f"deletion mubar at {np.sqrt(residual):.2f} sigma"
