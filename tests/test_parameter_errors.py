"""Parameter errors from the fitted objective, and the shift's Jacobian (#287).

What `port.extensions.integer_copy.decode_copy_state` takes as its second
argument, and what `cnaster` does not produce. Three claims:

*The covariance is the inverse observed information.* Checked on a Gaussian,
whose answer is known in closed form without computing it.

*A fit that is not at a maximum is refused.* The failure mode this exists to
prevent: an unidentifiable model inverts to an astronomically large
covariance that still looks like a number.

*The shift's Jacobian is singular along the constant direction.* The one that
needs the derivation rather than a library -- and the reason the shift cannot
be treated as an additive constant.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.analytic
def test_the_covariance_of_a_gaussian_is_its_variance() -> None:
    """`-log N(x; mu, S)` has Hessian `S^-1`, so the covariance is `S`.

    A case whose answer is known without computing it, which is what makes
    it a check on the machinery rather than on a second implementation.
    Realized **exact** -- 0.0 relative -- against a stated 1e-10, in
    `float64`; the tolerance is kept because exactness here is a property of
    a 2x2 inverse rather than a contract.
    """
    import jax.numpy as jnp
    from port.extensions.parameter_errors import parameter_errors

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


@pytest.mark.analytic
def test_a_flat_direction_is_refused_rather_than_inverted() -> None:
    """An unidentifiable model raises instead of returning a huge covariance.

    The whole reason the conditioning is checked. A model with an exactly
    flat direction produces an information matrix that is only *numerically*
    singular -- rounding leaves its smallest eigenvalue near zero rather than
    at it -- so `inv` succeeds and reports a variance of about 1e16, which a
    caller has no way to recognize as "not estimable".
    """
    import jax.numpy as jnp
    from port.extensions.parameter_errors import parameter_errors

    def flat(theta: jnp.ndarray) -> jnp.ndarray:
        # NB curved in the sum, flat in the difference: the classic
        #    unfixed-gauge shape, and what an over-parameterized fit looks
        #    like.
        return 0.5 * (theta[0] + theta[1]) ** 2

    with pytest.raises(ValueError, match="not identifiable"):
        parameter_errors(flat, np.array([0.0, 0.0]))


@pytest.mark.analytic
def test_a_saddle_is_refused() -> None:
    """A point that is not a maximum is not a fit, and is reported as such.

    Separate from the flat case because the diagnosis differs: here the
    information is *indefinite* rather than singular, and inverting it gives
    a negative variance -- a standard error that is `nan`, which propagates
    silently.
    """
    import jax.numpy as jnp
    from port.extensions.parameter_errors import parameter_errors

    def saddle(theta: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * theta[0] ** 2 - 0.5 * theta[1] ** 2

    with pytest.raises(ValueError, match="not identifiable"):
        parameter_errors(saddle, np.array([0.0, 0.0]))


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
    """Softmax weights over one clone's segments, gathered by state.

    The property everything below rests on: `w` is a distribution, so
    `J = I - 1 w'` annihilates the constant vector.
    """
    from port.extensions.parameter_errors import shift_weights

    log_mus, copy_states, normal_log_lambda = _weights_case()

    weights = shift_weights(log_mus, copy_states, normal_log_lambda)

    assert weights.shape == log_mus.shape

    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.oracle
def test_the_shift_weights_are_the_jax_derivative() -> None:
    """`w` against `jax`'s own gradient of `log Z` -- two routes, one number.

    The weights are written by hand here because the covariance propagation
    needs them as a matrix, and a hand-written derivative is exactly the
    thing to referee. `jax` differentiates
    `port.extensions.jax_hmm.shifted_rates`' reduction directly.

    Realized **1.11e-16** absolute against a stated 1e-12, and the weights
    sum to one exactly.
    """
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jsp
    from port.extensions.parameter_errors import shift_weights

    log_mus, copy_states, normal_log_lambda = _weights_case()

    def log_normalizer(rates: jnp.ndarray) -> jnp.ndarray:
        return jsp.logsumexp(rates[copy_states] + jnp.asarray(normal_log_lambda))

    theirs = np.asarray(jax.grad(log_normalizer)(jnp.asarray(log_mus)))
    ours = shift_weights(log_mus, copy_states, normal_log_lambda)

    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-12)


@pytest.mark.analytic
def test_the_propagated_covariance_is_singular_with_null_direction_w() -> None:
    """`J 1 = 0` makes the rates contrasts; the null vector is `w`, not `1`.

    **The two are easy to conflate and this test exists because they were.**
    `J 1 = 0` says every row sums to zero, so the *map* is unchanged by
    adding a constant to every `log_mu`. What follows for the *covariance*
    is that it is rank `n - 1`, with `J' w = 0` -- and `J' 1 = 1 - n w`,
    which is not zero, so `1' J S J' 1` is a perfectly ordinary positive
    number. Asserting that one was zero is the error this caught.

    Treating the shift as an additive constant would leave the covariance
    full rank, giving the debiased rates a scale uncertainty the debiasing
    exists to remove.
    """
    from port.extensions.parameter_errors import (
        shift_jacobian,
        shift_weights,
        shifted_covariance,
    )

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

    # NB the fixture has to have something to lose for the loss to be
    #    evidence: full rank before, rank `n - 1` after.
    assert np.linalg.matrix_rank(covariance) == n_states
    assert np.linalg.matrix_rank(shifted, tol=1e-10) == n_states - 1

    # NB `w` is the null direction, and `1` is not. Both asserted, because
    #    the second is what makes the first a statement.
    np.testing.assert_allclose(weights @ shifted @ weights, 0.0, rtol=0.0, atol=1e-10)

    assert constant @ shifted @ constant > 1e-6

    # NB symmetric and positive semi-definite still: a propagated covariance
    #    that lost either would be a different kind of wrong from being
    #    singular on purpose.
    np.testing.assert_allclose(shifted, shifted.T, rtol=0.0, atol=1e-12)

    assert np.linalg.eigvalsh(shifted).min() > -1e-10


@pytest.mark.oracle
def test_the_propagated_covariance_is_the_delta_method() -> None:
    """`J S J'` against a Monte Carlo of the map it linearizes.

    The delta method is an approximation, so the referee is a draw from the
    fit's own distribution pushed through the **exact** debiasing. They agree
    because the map is exactly linear in `log_mu` to first order and the
    fixture's covariance is small enough that the second order does not bite.

    Realized **2.78e-02** of the largest predicted entry, against a stated
    5e-02 over 2,000 draws -- a Monte Carlo tolerance, stated as one, and
    the reason it is not tighter is the draw count rather than the method.
    """
    from port.extensions.jax_hmm import shifted_rates
    from port.extensions.parameter_errors import shift_weights, shifted_covariance

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
    """`(2, 2)`, `(mubar, p)` in that order, ready for the decode.

    The seam this extension exists for. `decode_copy_state` documents a zero
    off-diagonal as a modelling statement -- at a fixed state posterior the
    two channels factorize -- so it is zero here rather than absent.
    """
    from port.extensions.parameter_errors import copy_state_covariance

    rates = np.diag([0.04, 0.09, 0.16])
    alleles = np.diag([0.0025, 0.01, 0.0225])

    block = copy_state_covariance(rates, alleles, 1)

    assert block.shape == (2, 2)
    assert block[0, 0] == pytest.approx(0.09)
    assert block[1, 1] == pytest.approx(0.01)
    assert block[0, 1] == 0.0
    assert block[1, 0] == 0.0
