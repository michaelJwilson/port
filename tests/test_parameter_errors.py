"""Parameter errors from the fitted objective, and the shift's Jacobian (#287).

What `port.extensions.integer_copy.decode_copy_state` takes as its second
argument, and what `cnaster` does not produce. Four claims:

*The covariance is the inverse observed information.* Checked on a Gaussian,
whose answer is known in closed form without computing it.

*A fit that is not at a maximum is refused.* The failure mode this exists to
prevent: an unidentifiable model inverts to an astronomically large
covariance that still looks like a number.

*The shift's Jacobian is singular along the constant direction.* The one that
needs the derivation rather than a library -- and the reason the shift cannot
be treated as an additive constant.

*The chain decodes planted integer copies.* Fit, Hessian, debiasing, delta
method and decode, on a genome whose pairs are known -- the use the three
above exist for.
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


DECODED = ((1, 1), (2, 1), (1, 0))
"""Balanced diploid, a single-allele gain, and a deletion, in state order.

The deletion is what makes the planted genome satisfy the paper's constraint
`sum_g lambda_g mubar_g = 1`: every state with both alleles present has
`mubar >= 1`, so without a state below one the lambda-weighted mean exceeds
it and `(A + B) / 2` is not what the debiased rate estimates. `_planted_genome`
gives the gain and the deletion equal lambda mass, so the mean is exactly one.
"""

LIBRARY = 1.7
"""The per-clone library factor the debiasing exists to remove.

Planted into the counts and absent from `DECODED`, so a fitted `log_mu` is
`log(1.7 mubar)` and only the shift takes it back to `log mubar`.
"""


def _planted_genome(seed: int = 23) -> dict[str, np.ndarray]:
    """One clone, 600 bins in blocks of 50, drawn from `DECODED`.

    `p_binom` is `cnaster`'s **major** allele fraction, `1 - p` in the paper's
    convention, so the deletion's is exactly 1: every allele read is from the
    one copy left, which is what the planted `counts_bb = trials` says.
    """
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
    # NB the deletion's fraction is drawn as 1/2 and discarded: a beta at 1
    #    has no second shape parameter, and the draw is replaced by the 1 the
    #    model says.
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
def test_a_fit_s_errors_decode_the_planted_integer_copies() -> None:
    """Fit, differentiate, debias, propagate, decode -- against planted pairs.

    **The use #287 exists for.** Everything above checks a piece; this runs
    the chain `decode_copy_state` sits at the end of, on a genome whose
    integer copies are known, with a library factor of 1.7 planted so the
    debiasing has something to remove.

    *The fit.* The marginal negative log-likelihood of
    `port.extensions.jax_hmm`, maximized by `scipy`'s BFGS over `log_mu`, the
    logit of the two interior `p_binom`, and a shared `log alpha` and
    `log tau`. The deletion's `p_binom` is fixed at 1: it sits on the
    boundary, where the observed information is not the covariance of
    anything, so it is neither fitted nor decoded -- its `log_mu` is both, and
    is what balances the constraint. The transitions are fixed at a sticky
    matrix; they are not what is being decoded. The module fits nothing, and
    this does not change that: the test fits so there is a fit to take errors
    of, and the objective it fits is the one `tests/test_jax_hmm.py` pins to
    `cnaster`'s kernels.

    *The debiasing.* The shift's value is `port.patch.hmm_nophasing.shifts`,
    the `numba` kernel the patched pipeline runs; its Jacobian is
    `shift_weights`, and the covariance is `J S J'` on the rate block.

    *The decode.* `(mubar, p)` by the delta method from `(log mubar,
    logit p_binom)`, with `p` the minor fraction `decode_copy_state` reads.

    Stated tolerances, and realized on seed 23: the Newton decrement at the
    fit below 1e-6, realized 3.3e-12; the rate-allele correlation below 0.1,
    realized 0.073; each debiased `mubar` within three standard errors of the
    planted one, realized 0.78, -1.57 and 1.59; each interior state decoded
    to exactly its planted pair inside the 95 per cent region, realized at
    squared distances 1.43 and 2.55 against 5.99; and the undebiased rate
    decoding to anything else. Seeds 1 to 6 pass the same assertions.
    """
    import jax
    import jax.numpy as jnp
    from port.extensions.integer_copy import decode_copy_state
    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood
    from port.extensions.parameter_errors import (
        copy_state_covariance,
        parameter_errors,
        shift_weights,
        shifted_covariance,
    )
    from port.patch.hmm_nophasing import shifts
    from scipy.optimize import minimize

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

    # NB started away from the truth by 0.2 in every rate and 0.3 in every
    #    logit, so the fit has distance to cover; not from a neutral point,
    #    because a label switch is a different test.
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

    # NB BFGS stops on a gradient in the parameters' own units, which says
    #    nothing about whether the remaining step matters. The Newton
    #    decrement `g' S g` is that step in chi-square units, so it is what
    #    decides whether the Hessian was taken at the optimum.
    gradient = scipy_objective(fit.x)[1]
    decrement = float(gradient @ errors.covariance @ gradient)

    assert decrement < 1e-6, f"not at an optimum: Newton decrement {decrement:.2e}"

    # NB `copy_state_covariance` zeros the rate-allele off-diagonal as a
    #    modelling statement. The marginal likelihood couples the channels
    #    through the state posterior, so it holds approximately here and the
    #    size of the approximation is asserted rather than assumed.
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

        # NB the control: the same decode on the raw rate, library factor
        #    and all, which is what reading `cnaster`'s `log_mu` directly
        #    would do. Refused, so the debiasing is the step that decoded.
        raw = decode_copy_state([np.exp(log_mus[state]), 1.0 - majors[state]], block)

        assert planted not in raw.consistent, f"state {state} decoded undebiased"

    # NB the deletion's allele fraction is not fitted, so its rate is judged
    #    alone: the one channel whose error this computed.
    deletion = DECODED[2]
    residual = (mubar[2] - sum(deletion) / 2.0) ** 2 / mubar_covariance[2, 2]

    assert residual < 3.0**2, f"deletion mubar at {np.sqrt(residual):.2f} sigma"
