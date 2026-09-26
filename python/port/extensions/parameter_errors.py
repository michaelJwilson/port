r"""Parameter errors from a fitted HMM objective, with the shift propagated.

**What `decode_copy_state` needs and nothing supplies.**
`port.extensions.integer_copy.decode_copy_state` takes a state's
:math:`(\bar\mu_k, p_k)` **and its covariance**, and decodes the integer pairs
whose implied observables fall inside a credible region. `cnaster` returns no
covariance a caller can use for that, so the decode has had no second argument
to be given. This computes it.

## It is an extension, not a patch

`cnaster` has `hmm_nophasing.unpack_param_errors`, which reads
`scipy.optimize`'s ``hess_inv`` and takes its diagonal. That is a **different
quantity**: L-BFGS-B's ``hess_inv`` is the low-rank inverse-Hessian
approximation the optimizer accumulated on its way to the optimum, not the
observed information at it. It is as good as the last few steps happened to
make it, and nothing in it is checked. So this is not a drop-in replacement
for that function -- it does not compute what it computes -- and #274's
four-job rule puts what has no `cnaster` counterpart under `extensions/`.

## The conditioning check is upstream's rule, applied here

`CLAUDE.md` asks for functionality to be considered first as something
upstream already carries. `sal.opt.fit.parameter_covariance`
is that thing, and it is **not called**: it takes a `torch` objective and
differentiates with `torch`, while this differentiates the `jax` objective
`port.extensions.jax_hmm` supplies, and adapting one to the other would mean
a second implementation of the objective -- the thing being avoided.

What is taken is its *rule*, restated with attribution: refuse a point where
the observed information is singular, indefinite or badly conditioned rather
than inverting it and returning an astronomically large covariance. An
unidentifiable model and a badly-converged fit both otherwise produce
something that looks like a number. Upstream's default `rcond` is not taken
with it, because it was chosen on phylogenetic fixtures and this model is a
different one.

## The shift is not a constant, and that is the content

`port.patch.hmm_nophasing` debiases the fitted rates by the per-clone
normalizer :math:`\log Z_c`, and

.. math::
    \log Z_c = \log \sum_{g \in c} \exp(\theta_{s(g)} + \lambda_g)

is a function of :math:`\theta`. Subtracting it is therefore not a shift of
origin that leaves the covariance alone; it is a map whose Jacobian has to be
carried:

.. math::
    \frac{\partial (\theta_k - \log Z_c)}{\partial \theta_j}
      = \delta_{kj} - w^{(c)}_j,
    \qquad
    w^{(c)}_j = \!\!\sum_{g \in c,\; s(g) = j}\!\! \frac{\exp(\theta_j + \lambda_g)}{Z_c}

so :math:`J = I - \mathbf{1} w^\top` and
:math:`\Sigma_{\text{shifted}} = J \Sigma J^\top`.

**The weights sum to one, so** :math:`J\mathbf{1} = 0`: every row of
:math:`J` sums to zero, so each debiased rate is a *contrast* of the raw
ones and the map is unchanged by adding a constant to every
:math:`\theta`. The covariance it induces is therefore singular -- rank
:math:`n - 1` -- and its null direction is :math:`w`, not :math:`\mathbf{1}`:
:math:`J^\top w = 0` because :math:`w` sums to one, while
:math:`J^\top \mathbf{1} = \mathbf{1} - n w` is not zero in general.

The two are easy to conflate and say different things. The *map* loses the
overall scale; the *covariance* is flat along the weighted combination the
normalizer itself averages. Treating the shift as an additive constant would
report the raw covariance for both, which is full rank -- so it would give
the debiased rates a scale uncertainty the debiasing exists to remove.
`tests/test_parameter_errors.py` checks the rank and the null vector, and
`tests/test_jax_hmm.py` the invariance of the map.

## jax, and one implementation of the objective

The previous work is `cnaster/sandbox/hmm_nophasing_jax.py`, named because
`CLAUDE.md` puts a dependency's `sandbox/` out of scope by default and an
excursion has to say which tree it read and why. It could not be answered
from the installed path: that tree is not in the wheel, so nothing installed
differentiates this objective.

The Hessian is `jax`'s, taken of `port.extensions.jax_hmm` -- **the same
objective, differentiated, rather than a second one written to be
differentiable**. That module is pinned against `cnaster`'s own kernels, so
what is differentiated here is refereed there.

`jax` is a dependency of this repository as of #287, added with permission
because the alternative was writing the objective a third time.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from port.extensions import jax_setup  # noqa: F401  (float64, before any array)

__all__ = [
    "ParameterErrors",
    "copy_state_covariance",
    "parameter_errors",
    "shift_jacobian",
    "shift_weights",
    "shifted_covariance",
]


class ParameterErrors(NamedTuple):
    """A fit's covariance and the standard errors read off its diagonal.

    `covariance` is the full matrix rather than the diagonal alone, because
    `decode_copy_state` takes a `(2, 2)` block and the off-diagonal is a
    modelling statement there rather than a rounding error.
    """

    covariance: np.ndarray
    standard_errors: np.ndarray


RCOND = 1e-8
"""Smallest acceptable eigenvalue ratio of the observed information.

Upstream's `parameter_covariance` defaults to a value chosen for
phylogenetic fixtures. This one is stated here because the model differs and
the default was not measured on it; it is the ratio below which the
information is treated as singular rather than inverted.
"""


def observed_information(objective: Any, theta: Any) -> np.ndarray:
    """The Hessian of a **negative** log-likelihood at `theta`.

    No sign flip: the objective is a negative log-likelihood by
    `sal.opt`'s convention, so its Hessian is the observed
    Fisher information directly.

    `objective` takes one parameter vector and returns a scalar, and is
    traced by `jax`, so it must be written in `jax` operations --
    `port.extensions.jax_hmm` is what supplies them, though any `jax`
    objective works and this module does not require that one.

    `float64` comes from `port.extensions.jax_setup`, imported at module
    scope. Without it this returns a `float32` Hessian, which is good to
    about 5.8e-08 and decides identifiability on its smallest eigenvalue.
    """
    import jax

    point = jax.numpy.asarray(np.asarray(theta, dtype=np.float64).reshape(-1))

    return np.asarray(jax.hessian(objective)(point), dtype=np.float64)


def parameter_errors(
    objective: Any, theta: Any, *, rcond: float = RCOND
) -> ParameterErrors:
    """The covariance of a fitted objective, in its own coordinates.

    Parameters
    ----------
    objective
        The fitted objective: callable on a parameter vector, returning the
        scalar **negative** log-likelihood, written in `jax` operations.
    theta
        The best-fit parameters. Must be an optimum -- away from one the
        Hessian need not be positive definite, and a point that is not one is
        refused rather than returned as a covariance.
    rcond : float
        Smallest acceptable ratio of the smallest to the largest eigenvalue
        of the observed information.

    Raises
    ------
    ValueError
        Where the information is singular, indefinite or worse conditioned
        than `rcond`. All three mean the model as parameterized is not
        identifiable from this data, and are reported as that rather than as
        a linear-algebra error.
    """
    information = observed_information(objective, theta)

    # NB symmetric by construction, so `eigvalsh` is exact where a general
    #    eigensolver would introduce a spurious imaginary part. The check is
    #    upstream's, restated here because the Hessian is `jax`'s rather than
    #    `torch`'s and `parameter_covariance` takes a `torch` objective.
    eigenvalues = np.linalg.eigvalsh(information)
    smallest, largest = float(eigenvalues.min()), float(eigenvalues.max())

    if largest <= 0.0 or smallest <= rcond * largest:
        ratio = smallest / largest if largest else float("nan")
        msg = (
            f"observed information is not positive definite to within "
            f"rcond={rcond:.0e} (eigenvalue ratio {ratio:.2e}): the model is "
            f"not identifiable from this data, or the point is not a maximum"
        )
        raise ValueError(msg)

    covariance = np.linalg.inv(information)

    return ParameterErrors(
        covariance=covariance,
        standard_errors=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
    )


def shift_weights(
    log_mus: np.ndarray, copy_states: np.ndarray, normal_log_lambda: np.ndarray
) -> np.ndarray:
    """:math:`w_j`, the derivative of one clone's :math:`\\log Z` by state.

    The softmax weight of every segment decoding to state `j`, summed. One
    clone's segments, so the caller slices; `port.patch.hmm_nophasing`'s
    `shifts` computes the same reduction's value and this computes its
    gradient.

    **The result sums to one**, which is what makes the Jacobian below
    singular along the constant direction. Asserted by the caller rather
    than here, because it is a property to be checked rather than relied on.
    """
    states = np.asarray(copy_states, dtype=np.int64).reshape(-1)
    terms = np.asarray(log_mus, dtype=np.float64).reshape(-1)[states]
    terms = terms + np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)

    largest = terms.max(initial=-np.inf)

    if not np.isfinite(largest):
        # NB every term is -inf, which upstream's reduction returns -inf for.
        #    The gradient is not defined there and a zero vector says so
        #    without pretending the shift moved.
        return np.zeros(np.asarray(log_mus).size, dtype=np.float64)

    weights = np.exp(terms - largest)
    weights /= weights.sum()

    return np.bincount(states, weights=weights, minlength=np.asarray(log_mus).size)


def shift_jacobian(weights: np.ndarray) -> np.ndarray:
    """:math:`I - \\mathbf{1} w^\\top`, the map the debiasing applies."""
    column = np.asarray(weights, dtype=np.float64).reshape(-1)

    return np.eye(column.size) - np.outer(np.ones(column.size), column)


def shifted_covariance(covariance: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """:math:`J \\Sigma J^\\top`, the covariance of the debiased rates.

    Singular by construction, with rank `n - 1` and null direction `w`: the
    debiased rates are contrasts, so one combination of them is determined.
    A caller decoding integer copies wants that -- the scale is what the
    debiasing removes, and a full-rank covariance would put its uncertainty
    back.
    """
    jacobian = shift_jacobian(weights)
    shifted = jacobian @ np.asarray(covariance, dtype=np.float64)
    propagated: np.ndarray = shifted @ jacobian.T

    return propagated


def copy_state_covariance(
    rate_covariance: np.ndarray, allele_covariance: np.ndarray, state: int
) -> np.ndarray:
    """One state's `(2, 2)` block, in the order `decode_copy_state` reads.

    `(mubar, p)`, with a zero off-diagonal unless a caller measured one. That
    zero is `integer_copy`'s modelling statement rather than this function's
    approximation: at a fixed state posterior the count and allele channels
    factorize, so the M step separates and the two are estimated
    independently.
    """
    return np.array(
        [
            [float(np.asarray(rate_covariance)[state, state]), 0.0],
            [0.0, float(np.asarray(allele_covariance)[state, state])],
        ],
        dtype=np.float64,
    )
