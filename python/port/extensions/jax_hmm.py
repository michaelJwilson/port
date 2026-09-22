r"""The HMM objective in `jax`, so it can be differentiated (#287).

**Why this exists, and why it is an extension rather than a patch.**
`cnaster`'s emission and forward recursion are `numba` kernels writing into
buffers. That is the right shape for evaluating them and the wrong one for
differentiating them: `numba` carries no derivative, so nothing downstream
can ask for a gradient, let alone a Hessian. The quantity this repository
wants -- **the observed information at the fit, and from it the parameter
errors `port.extensions.integer_copy.decode_copy_state` takes as its second
argument** -- is not something `cnaster` computes at all. #274's four-job
rule puts what has no `cnaster` counterpart under `extensions/`, so that is
where it is, and `port.patch.hmm_nophasing` is untouched.

## Following `cnaster/sandbox/hmm_nophasing_jax.py`

That file is the previous work and is where this comes from. Naming it is
required rather than polite: `CLAUDE.md` puts a dependency's `sandbox/` out
of scope **by default**, and an excursion has to say which tree it read and
why the question could not be answered from the installed path. It could
not: the sandbox tree is not in the wheel, so nothing installed differentiates
this objective, and the alternative is a third implementation of it.

Followed *loosely*, and the differences are stated:

*One implementation of the objective, not two.* The sandbox file overrides
`_run_optimization_pipeline` and fits with its own `scipy` call, so the jax
objective and `cnaster`'s run side by side and a divergence between them
shows up as a different fit. Here the jax form is only ever **read** --
nothing fits with it -- and `tests/test_jax_hmm.py` pins it against
`cnaster`'s own kernels on the same inputs. It is a second implementation of
one quantity, which is what `oracle` means, rather than a second route to a
different answer.

*The errors come from the Hessian, not from `hess_inv`.* The sandbox returns
`scipy.optimize`'s `hess_inv` -- the low-rank inverse-Hessian approximation
L-BFGS-B accumulated on its way to the optimum, as good as its last few steps
happened to make it. `port.extensions.parameter_errors` differentiates this
objective twice instead.

*One column.* The sandbox carries the spot axis the fit cannot fill; these
take `(n_states,)` or `(n_states, 1)` and nothing else (#278).

## `float64`

`jax` defaults to `float32`, which would make every comparison below a
tolerance of about `1e-7`. `port.extensions.jax_setup` is where the flag is
set and why it cannot be set anywhere later.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from port.extensions import jax_setup  # noqa: F401  (float64, before any array)
from port.patch.plotting.clone_paths import state_vector

__all__ = [
    "emission",
    "marginal_negative_log_likelihood",
    "shifted_rates",
]


def _column(values: Any) -> jnp.ndarray:
    """A state parameter as `(n_states,)`, tracer or array.

    `state_vector` refuses anything but the two shapes a fit produces, but it
    calls `np.asarray`, which a `jax` tracer does not survive. So a tracer is
    reshaped directly and only a concrete array is put through the guard --
    the shape is checked where it can be, and never silently.
    """
    if isinstance(values, jax.core.Tracer):
        return values.reshape(-1)

    return jnp.asarray(state_vector(values))


def emission(
    log_mu: Any,
    alphas: Any,
    p_binom: Any,
    taus: Any,
    counts_nb: Any,
    base_nb_mean: Any,
    counts_bb: Any,
    total_bb_RD: Any,
) -> jnp.ndarray:
    """`(n_states, n_obs)` log emission, differentiable in the parameters.

    The negative binomial in the `(r, p)` parameterization `cnaster` uses --
    `r = 1 / alpha`, `p = 1 / (1 + alpha * lambda)` with
    `lambda = exposure * exp(log_mu)` -- plus the beta-binomial, summed. Both
    channels in one array because both are read together and neither is
    wanted alone.

    A zero exposure contributes zero rather than `-inf`: that is `cnaster`'s
    own convention (`_nb_logpmf_1d` skips a bin with no baseline), and it is
    what keeps an empty bin from taking the whole likelihood with it.
    """
    rates = _column(log_mu)[:, None]
    dispersions = _column(alphas)[:, None]
    probabilities = _column(p_binom)[:, None]
    concentrations = _column(taus)[:, None]

    observed = jnp.asarray(counts_nb)[None, :]
    exposure = jnp.asarray(base_nb_mean)[None, :]

    mean = exposure * jnp.exp(rates)
    size = 1.0 / jnp.maximum(dispersions, 1e-10)
    success = 1.0 / (1.0 + dispersions * mean)

    read_depth = jnp.where(
        mean <= 0.0,
        0.0,
        jsp.gammaln(observed + size)
        - jsp.gammaln(size)
        - jsp.gammaln(observed + 1.0)
        + size * jnp.log(success)
        + observed * jnp.log1p(-success),
    )

    successes = jnp.asarray(counts_bb)[None, :]
    trials = jnp.asarray(total_bb_RD)[None, :]

    alpha = jnp.maximum(probabilities * concentrations, 1e-10)
    beta = jnp.maximum((1.0 - probabilities) * concentrations, 1e-10)

    allele = (
        jsp.gammaln(trials + 1.0)
        - jsp.gammaln(successes + 1.0)
        - jsp.gammaln(trials - successes + 1.0)
        + jsp.gammaln(successes + alpha)
        + jsp.gammaln(trials - successes + beta)
        - jsp.gammaln(trials + alpha + beta)
        - (jsp.gammaln(alpha) + jsp.gammaln(beta) - jsp.gammaln(alpha + beta))
    )

    return read_depth + allele


def marginal_negative_log_likelihood(
    log_emission: Any, log_startprob: Any, log_transmat: Any, lengths: Any
) -> jnp.ndarray:
    """The forward recursion's `-log P(x)`, summed over sequences.

    `lengths` gives the segments per sequence, so a clone-stacked run is one
    call: `is_start` restarts the recursion at each boundary and the
    normalizer is read at each end. This is `jax.lax.scan` rather than a
    Python loop so the whole recursion is one traced computation and the
    Hessian does not unroll `n_obs` frames.
    """
    emissions = jnp.asarray(log_emission)
    starts = np.zeros(emissions.shape[1], dtype=bool)
    ends = np.zeros(emissions.shape[1], dtype=bool)

    offset = 0

    for length in np.asarray(lengths).reshape(-1):
        starts[offset] = True
        offset += int(length)
        ends[offset - 1] = True

    start_log_prob = jnp.asarray(log_startprob).reshape(-1)
    transitions = jnp.asarray(log_transmat)

    def step(
        previous: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        is_start, scores = inputs
        carried = jsp.logsumexp(previous[:, None] + transitions, axis=0)

        updated = jnp.where(is_start, start_log_prob + scores, carried + scores)

        return updated, updated

    _, alphas = jax.lax.scan(
        step,
        jnp.zeros_like(start_log_prob),
        (jnp.asarray(starts), emissions.T),
    )

    normalizers = jsp.logsumexp(alphas.T, axis=0)

    return -jnp.sum(jnp.where(jnp.asarray(ends), normalizers, 0.0))


def shifted_rates(
    log_mu: Any, copy_states: Any, normal_log_lambda: Any, clone_lengths: Any
) -> jnp.ndarray:
    """`log_mu` debiased by the per-clone normalizer, differentiably.

    The same quantity `port.patch.hmm_nophasing.logmu_shift.shifts` computes,
    written so `jax` can differentiate through it -- which is the point, and
    what a `numba` kernel cannot give. Returns `(n_clones, n_states)`: one
    debiased vector per clone, because the shift differs between them.

    **The shift depends on `log_mu`**, so differentiating through it is not
    the same as subtracting a constant. `port.extensions.parameter_errors`
    carries the consequence: the debiased rates are invariant under a
    constant added to every `log_mu`, so their covariance is singular in that
    direction and reporting the raw variance there would put back the scale
    the debiasing removes.
    """
    rates = _column(log_mu)
    states = jnp.asarray(np.asarray(copy_states, dtype=np.int64).reshape(-1))
    exposures = jnp.asarray(np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1))

    terms = rates[states] + exposures

    lengths = np.asarray(clone_lengths, dtype=np.int64).reshape(-1)
    bounds = np.concatenate(([0], np.cumsum(lengths)))

    shifts = jnp.stack(
        [
            jsp.logsumexp(terms[int(bounds[clone]) : int(bounds[clone + 1])])
            for clone in range(lengths.size)
        ]
    )

    return rates[None, :] - shifts[:, None]
