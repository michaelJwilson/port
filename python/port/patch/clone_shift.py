r"""The per-clone library normalizer, weighted by the state posterior.

**#259 stage 4.** `cnaster`'s `compute_logmu_shifts` computes

.. math::
    \log Z_c = \log \sum_{g \in c} \lambda_g \exp(\theta_{k(g)})

with :math:`k(g)` a **hard** decode, and discards the result --
`# TODO fold in logmu_shifts`. The form #259 settled on is the same quantity
with the decode replaced by the posterior :math:`\gamma` the M step already
holds fixed:

.. math::
    \log Z_c = \log \sum_{g \in c} \lambda_g \sum_k \gamma_{g,k} \exp(\theta_k)

Both are one scalar per clone, and the second **is** the first when
:math:`\gamma` is one-hot -- which is what makes this a rewrite refereed
against `cnaster` rather than a formula this repository invented.
`tests/test_clone_shift.py` pins that reduction.

## Why the posterior and not the decode

Inside an M step :math:`\gamma` is fixed by construction: that is what makes
the step a maximization of a bound rather than of the likelihood. So the
`\gamma`-weighted normalizer is the one whose derivative

.. math::
    \frac{\partial \log Z_c}{\partial \theta_j}
      = \frac{\exp(\theta_j) \sum_{g \in c} \lambda_g \gamma_{g,j}}{Z_c}

is the gradient stage 5 needs. A hard decode is not differentiable in
:math:`\theta` at all, so folding it in would leave the M step optimizing an
objective whose constraint its gradient does not know about.

## Where the clone boundaries come from

`num_segments_clones`, which `hmrf.py:564` builds as
``X.shape[0] * np.ones(X.shape[2])`` -- so on the live path every clone has
the same length and the blocks are rectangular. Nothing here assumes that:
the lengths are walked, because a vector that happens to be constant today is
not a rectangle tomorrow.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.special

__all__ = ["clone_state_weights", "log_normalizers", "log_normalizers_from_weights"]


def clone_state_weights(
    state_posteriors: np.ndarray,
    normal_log_lambda: np.ndarray,
    num_segments_clones: np.ndarray | Sequence[int],
) -> np.ndarray:
    r"""The part of :math:`Z_c` that does not depend on :math:`\theta`.

    .. math::
        \log W_{c,k} = \log \sum_{g \in c} \lambda_g \gamma_{g,k}

    so that :math:`Z_c = \sum_k W_{c,k} \exp(\theta_k)`. **This is the whole
    reason the shift is affordable.** :math:`\gamma` is fixed inside an M step
    and :math:`\lambda` is fixed for the fit, so the `(n_states, n_segments)`
    work happens once per M step rather than once per emission -- and the
    per-call cost falls to an `(n_clones, n_states)` reduction.

    It is also the quantity the gradient needs:

    .. math::
        \frac{\partial \log Z_c}{\partial \theta_j}
          = \frac{\exp(\theta_j) W_{c,j}}{Z_c}

    Returns
    -------
    ``(n_clones, n_states)``, in log space. A state no segment of a clone
    gives weight to is :math:`-\infty` there, which contributes nothing.
    """
    gamma = np.asarray(state_posteriors, dtype=np.float64)
    lambdas = np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)
    lengths = np.asarray(num_segments_clones, dtype=np.int64).reshape(-1)

    if gamma.ndim != 2 or gamma.shape[1] != lambdas.size:
        msg = (
            f"state_posteriors is {gamma.shape}, expected "
            f"(n_states, {lambdas.size}) -- (n_states, n_segments)"
        )
        raise ValueError(msg)

    if int(lengths.sum()) != lambdas.size:
        msg = f"clone lengths sum to {int(lengths.sum())}, not {lambdas.size}"
        raise ValueError(msg)

    # NB an exact zero kept as -inf rather than warned about: a state the
    #    posterior rules out contributes nothing, and `logsumexp` is what
    #    makes that the right answer instead of a nan.
    with np.errstate(divide="ignore"):
        terms = lambdas[None, :] + np.log(gamma)

    weights = np.empty((lengths.size, gamma.shape[0]), dtype=np.float64)
    start = 0

    for clone, length in enumerate(lengths):
        stop = start + int(length)
        block = terms[:, start:stop]

        weights[clone] = np.where(
            np.any(np.isfinite(block), axis=1),
            scipy.special.logsumexp(
                np.where(np.isfinite(block), block, -np.inf), axis=1
            ),
            -np.inf,
        )
        start = stop

    return weights


def log_normalizers_from_weights(
    log_weights: np.ndarray, log_mu: np.ndarray
) -> np.ndarray:
    r""":math:`\log Z_c = \log \sum_k W_{c,k} \exp(\theta_k)`, one per clone.

    The per-call half of the split above: an ``(n_clones, n_states)``
    reduction, which is what makes the shift cost `O(n_clones * n_states)`
    per emission instead of `O(n_states * n_segments)`.
    """
    rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    weights = np.asarray(log_weights, dtype=np.float64)

    if weights.shape[1] != rates.size:
        msg = f"weights are {weights.shape}, expected (n_clones, {rates.size})"
        raise ValueError(msg)

    terms = weights + rates[None, :]
    finite = np.any(np.isfinite(terms), axis=1)

    return np.where(
        finite,
        scipy.special.logsumexp(np.where(np.isfinite(terms), terms, -np.inf), axis=1),
        -np.inf,
    )


def log_normalizers(
    log_mu: np.ndarray,
    state_posteriors: np.ndarray,
    normal_log_lambda: np.ndarray,
    num_segments_clones: np.ndarray | Sequence[int],
) -> np.ndarray:
    r"""One :math:`\log Z_c` per clone, from the posterior rather than a decode.

    **This is the reference, and it is a single pass.** It sums every
    ``(segment, state)`` term of a clone in one `logsumexp`, which is the
    order `cnaster`'s loop uses -- so it agrees with `compute_logmu_shifts`
    **bitwise** on a one-hot posterior.

    `clone_state_weights` and `log_normalizers_from_weights` above are the
    fast path the emission takes. They reduce over segments and then over
    states, which is the same sum in a different order and therefore not
    bitwise: measured at 8.88e-16. Two claims, two referees --
    `tests/test_clone_shift.py` pins this one against `cnaster` and that one
    against this.

    Parameters
    ----------
    log_mu
        :math:`\theta`, the log rates, ``(n_states,)`` or ``(n_states, 1)``.
    state_posteriors
        :math:`\gamma`, ``(n_states, n_segments)``, **not** in log space --
        that is what `optimize_params` stores on the instance.
    normal_log_lambda
        :math:`\log \lambda_g`, ``(n_segments,)``.
    num_segments_clones
        One length per clone, summing to ``n_segments``.

    Returns
    -------
    ``(n_clones,)``. A clone whose terms are all :math:`-\infty` returns
    :math:`-\infty` rather than a ``nan``, which is the branch `cnaster`'s
    loop takes and the one a naive rewrite gets wrong.
    """
    rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    gamma = np.asarray(state_posteriors, dtype=np.float64)

    if gamma.shape[:1] != (rates.size,):
        msg = (
            f"state_posteriors is {gamma.shape}, expected "
            f"({rates.size}, n_segments) -- (n_states, n_segments)"
        )
        raise ValueError(msg)

    lambdas = np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)
    lengths = np.asarray(num_segments_clones, dtype=np.int64).reshape(-1)

    if gamma.shape != (rates.size, lambdas.size):
        msg = (
            f"state_posteriors is {gamma.shape}, expected "
            f"{(rates.size, lambdas.size)} -- (n_states, n_segments)"
        )
        raise ValueError(msg)

    if int(lengths.sum()) != lambdas.size:
        msg = f"clone lengths sum to {int(lengths.sum())}, not {lambdas.size}"
        raise ValueError(msg)

    with np.errstate(divide="ignore"):
        terms = lambdas[None, :] + np.log(gamma) + rates[:, None]

    out = np.empty(lengths.size, dtype=np.float64)
    start = 0

    for clone, length in enumerate(lengths):
        stop = start + int(length)
        block = terms[:, start:stop]

        out[clone] = (
            -np.inf
            if not np.any(np.isfinite(block))
            else scipy.special.logsumexp(block)
        )
        start = stop

    return out
