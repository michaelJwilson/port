r"""The M step's gradient, derived rather than differenced (#259 stage 5).

**`optimize_params` calls BFGS with no `jac`, so `scipy` differences the
objective, and every evaluation is a full emission.** Traced on the
`test_core_inference_end_to_end` fixture: **516 of the fit's 520 emission
evaluations are `nfev`**, 21.6 to 45.0 per BFGS step, for 19 steps. The
parameter count is `~5 n_states`, so it worsens as the model grows.

## The objective

Inside an M step :math:`\gamma` is fixed, and the cost is

.. math::
    L = -\sum_{g,k} \gamma_{g,k} \left[
        \mathrm{nb}(x_g ; \lambda_g \bar\mu_k, \alpha_k)
      + \mathrm{bb}(y_g ; n_g, p_k, \tau_k) \right]

with :math:`\bar\mu_k = \exp(\theta_k - \log Z_{c(g)})` when the shift is on
and :math:`\exp(\theta_k)` when it is off.

## The blocks

`pack_params` lays out `[log_startprob, log_mu, logit p, log alphas,
log taus]`, and each is differentiated in the variable it is packed in.

**Start probabilities.** :math:`\partial L / \partial s = 0`. The EM cost is
a posterior-weighted emission and carries no start term; upstream packs it
anyway, and the optimizer has been differencing a block that cannot move the
objective. That is `n_states` wasted evaluations per gradient, and the
analytic form says so in one line.

**RDR mean.** With :math:`r = 1/\alpha` and :math:`p = 1/(1 + \alpha\lambda)`
as `_nb_logpmf_1d` writes them, the whole expression collapses:

.. math::
    \frac{\partial}{\partial \theta} \log \mathrm{nb}
      = \frac{x - \lambda}{1 + \alpha \lambda}

**NB dispersion**, in :math:`a = \log\alpha`, needs the digammas:

.. math::
    \frac{\partial}{\partial \alpha} \log \mathrm{nb}
      = \frac{\psi(1/\alpha) - \psi(x + 1/\alpha) + \log(1+\alpha\lambda)}{\alpha^2}
      - \frac{\lambda}{\alpha(1+\alpha\lambda)} + \frac{x}{\alpha}
      - \frac{x\lambda}{1+\alpha\lambda}

**BAF.** With :math:`A = p\tau` and :math:`B = (1-p)\tau`,
:math:`\partial/\partial A = \psi(y+A) - \psi(n+A+B) - \psi(A) + \psi(A+B)`
and :math:`\partial/\partial B` its mirror in :math:`n-y`. The packed
variables are :math:`u = \mathrm{logit}\,p` and :math:`t = \log\tau`, so

.. math::
    \frac{\partial}{\partial u} = \tau\,p(1-p)
        \left(\frac{\partial}{\partial A} - \frac{\partial}{\partial B}\right),
    \qquad
    \frac{\partial}{\partial t} = \tau
        \left(p\frac{\partial}{\partial A}
            + (1-p)\frac{\partial}{\partial B}\right)

## The constraint, in the gradient

When the shift is on the rates stop being separable: :math:`\log Z_c` is in
every state's term, so

.. math::
    \frac{\partial L}{\partial \theta_j} = -\sum_c \sum_{g \in c} \sum_k
        \gamma_{g,k} D_{g,k}
        \left( \delta_{kj} - \frac{\partial \log Z_c}{\partial \theta_j} \right),
    \qquad
    \frac{\partial \log Z_c}{\partial \theta_j}
      = \frac{\exp(\theta_j) W_{c,j}}{Z_c}

with :math:`W` from `port.patch.clone_shift.clone_state_weights` and
:math:`D` the RDR derivative above at the shifted rate. **Substituted, not
penalized**: the constraint is inside the objective the optimizer sees, so
the observed information stays non-singular in the removed direction.

## Compression

:math:`\gamma` is per segment and :math:`D` depends on the segment only
through its ``(obs, total)`` pair, so
:math:`\sum_g \gamma_{g,k} D_{g,k} = \sum_u D_{u,k} (\gamma M)_{k,u}` --
which is `CountEncoder.encode_array`. The gradient therefore costs what the
emission costs, over the same unique pairs, rather than over the genome.

`tests/test_em_gradient.py` refereees every block against
`scipy.optimize.check_grad`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special

__all__ = ["EmGradient", "nb_dispersion_derivative", "nb_rate_derivative"]


def nb_rate_derivative(
    obs: np.ndarray, exposure: np.ndarray, mu: float, alpha: float
) -> np.ndarray:
    r""":math:`\partial \log \mathrm{nb} / \partial \theta`, per observation.

    `(x - lambda) / (1 + alpha lambda)`, which is what
    `r log p + x log(1-p)` differentiates to once `r = 1/alpha` and
    `p = 1/(1 + alpha lambda)` are substituted. Zero where the exposure is,
    matching `_nb_logpmf_1d`'s own guard -- an observation with no exposure
    contributes no term, so it contributes no derivative.
    """
    lam = exposure * mu
    out = np.zeros_like(lam, dtype=np.float64)
    live = lam > 0.0

    out[live] = (obs[live] - lam[live]) / (1.0 + alpha * lam[live])

    return out


def nb_dispersion_derivative(
    obs: np.ndarray, exposure: np.ndarray, mu: float, alpha: float
) -> np.ndarray:
    r""":math:`\partial \log \mathrm{nb} / \partial \alpha`, per observation."""
    lam = exposure * mu
    out = np.zeros_like(lam, dtype=np.float64)
    live = lam > 0.0

    x = obs[live]
    lm = lam[live]
    inv = 1.0 / alpha
    denom = 1.0 + alpha * lm

    out[live] = (
        (scipy.special.digamma(inv) - scipy.special.digamma(x + inv) + np.log(denom))
        / (alpha * alpha)
        - lm / (alpha * denom)
        + x / alpha
        - x * lm / denom
    )

    return out


def _bb_derivatives(
    obs: np.ndarray, total: np.ndarray, p_binom: float, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`\partial/\partial A` and :math:`\partial/\partial B`, per observation.

    Zero where `betabinom_logpmf_numba` returns a constant 0.0 -- outside
    `0 <= y <= n` it contributes no term and so no derivative.
    """
    eps = 1e-10
    a = max(p_binom * tau, eps)
    b = max((1.0 - p_binom) * tau, eps)

    d_a = np.zeros_like(total, dtype=np.float64)
    d_b = np.zeros_like(total, dtype=np.float64)

    live = (total >= 0) & (obs >= 0) & (obs <= total)

    y = obs[live]
    n = total[live]

    shared = scipy.special.digamma(a + b) - scipy.special.digamma(n + a + b)

    d_a[live] = scipy.special.digamma(y + a) - scipy.special.digamma(a) + shared
    d_b[live] = scipy.special.digamma(n - y + b) - scipy.special.digamma(b) + shared

    return d_a, d_b


class EmGradient:
    """The EM objective's gradient, in the variables `pack_params` packs.

    A mixin on the patched `hmm_nophasing`, used by
    `port.patch.optimization_pipeline` when `jac` is asked for. It is written
    against the same encoders and the same `self.params` layout the objective
    uses, because a gradient of a different objective is worse than none.
    """

    params: str
    apply_logmu_shift: bool

    def _em_gradient(
        self,
        *,
        nbEncoder: Any,
        bbEncoder: Any,
        posteriors: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
        optimize_nb: bool,
        fix_NB_dispersion: bool,
        shared_NB_dispersion: bool,
        fix_BB_dispersion: bool,
        shared_BB_dispersion: bool,
        use_logit: bool,
        shifts: np.ndarray | None = None,
        clone_weights: np.ndarray | None = None,
        lengths: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """`-dL/dx` for the packed vector `x`, block by block."""
        n_states = log_mu.shape[0]

        nb_obs = nbEncoder.get_unique_obs(0)
        nb_exposure = nbEncoder.get_unique_total(0)
        bb_obs = bbEncoder.get_unique_obs(0)
        bb_total = bbEncoder.get_unique_total(0)

        # NB gamma summed onto the unique pairs, which is what makes the
        #    gradient cost what the emission costs.
        gamma_nb = nbEncoder.encode_array(posteriors, 0)
        gamma_bb = bbEncoder.encode_array(posteriors, 0)

        rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)

        d_theta = np.zeros(n_states)
        d_log_alpha = np.zeros(n_states)
        d_logit_p = np.zeros(n_states)
        d_log_tau = np.zeros(n_states)

        shifted = (
            rates - shifts[0] if shifts is not None and shifts.size == 1 else rates
        )

        for state in range(n_states):
            mu = float(np.exp(shifted[state]))
            alpha = float(alphas[state, 0])

            weights = gamma_nb[state]

            d_theta[state] = float(
                np.sum(weights * nb_rate_derivative(nb_obs, nb_exposure, mu, alpha))
            )
            d_log_alpha[state] = alpha * float(
                np.sum(
                    weights * nb_dispersion_derivative(nb_obs, nb_exposure, mu, alpha)
                )
            )

            p_value = float(p_binom[state, 0])
            tau = float(taus[state, 0])

            d_a, d_b = _bb_derivatives(bb_obs, bb_total, p_value, tau)
            baf_weights = gamma_bb[state]

            to_a = float(np.sum(baf_weights * d_a))
            to_b = float(np.sum(baf_weights * d_b))

            # NB `unpack_params` packs `logit p` when `use_logit` and `p`
            #    itself otherwise, so the chain rule picks up `p(1-p)` in the
            #    first case and nothing in the second. Getting this wrong
            #    would be a gradient of a different parameterization.
            chain = p_value * (1.0 - p_value) if use_logit else 1.0

            d_logit_p[state] = tau * chain * (to_a - to_b)
            d_log_tau[state] = tau * (p_value * to_a + (1.0 - p_value) * to_b)

        if shifts is not None and clone_weights is not None and lengths is not None:
            d_theta = self._couple(
                d_theta=d_theta,
                rates=rates,
                clone_weights=clone_weights,
                shifts=shifts,
            )

        blocks: list[np.ndarray] = []

        if "s" in self.params:
            # NB the EM cost is a posterior-weighted emission; no start term
            #    appears in it, so this block cannot move the objective.
            blocks.append(np.zeros(n_states))

        if optimize_nb and "m" in self.params:
            blocks.append(d_theta)

        if "p" in self.params:
            blocks.append(d_logit_p)

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            blocks.append(
                np.array([np.sum(d_log_alpha)]) if shared_NB_dispersion else d_log_alpha
            )

        if "p" in self.params and not fix_BB_dispersion:
            blocks.append(
                np.array([np.sum(d_log_tau)]) if shared_BB_dispersion else d_log_tau
            )

        # NB the objective is the negative of the weighted log-likelihood, so
        #    every block is negated once, here, rather than in each derivation.
        return -np.concatenate(blocks) if blocks else np.array([])

    @staticmethod
    def _couple(
        *,
        d_theta: np.ndarray,
        rates: np.ndarray,
        clone_weights: np.ndarray,
        shifts: np.ndarray,
    ) -> np.ndarray:
        r"""Fold :math:`\partial \log Z / \partial \theta_j` into the rate block.

        With :math:`\bar\theta_k = \theta_k - \log Z` and
        :math:`S_k = \sum_g \gamma_{g,k} f'(\bar\theta_k)`,

        .. math::
            \frac{\partial L}{\partial \theta_j}
              = S_j - \left(\sum_k S_k\right)
                \frac{\partial \log Z}{\partial \theta_j},
            \qquad
            \frac{\partial \log Z}{\partial \theta_j}
              = e^{\theta_j + \log W_j - \log Z}

        The :math:`\theta_j` in that exponent is the **unshifted** rate, and
        leaving it out is the mistake this docstring exists to stop: the
        weights alone are :math:`W_j / Z`, which is not a derivative of
        anything.

        One clone, which is what the caller's `shifts.size == 1` guard
        already established.
        """
        total = float(np.sum(d_theta))
        dlogz: np.ndarray = np.exp(rates + clone_weights[0] - shifts[0])

        return d_theta - total * dlogz
