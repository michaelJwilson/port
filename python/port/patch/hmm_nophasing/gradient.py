r"""The M step's gradient in closed form, where `cnaster` differences it (#433).

`hmm_nophasing._run_optimization_pipeline` fits the emission by `scipy`'s
BFGS with no `jac`, so every gradient is `approx_derivative`: one objective
call per coordinate. At `K = 8` with shared dispersions that is 26
coordinates, 8 of them start probabilities the EM objective never reads, and
a three-iteration fit on the dev instance makes 90 to 162 objective calls.

The objective is closed form in the packed coordinates. With `gamma` the
posteriors the fit is holding and `u` a unique `(obs, total)` code,

.. math::
    f(x) = -\sum_{i,u} W_{iu} \left[\ell^{NB}_{iu} + \ell^{BB}_{iu}\right],
    \qquad W_{iu} = \sum_{g \to u} \gamma_{ig},

and each term's derivative is a digamma or a ratio:

- negative binomial, `r = 1 / max(alpha, 1e-10)`, `mu = c exp(eta)`,
  `p = 1 / (1 + alpha mu)`: `d ell / d eta = k - (r + k) alpha mu / (1 + alpha mu)`,
  and, where `alpha` is above the floor, `d ell / d log alpha =
  -r (psi(k + r) - psi(r) + log p) + k - (r + k) alpha mu / (1 + alpha mu)`;
- beta-binomial, `a = max(p tau, 1e-10)`, `b = max((1 - p) tau, 1e-10)`:
  `d ell / d a = psi(k + a) - psi(n + a + b) - psi(a) + psi(a + b)`, and
  `b`'s with `n - k` for `k`.

**Under the shift the rate is `exp(log_mu_i - S_c)`**, with
`S_c = logsumexp_g(log_mu_{d(g)} + log lambda_g)` over clone `c`'s segments
at the decode `d` (`port.patch.hmm_nophasing.logmu_shift`). `S_c` depends on
`log_mu`, so `d eta / d log_mu_j = delta_ij - P_cj`, with `P_cj` the softmax
weight clone `c`'s segments decoded to `j` carry. That term is what makes
the shifted likelihood flat along `log_mu -> log_mu + c`, and the gradient
here has zero sum over states there, as it must.

**Referee:** `port.extensions.jax_hmm`, the same objective differentiated by
`jax` (`tests/test_mstep_gradient.py`). Nothing here is a second route to a
fit: `cnaster`'s own `cost_fn` still computes every value BFGS reads, and
its own callback still updates the posteriors.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize
from cnaster.count_encoder import CountEncoder
from scipy.special import digamma, expit

__all__ = [
    "ALPHA_MIN",
    "DISPERSION_FLOOR",
    "DISPERSION_PRIOR_ROWS",
    "TAU_MAX",
    "DispersionBounds",
    "DispersionShrinkage",
    "EmGradient",
    "analytic_bfgs",
    "bb_partials",
    "configured_method",
    "configured_solver",
    "dispersion_blocks",
    "nb_partials",
]

DISPERSION_FLOOR = 1e-10
"""`cnaster`'s floor on `alpha` in `_nb_logpmf_1d` and on `a`, `b` in `_bb_logpmf_1d`.

The one statement of it: every port kernel that scores those two laws reads
this, rather than restating the literal (#517).
"""


def nb_partials(
    obs: np.ndarray, mean: np.ndarray, dispersion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d log mean` and `d ell / d log alpha` of `cnaster`'s negative binomial.

    Broadcasts. A bin `cnaster` scores 0 -- no exposure, or `p` rounded to 1
    -- has zero derivative, because its score does not move.
    """
    alpha = np.asarray(dispersion, dtype=np.float64)
    size = 1.0 / np.maximum(alpha, DISPERSION_FLOOR)
    scaled = alpha * mean
    success = 1.0 / (1.0 + scaled)
    live = (mean > 0.0) & (success < 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (size + obs) * scaled / (1.0 + scaled)
        d_eta = np.where(live, obs - pull, 0.0)

        # NB below the floor `r` is a constant and only `p` moves with alpha.
        through_size = np.where(
            alpha > DISPERSION_FLOOR,
            -size * (digamma(obs + size) - digamma(size) + np.log(success)),
            0.0,
        )
        d_alpha = np.where(live, through_size + obs - pull, 0.0)

    return d_eta, d_alpha


def bb_partials(
    obs: np.ndarray, total: np.ndarray, p_binom: np.ndarray, taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d p` and `d ell / d log tau` of `cnaster`'s beta-binomial.

    A floored `a` or `b` is a constant, so contributes nothing; a code
    `cnaster` scores 0 (`k > n`) has zero derivative.
    """
    tau = taus
    shape_a = p_binom * tau
    shape_b = (1.0 - p_binom) * tau
    a = np.maximum(shape_a, DISPERSION_FLOOR)
    b = np.maximum(shape_b, DISPERSION_FLOOR)

    joint = digamma(total + a + b) - digamma(a + b)
    d_a = digamma(obs + a) - digamma(a) - joint
    d_b = digamma(total - obs + b) - digamma(b) - joint

    live_a = shape_a > DISPERSION_FLOOR
    live_b = shape_b > DISPERSION_FLOOR
    valid = (obs >= 0) & (total >= 0) & (obs <= total)

    d_a = np.where(valid & live_a, d_a, 0.0)
    d_b = np.where(valid & live_b, d_b, 0.0)

    return d_a * tau - d_b * tau, d_a * shape_a + d_b * shape_b


def _sums(weight: np.ndarray, partial: np.ndarray, spread: bool) -> np.ndarray:
    """`-sum_c w_kc d_kc` per state, or, with `spread`, its weighted variance about each state's mean.

    The second is the centred outer product of the per-row scores, `sum_c
    w_kc d_kc^2 - (sum_c w_kc d_kc)^2 / n_k`: an information estimate that is
    non-negative at any point, which the curvature is not away from the
    optimum (#566).
    """
    linear = np.sum(weight * partial, axis=1)

    if not spread:
        return -linear

    occupancy = np.sum(weight, axis=1)
    square = np.sum(weight * partial * partial, axis=1)
    safe = np.where(occupancy > 0.0, occupancy, 1.0)
    return np.where(occupancy > 0.0, square - linear * linear / safe, 0.0)


@dataclass
class EmGradient:
    """The EM objective's gradient for one fit, in the fit's packed coordinates.

    Built once per `_run_optimization_pipeline` from what it is called with,
    and read at each `x` against the posteriors the model is holding then,
    which are the ones `cnaster`'s `cost_fn` weighs by.
    """

    model: Any
    n_states: int
    initial: tuple[Any, ...]
    flags: dict[str, Any]
    nb: CountEncoder | None
    bb: CountEncoder
    normal_log_lambda: np.ndarray | None
    clone_lengths: Any
    rescale: Any = None
    """`port.patch.hmm_nophasing.rescale.Rescale`, or `None`: per-row dispersions (#566)."""
    rows: tuple[np.ndarray, ...] = ()
    """Per-row NB counts and exposure, BB counts and trials: the rescaled path's codes."""

    @classmethod
    def for_fit(
        cls,
        model: Any,
        X: np.ndarray,
        n_states: int,
        base_nb_mean: np.ndarray,
        total_bb_RD: np.ndarray,
        **kwargs: Any,
    ) -> EmGradient:
        """Read the fit's settings as `_run_optimization_pipeline` would.

        Defaults come from that function's own signature rather than a copy
        of them here, so a changed default upstream reaches both.
        """
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(
                type(model)._run_optimization_pipeline
            ).parameters.items()
        }
        setting = defaults | kwargs

        base = np.array(base_nb_mean, dtype=np.float64)
        max_rdr = setting["max_rdr"]

        # NB `hmm_nophasing.py:822-826`, restated: bins above `max_rdr`
        #    lose their baseline, so score 0 and carry no gradient.
        if max_rdr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = X[:, 0, :] / base
                ratio[np.isnan(ratio)] = 0.0
                base[ratio > max_rdr] = 0.0

        optimize_nb = bool(np.any(base > 0))
        normal_lambda = setting["normal_lambda"]

        (log_mu, p_binom, alphas, taus, log_startprob, _) = model.get_initial_params(
            n_states,
            X.shape[2],
            setting["init_log_mu"],
            setting["init_p_binom"],
            setting["init_alphas"],
            setting["init_taus"],
        )

        flags = {
            "optimize_nb": optimize_nb,
            "fix_NB_dispersion": setting["fix_NB_dispersion"],
            "shared_NB_dispersion": setting["shared_NB_dispersion"],
            "fix_BB_dispersion": setting["fix_BB_dispersion"],
            "shared_BB_dispersion": setting["shared_BB_dispersion"],
            "use_logit": setting["use_logit"],
        }

        return cls(
            model=model,
            n_states=n_states,
            initial=(log_startprob, log_mu, p_binom, alphas, taus),
            flags=flags,
            nb=CountEncoder(X[:, 0, :], base) if optimize_nb else None,
            bb=CountEncoder(X[:, 1, :], total_bb_RD),
            normal_log_lambda=(
                None if normal_lambda is None else np.log(normal_lambda)
            ),
            clone_lengths=setting["clone_lengths"],
            rescale=getattr(model, "_rescale", None),
            rows=(
                np.asarray(X[:, 0, 0], dtype=np.float64),
                base[:, 0],
                np.asarray(X[:, 1, 0], dtype=np.float64),
                np.asarray(total_bb_RD, dtype=np.float64)[:, 0],
            ),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """`d f / d x` at `x`, `f` being `cnaster`'s `cost_fn`."""
        model = self.model
        _, log_mu, p_binom, alphas, taus = model.unpack_params(
            x, self.n_states, *self.initial, **self.flags
        )
        gamma = np.asarray(model.state_posteriors)

        if self.rescale is not None:
            g_p, g_tau = self._allele_rows(gamma, p_binom, taus)
        else:
            g_p, g_tau = self._allele(gamma, p_binom, taus)

        # NB zeros where the depth channel is not fitted; `_pack` then
        #    leaves its blocks out, as `pack_params` does.
        g_mu = g_alpha = np.zeros(self.n_states)

        if self.nb is not None and "m" in model.params:
            if self.rescale is not None:
                g_mu, g_alpha = self._depth_rows(gamma, log_mu, alphas)
            else:
                g_mu, g_alpha = self._depth(self.nb, gamma, log_mu, alphas)

        return self._pack(x, g_mu, g_p, g_alpha, g_tau)

    def information(self, x: np.ndarray) -> dict[str, float]:
        """Per-row information in `log alpha` and `log tau` at `x`, pooled over states (#566).

        The centred score spread (:func:`_sums`) summed over states, over the
        rows `N = sum gamma`: what one row carries about the shared
        dispersion, for :class:`DispersionShrinkage`'s `h`.
        """
        model = self.model
        _, log_mu, p_binom, alphas, taus = model.unpack_params(
            x, self.n_states, *self.initial, **self.flags
        )
        gamma = np.asarray(model.state_posteriors)
        rows = max(float(gamma.sum()), 1.0)

        if self.rescale is not None:
            _, tau_spread = self._allele_rows(gamma, p_binom, taus, spread=True)
        else:
            _, tau_spread = self._allele(gamma, p_binom, taus, spread=True)
        out = {"bb": float(tau_spread.sum()) / rows}

        if self.nb is not None and "m" in model.params:
            if self.rescale is not None:
                _, alpha_spread = self._depth_rows(gamma, log_mu, alphas, spread=True)
            else:
                _, alpha_spread = self._depth(
                    self.nb, gamma, log_mu, alphas, spread=True
                )
            out["nb"] = float(alpha_spread.sum()) / rows

        return out

    def _allele(
        self,
        gamma: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
        spread: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-state `d f / d p` and `d f / d log tau`."""
        obs = self.bb.get_unique_obs(0)
        total = self.bb.get_unique_total(0)
        weight = np.asarray(self.bb.encode_array(gamma, 0))

        d_p, d_tau = bb_partials(
            obs[None, :], total[None, :], p_binom[:, :1], taus[:, :1]
        )

        return -np.sum(weight * d_p, axis=1), _sums(weight, d_tau, spread)

    def _depth(
        self,
        encoder: CountEncoder,
        gamma: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        spread: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-state `d f / d log mu` and `d f / d log alpha`, shifted or not."""
        rates = np.asarray(log_mu, dtype=np.float64)[:, 0]
        dispersions = np.asarray(alphas, dtype=np.float64)[:, :1]

        shifted = self._shift_inputs()

        if shifted is None:
            obs = encoder.get_unique_obs(0)
            exposure = encoder.get_unique_total(0)
            weight = np.asarray(encoder.encode_array(gamma, 0))
            mean = exposure[None, :] * np.exp(rates)[:, None]

            d_eta, d_alpha = nb_partials(obs[None, :], mean, dispersions)

            return -np.sum(weight * d_eta, axis=1), _sums(weight, d_alpha, spread)

        return self._shifted_depth(
            encoder, gamma, rates, dispersions, *shifted, spread=spread
        )

    def _allele_rows(
        self,
        gamma: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
        spread: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`_allele` per row, at `tau_row = (1 + tau) / g_row - 1` (#566).

        `d log tau_row / d log tau = tau / (g_row tau_row)`.
        """
        from port.patch.hmm_nophasing.rescale import tau_rows

        _, _, obs, total = self.rows
        tau = np.asarray(taus, dtype=np.float64)[:, :1]
        per_row = tau_rows(tau, self.rescale)
        # NB `g = 0`, no spot with two trials: the binomial, which `tau`
        #    does not move.
        binomial = np.isinf(per_row)
        finite = np.where(binomial, 1.0, per_row)

        d_p, d_tau = bb_partials(obs[None, :], total[None, :], p_binom[:, :1], finite)
        share = np.clip(p_binom[:, :1], DISPERSION_FLOOR, 1.0 - DISPERSION_FLOOR)
        valid = (obs >= 0) & (total >= 0) & (obs <= total)
        limit = np.where(valid, obs / share - (total - obs) / (1.0 - share), 0.0)
        d_p = np.where(binomial, limit, d_p)
        chain = np.where(
            binomial,
            0.0,
            tau / (np.where(binomial, 1.0, self.rescale.bb[None, :]) * finite),
        )

        return -np.sum(gamma * d_p, axis=1), _sums(gamma, d_tau * chain, spread)

    def _depth_rows(
        self,
        gamma: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        spread: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`_depth` per row, at `alpha_row = alpha f_c` (#566); shifted where the fit is.

        `log alpha_row = log alpha + log f_c`, so `d / d log alpha` is the
        row's own.
        """
        from port.patch.hmm_nophasing.rescale import alpha_rows

        obs, exposure, _, _ = self.rows
        rates = np.asarray(log_mu, dtype=np.float64)[:, 0]
        n_states = rates.size
        shifted = self._shift_inputs()

        if shifted is None:
            lengths = self.rescale.lengths
            shifts = np.zeros(len(lengths))
            share = np.zeros((len(lengths), n_states))
        else:
            decode, lengths = shifted
            shifts, share = self._clone_softmax(rates, decode, lengths)

        clone_of = np.repeat(np.arange(len(lengths)), lengths)
        mean = exposure[None, :] * np.exp(rates[:, None] - shifts[clone_of][None, :])
        dispersions = alpha_rows(np.asarray(alphas)[:, :1], self.rescale)

        d_eta, d_alpha = nb_partials(obs[None, :], mean, dispersions)
        per_clone = np.zeros((n_states, len(lengths)))

        for state in range(n_states):
            per_clone[state] = np.bincount(
                clone_of, weights=gamma[state] * d_eta[state], minlength=len(lengths)
            )

        g_eta = per_clone.sum(axis=1) - share.T @ per_clone.sum(axis=0)

        return -g_eta, _sums(gamma, d_alpha, spread)

    def _clone_softmax(
        self, rates: np.ndarray, decode: np.ndarray, lengths: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """`S_c` per clone and the softmax weights `P[c, j]` of its segments decoded to `j`."""
        from port.patch.hmm_nophasing.shifted_emission import _stacked

        n_states = rates.size
        n_clones = len(lengths)
        terms = rates[decode] + _stacked(self.normal_log_lambda, lengths)
        clone_of = np.repeat(np.arange(n_clones), lengths)
        shifts = np.full(n_clones, -np.inf)
        np.maximum.at(shifts, clone_of, terms)
        finite = np.isfinite(shifts)
        safe = np.where(finite, shifts, 0.0)
        mass = np.exp(terms - safe[clone_of])
        sums = np.bincount(clone_of, weights=mass, minlength=n_clones)
        shifts = np.where(finite, safe + np.log(np.where(sums > 0, sums, 1.0)), shifts)

        share = np.zeros((n_clones, n_states))
        np.add.at(share, (clone_of, decode), mass)
        share = np.where(
            sums[:, None] > 0, share / np.where(sums > 0, sums, 1.0)[:, None], 0.0
        )

        return shifts, share

    def _shift_inputs(self) -> tuple[np.ndarray, tuple[int, ...]] | None:
        """The decode and clone lengths the shifted emission uses, or `None`.

        The same four conditions `compute_emission_probability_nb_betabinom_coded`
        reads, so the gradient is of the objective actually scored.
        """
        from port.patch.hmm_nophasing.shifted_emission import _current, shifted

        model = self.model
        decode = model._decode() if hasattr(model, "_decode") else None

        if (
            not shifted(model)
            or self.normal_log_lambda is None
            or self.clone_lengths is None
            or decode is None
        ):
            return None

        decode = np.asarray(decode, dtype=np.int64)
        lengths = _current(
            tuple(int(length) for length in np.asarray(self.clone_lengths)),
            int(decode.size),
        )

        return decode, lengths

    def _shifted_depth(
        self,
        encoder: CountEncoder,
        gamma: np.ndarray,
        rates: np.ndarray,
        dispersions: np.ndarray,
        decode: np.ndarray,
        lengths: tuple[int, ...],
        spread: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The shifted channel: rates `exp(log_mu_i - S_c)`, `S_c` moving with `log_mu`."""
        from port.patch.hmm_nophasing.shifted_emission import _triples

        n_states = rates.size
        triples = _triples(encoder.obs_count, encoder.total_count, lengths)
        n_codes = triples.obs.size
        n_clones = len(lengths)

        # NB the per-clone softmax over segments, `P[c, j]`, and `S_c` with it.
        shifts, share = self._clone_softmax(rates, decode, lengths)

        code_clone = np.repeat(np.arange(n_clones), np.diff(triples.bounds))
        weight = np.zeros((n_states, n_codes))

        for state in range(n_states):
            weight[state] = np.bincount(
                triples.inverse, weights=gamma[state], minlength=n_codes
            )

        log_rate = rates[:, None] - shifts[code_clone][None, :]
        mean = triples.total[None, :] * np.exp(log_rate)

        d_eta, d_alpha = nb_partials(triples.obs[None, :], mean, dispersions)
        per_clone = np.zeros((n_states, n_clones))

        for state in range(n_states):
            per_clone[state] = np.bincount(
                code_clone, weights=weight[state] * d_eta[state], minlength=n_clones
            )

        g_eta = per_clone.sum(axis=1) - share.T @ per_clone.sum(axis=0)

        return -g_eta, _sums(weight, d_alpha, spread)

    def _pack(
        self,
        x: np.ndarray,
        g_mu: np.ndarray,
        g_p: np.ndarray,
        g_alpha: np.ndarray,
        g_tau: np.ndarray,
    ) -> np.ndarray:
        """Chain rule into `pack_params`' layout, block by block, in its order."""
        params = self.model.params
        flags = self.flags
        n_states = self.n_states
        fitting_nb = self.nb is not None and "m" in params
        blocks: list[np.ndarray] = []
        idx = 0

        if "s" in params:
            # NB `cost_fn` in "em" mode never reads the start probabilities.
            blocks.append(np.zeros(n_states))
            idx += n_states

        if fitting_nb:
            blocks.append(g_mu)
            idx += n_states

        if "p" in params:
            raw = x[idx : idx + n_states]

            if flags["use_logit"]:
                p = expit(raw)
                blocks.append(g_p * p * (1.0 - p))
            else:
                inside = (raw > 1e-6) & (raw < 1.0 - 1e-6)
                blocks.append(np.where(inside, g_p, 0.0))

            idx += n_states

        if fitting_nb and not flags["fix_NB_dispersion"]:
            blocks.append(
                np.array([g_alpha.sum()]) if flags["shared_NB_dispersion"] else g_alpha
            )

        if "p" in params and not flags["fix_BB_dispersion"]:
            blocks.append(
                np.array([g_tau.sum()]) if flags["shared_BB_dispersion"] else g_tau
            )

        return np.concatenate(blocks) if blocks else np.zeros(0)


BFGS_OPTIONS = frozenset(
    {
        "maxiter",
        "gtol",
        "norm",
        "eps",
        "disp",
        "return_all",
        "finite_diff_rel_step",
        "xrtol",
        "c1",
        "c2",
        "hess_inv0",
    }
)
"""The options `scipy.optimize.minimize(method="BFGS")` reads (scipy 1.18)."""


DISPERSION_PRIOR_ROWS = 0.0
"""`n0`, the rows' worth of pooled information each state's dispersion is shrunk with (#566).

0, no shrinkage, unless `--dispersion-prior-rows` states one. Held-out
likelihood on #540's `dev_tree_1s_hard` r0-r2 (`d2938975`, `764709dc`,
`3adf249a`; fit on r0 and r1, scored on the other two; grid 0, 10, 30, 100,
300, 1000) picks 30, by a mode change on r1 rather than a trend: there the
neutral state splits in two at `n0 >= 30` (PR #567).
"""


def dispersion_blocks(gradient: EmGradient) -> dict[str, slice]:
    """Where `pack_params` puts the per-state `log alpha` (`nb`) and `log tau` (`bb`); shared or fixed blocks are left out."""
    params = gradient.model.params
    flags = gradient.flags
    n_states = gradient.n_states
    fitting_nb = gradient.nb is not None and "m" in params

    idx = n_states * (("s" in params) + fitting_nb + ("p" in params))
    blocks: dict[str, slice] = {}

    if fitting_nb and not flags["fix_NB_dispersion"]:
        if not flags["shared_NB_dispersion"]:
            blocks["nb"] = slice(idx, idx + n_states)
        idx += 1 if flags["shared_NB_dispersion"] else n_states

    if (
        "p" in params
        and not flags["fix_BB_dispersion"]
        and not flags["shared_BB_dispersion"]
    ):
        blocks["bb"] = slice(idx, idx + n_states)

    return blocks


ALPHA_MIN = 1e-3
"""The per-state NB `alpha`'s lower bound with `--per-state-dispersion` (#566).

Below every pooled `alpha` measured: 0.589 and 0.593 on CalicoST hard
(`1ae26365`) and easy (`23989aa4`) under `--sal`, 0.058 and 0.072 on #540's
`dev_tree_1s_hard` r1 (`764709dc`) and r0 (`d2938975`).
"""

TAU_MAX = 1e5
"""The per-state BB `tau`'s upper bound, a concentration cap (#566).

Above every finite pooled `tau` measured: 1,062 and 4,882 on CalicoST hard
(`1ae26365`) and easy (`23989aa4`), 15,324 on #540's r1 (`764709dc`). #540's
r0 (`d2938975`) pools at 3.5e6, the binomial in effect; at 111 trials, `tau = 1e5` inflates the variance by 0.1 per cent.
"""


@dataclass
class DispersionBounds:
    """`alpha_k >= alpha_min` and `tau_k <= tau_max` on the per-state blocks, by projection (#566).

    `cnaster` minimizes with BFGS and `bounds=None`, and BFGS takes no box.
    So the M step minimizes the projected objective `f(clip(x))`: `cost_fn`
    and the gradient are read at the clipped point, a coordinate past its
    bound has zero gradient, and the result is returned clipped, so what
    `cnaster` unpacks is inside the box. Bounds stop a runaway -- `alpha`
    to 0 or `tau` to the binomial -- and sit outside what a fit reaches.
    """

    blocks: dict[str, slice]
    log_alpha_min: float
    log_tau_max: float

    @classmethod
    def for_fit(
        cls, gradient: EmGradient, alpha_min: float, tau_max: float
    ) -> DispersionBounds:
        return cls(
            dispersion_blocks(gradient),
            float(np.log(alpha_min)),
            float(np.log(tau_max)),
        )

    def clip(self, x: np.ndarray) -> np.ndarray:
        """`x` with each per-state block inside its bound."""
        out = np.array(x, dtype=np.float64, copy=True)

        if "nb" in self.blocks:
            out[self.blocks["nb"]] = np.maximum(
                out[self.blocks["nb"]], self.log_alpha_min
            )
        if "bb" in self.blocks:
            out[self.blocks["bb"]] = np.minimum(
                out[self.blocks["bb"]], self.log_tau_max
            )

        return out

    def free(self, x: np.ndarray) -> np.ndarray:
        """1 where a coordinate is not past its bound, else 0: the projected gradient's mask."""
        mask = np.ones_like(x, dtype=np.float64)

        if "nb" in self.blocks:
            mask[self.blocks["nb"]] = x[self.blocks["nb"]] >= self.log_alpha_min
        if "bb" in self.blocks:
            mask[self.blocks["bb"]] = x[self.blocks["bb"]] <= self.log_tau_max

        return mask


@dataclass
class DispersionShrinkage:
    r"""Per-state dispersions shrunk toward their pooled value, as a penalty (#566).

    With per-state `log alpha_k` (and `log tau_k`) a state holding a handful
    of rows can drive its `alpha` to 0, which scores those rows at
    near-certainty (#560). Each per-state block `theta` is penalized

    .. math::
        P(\theta) = \frac{n_0 h}{2} \sum_k (\theta_k - \bar\theta)^2,
        \qquad \bar\theta = \sum_k \frac{n_k}{N} \theta_k,

    with `n_k = sum_i gamma_ik` a state's posterior occupancy (held, as the
    data term holds it) and `h` the per-row information of the pooled fit
    in `theta`. The data term's curvature in `theta_k` is then `n_k h`, so
    to second order the minimizer is

    .. math::
        \theta_k \approx \frac{n_k \hat\theta_k + n_0 \bar\theta}{n_k + n_0},

    the shrinkage toward the pooled estimate with weight `n0 / (n0 + n_k)`
    #566 states. A state with no rows sits at the pooled value; equal
    `theta` cost nothing, so a fit whose states agree returns the pooled one.

    `h` is held per fit: the per-row score variance in `theta`, each state's
    scores centred on their own mean, pooled over states and divided by `N`
    (:meth:`EmGradient.information`), at the first point BFGS evaluates.
    Held, so `P` is one function for the whole fit and its gradient is
    exact. The score variance rather than the curvature, because the
    curvature can be negative away from the optimum, and the first point is
    the fit's start.
    """

    prior_rows: float
    model: Any
    blocks: dict[str, slice]
    information: dict[str, float]

    @classmethod
    def for_fit(cls, gradient: EmGradient, prior_rows: float) -> DispersionShrinkage:
        """The per-state blocks of `gradient`'s packed layout; shared or fixed ones are left out."""
        return cls(float(prior_rows), gradient.model, dispersion_blocks(gradient), {})

    def _occupancy(self) -> np.ndarray:
        """`n_k`: each state's summed posterior over the rows."""
        return np.asarray(self.model.state_posteriors, dtype=np.float64).sum(axis=1)

    def hold_information(self, x: np.ndarray, data_gradient: EmGradient) -> None:
        """`h` per block, once: the per-row information `data_gradient` reads at `x`."""
        if self.information:
            return

        held = data_gradient.information(x)
        self.information = {name: held.get(name, 0.0) for name in self.blocks}

    def _parts(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`theta - theta_bar` and the occupancy weights `n_k / N`."""
        occupancy = self._occupancy()
        weights = occupancy / max(float(occupancy.sum()), np.finfo(float).tiny)
        return theta - float(weights @ theta), weights

    def value(self, x: np.ndarray) -> float:
        """`P(x)`, summed over the per-state blocks."""
        total = 0.0

        for name, block in self.blocks.items():
            spread, _ = self._parts(x[block])
            scale = self.prior_rows * self.information.get(name, 0.0)
            total += 0.5 * scale * float(spread @ spread)

        return total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """`d P / d x`: `n0 h (d_j - w_j sum_k d_k)` in each block, zero elsewhere."""
        out = np.zeros_like(x, dtype=np.float64)

        for name, block in self.blocks.items():
            spread, weights = self._parts(x[block])
            scale = self.prior_rows * self.information.get(name, 0.0)
            out[block] = scale * (spread - weights * spread.sum())

        return out


def analytic_bfgs(
    gradient: Callable[[np.ndarray], np.ndarray],
    penalty: DispersionShrinkage | None = None,
    bounds: DispersionBounds | None = None,
) -> Any:
    """A `scipy.optimize.minimize` `method` that is BFGS with `gradient` as `jac`.

    `cnaster` passes its `optimizer` argument to `minimize` as `method`, and a
    callable there is `scipy`'s custom-method protocol. So the fit stays
    `cnaster`'s -- its `cost_fn`, its callback, its options -- and only the
    gradient changes. Value and gradient are taken together (`jac=True`), so
    the gradient always reads the posteriors the value just used, and the
    callback's E step reads the emission at a point BFGS evaluated rather
    than at a finite-difference probe.

    With `penalty` (#566) BFGS minimizes `cost_fn + P` and its gradient;
    `cnaster`'s `cost_fn` and callback are unchanged, and `P`'s information
    scale is held at the first point evaluated, once the posteriors exist.
    With `bounds` (#566) it minimizes `f(clip(x))` and returns the clipped
    point (:class:`DispersionBounds`).
    """

    def method(
        fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **kwargs: Any
    ) -> scipy.optimize.OptimizeResult:
        callback = kwargs.pop("callback", None)
        # NB BFGS's own options only (#448): `cnaster` passes `ftol`, which
        #    BFGS has not got, and `scipy` warns "Unknown solver options:
        #    ftol" and drops it -- 13 times a run. Dropping it here is the
        #    same fit, without the warning.
        options = {key: value for key, value in kwargs.items() if key in BFGS_OPTIONS}

        def value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
            point = x if bounds is None else bounds.clip(x)
            value = float(fun(point, *args))
            slope = gradient(point)

            if penalty is not None:
                if isinstance(gradient, EmGradient):
                    penalty.hold_information(point, gradient)
                value += penalty.value(point)
                slope = slope + penalty(point)

            return value, slope if bounds is None else slope * bounds.free(x)

        result = scipy.optimize.minimize(
            value_and_gradient,
            x0 if bounds is None else bounds.clip(x0),
            jac=True,
            method="BFGS",
            callback=callback,
            options=options,
        )

        if bounds is not None:
            result.x = bounds.clip(result.x)

        return result

    return method


def configured_solver() -> tuple[str, dict[str, float]]:
    """`hmm.solver` and the `em_*` options `cnaster` pairs with it (#448).

    `get_em_solver_params` is `cnaster`'s own map from the solver to its keys
    -- `L-BFGS-B` reads `em_maxiter` and `em_ftol`, `BFGS` `em_xrtol` -- and
    its M step reads neither: it runs BFGS whatever the configuration says.
    `("BFGS", {})` where no configuration is set.
    """
    from cnaster.config import get_global_config
    from cnaster.hmm_utils import get_em_solver_params

    try:
        solver = str(get_global_config().hmm.solver)
        params = get_em_solver_params()
    except (AttributeError, KeyError, TypeError, ValueError):
        return "BFGS", {}

    params.pop("disp", None)
    return solver, params


def configured_method(gradient: Callable[[np.ndarray], np.ndarray]) -> Any:
    """The M step as the configuration states it, with `gradient` where it is used.

    **Not installed** (#448): `cnaster` runs BFGS whatever `hmm.solver` says,
    and the shipped configurations state `L-BFGS-B`, so installing this
    changes the default fit. It is here for that decision, measured.

    `BFGS` is :func:`analytic_bfgs` plus `em_xrtol`; `L-BFGS-B` takes
    `em_maxiter` and `em_ftol` and `cnaster`'s `gtol`; `Nelder-Mead`, which
    uses no gradient, takes `em_maxiter`, `em_xtol` and `em_ftol` as its
    `maxiter`, `xatol` and `fatol`.
    """
    solver, params = configured_solver()

    if solver == "BFGS":
        bfgs = analytic_bfgs(gradient)

        def with_xrtol(
            fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **kwargs: Any
        ) -> Any:
            if "xrtol" in params:
                kwargs["xrtol"] = params["xrtol"]
            return bfgs(fun, x0, args, **kwargs)

        return with_xrtol

    def method(
        fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **kwargs: Any
    ) -> scipy.optimize.OptimizeResult:
        callback = kwargs.pop("callback", None)

        if solver == "L-BFGS-B":
            options = {"gtol": kwargs.get("gtol", 1e-5)}
            options["maxiter"] = int(
                params.get("maxiter", kwargs.get("maxiter", 15000))
            )
            options["ftol"] = float(params.get("ftol", 2.2e-9))

            def value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
                return float(fun(x, *args)), gradient(x)

            return scipy.optimize.minimize(
                value_and_gradient,
                x0,
                jac=True,
                method="L-BFGS-B",
                callback=callback,
                options=options,
            )

        if solver == "Nelder-Mead":
            options = {
                "maxiter": int(params.get("maxiter", kwargs.get("maxiter", 1000))),
                "xatol": float(params.get("xtol", 1e-4)),
                "fatol": float(params.get("ftol", 1e-4)),
            }
            return scipy.optimize.minimize(
                fun,
                x0,
                args=args,
                method="Nelder-Mead",
                callback=callback,
                options=options,
            )

        msg = f"hmm.solver {solver!r} is not one cnaster supports"
        raise ValueError(msg)

    return method
