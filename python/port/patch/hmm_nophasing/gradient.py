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
  `p = 1 / (1 + max(alpha, 1e-10) mu)`, logged as `-log1p` (#560): `d ell / d eta = k - (r + k) alpha mu / (1 + alpha mu)`,
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
    "DISPERSION_FLOOR",
    "EmGradient",
    "analytic_bfgs",
    "bb_partials",
    "configured_method",
    "configured_solver",
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

    Broadcasts. A bin with no exposure scores 0 and has zero derivative.

    The derivative of the #560-corrected score
    (`port.sandbox.patch.hmm_nophasing.nb_logpmf`), with `a = max(alpha, 1e-10) *
    mean` and `log p = -log1p(a)`. Upstream's unpatched kernel also scores 0
    where `p = 1 / (1 + alpha * mean)` rounds to 1 (`a` below about
    1.1e-16) and floors `alpha` in `r` but not in `p`; this derivative
    follows neither defect, so below the floor `alpha` moves nothing.
    """
    alpha = np.asarray(dispersion, dtype=np.float64)
    floored = np.maximum(alpha, DISPERSION_FLOOR)
    size = 1.0 / floored
    scaled = floored * mean
    live = mean > 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (size + obs) * scaled / (1.0 + scaled)
        d_eta = np.where(live, obs - pull, 0.0)

        # NB below the floor `r` is a constant and only `p` moves with alpha.
        through_size = np.where(
            alpha > DISPERSION_FLOOR,
            -size * (digamma(obs + size) - digamma(size) - np.log1p(scaled)),
            0.0,
        )
        d_alpha = np.where(
            live & (alpha > DISPERSION_FLOOR), through_size + obs - pull, 0.0
        )

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
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """`d f / d x` at `x`, `f` being `cnaster`'s `cost_fn`."""
        model = self.model
        _, log_mu, p_binom, alphas, taus = model.unpack_params(
            x, self.n_states, *self.initial, **self.flags
        )
        gamma = np.asarray(model.state_posteriors)

        g_p, g_tau = self._allele(gamma, p_binom, taus)

        # NB zeros where the depth channel is not fitted; `_pack` then
        #    leaves its blocks out, as `pack_params` does.
        g_mu = g_alpha = np.zeros(self.n_states)

        if self.nb is not None and "m" in model.params:
            g_mu, g_alpha = self._depth(self.nb, gamma, log_mu, alphas)

        return self._pack(x, g_mu, g_p, g_alpha, g_tau)

    def _allele(
        self, gamma: np.ndarray, p_binom: np.ndarray, taus: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-state `d f / d p` and `d f / d log tau`."""
        obs = self.bb.get_unique_obs(0)
        total = self.bb.get_unique_total(0)
        weight = np.asarray(self.bb.encode_array(gamma, 0))

        d_p, d_tau = bb_partials(
            obs[None, :], total[None, :], p_binom[:, :1], taus[:, :1]
        )

        return -np.sum(weight * d_p, axis=1), -np.sum(weight * d_tau, axis=1)

    def _depth(
        self,
        encoder: CountEncoder,
        gamma: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
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

            return -np.sum(weight * d_eta, axis=1), -np.sum(weight * d_alpha, axis=1)

        return self._shifted_depth(encoder, gamma, rates, dispersions, *shifted)

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """The shifted channel: rates `exp(log_mu_i - S_c)`, `S_c` moving with `log_mu`."""
        from port.patch.hmm_nophasing.shifted_emission import _stacked, _triples

        n_states = rates.size
        triples = _triples(encoder.obs_count, encoder.total_count, lengths)
        n_codes = triples.obs.size
        n_clones = len(lengths)

        # NB the per-clone softmax over segments, `P[c, j]`, and `S_c` with it.
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

        return -g_eta, -np.sum(weight * d_alpha, axis=1)

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


def analytic_bfgs(gradient: Callable[[np.ndarray], np.ndarray]) -> Any:
    """A `scipy.optimize.minimize` `method` that is BFGS with `gradient` as `jac`.

    `cnaster` passes its `optimizer` argument to `minimize` as `method`, and a
    callable there is `scipy`'s custom-method protocol. So the fit stays
    `cnaster`'s -- its `cost_fn`, its callback, its options -- and only the
    gradient changes. Value and gradient are taken together (`jac=True`), so
    the gradient always reads the posteriors the value just used, and the
    callback's E step reads the emission at a point BFGS evaluated rather
    than at a finite-difference probe.
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
            value = float(fun(x, *args))
            return value, gradient(x)

        return scipy.optimize.minimize(
            value_and_gradient,
            x0,
            jac=True,
            method="BFGS",
            callback=callback,
            options=options,
        )

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
