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

and each term's derivative is a rising digamma or a ratio, the partials of
`port.patch.emission`'s densities, the one evaluation every site scores
(T- #776, `sal`'s Rust `coded.log_emission_partials` since sal #1353;
:func:`~port.patch.emission.nb_partials`,
:func:`~port.patch.emission.bb_partials`):

- negative binomial, `r = 1 / max(alpha, 1e-10)`, `mu = c exp(eta)`,
  `p = 1 / (1 + max(alpha, 1e-10) mu)`, logged as `-log1p` (#560): `d ell / d eta = k - (r + k) alpha mu / (1 + alpha mu)`,
  and, where `alpha` is above the floor, `d ell / d log alpha =
  -r (psi(k + r) - psi(r) + log p) + k - (r + k) alpha mu / (1 + alpha mu)`,
  the `psi` pair `sal`'s `digamma_rising`, so nothing cancels at `r` up to
  1e10;
- beta-binomial, `a = max(p tau, 1e-10)`, `b = max((1 - p) tau, 1e-10)`:
  `d ell / d a = psi(k + a) - psi(n + a + b) - psi(a) + psi(a + b)`, and
  `b`'s with `n - k` for `k`, each `digamma` pair `sal`'s `digamma_rising`
  so nothing near `log tau` cancels at a large `tau` (#561, T- #781).

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
from scipy.special import expit

from port.patch.emission import (
    DISPERSION_FLOOR,
    bb_partial_sums,
    bb_partials,
    nb_partial_sums,
    nb_partials,
)

__all__ = [
    "DISPERSION_FLOOR",
    "EmGradient",
    "analytic_bfgs",
    "bb_partials",
    "configured_method",
    "configured_solver",
    "nb_partials",
]


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

        d_p, d_tau = bb_partial_sums(
            obs,
            total,
            p_binom[:, 0],
            taus[:, 0],
            np.broadcast_to(weight, (p_binom.shape[0], obs.size)),
        )

        return -d_p, -d_tau

    def _depth(
        self,
        encoder: CountEncoder,
        gamma: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-state `d f / d log mu` and `d f / d log alpha`, shifted or not."""
        rates = np.asarray(log_mu, dtype=np.float64)[:, 0]
        dispersions = np.asarray(alphas, dtype=np.float64)[:, 0]

        shifted = self._shift_inputs()

        if shifted is None:
            obs = encoder.get_unique_obs(0)
            exposure = encoder.get_unique_total(0)
            weight = np.asarray(encoder.encode_array(gamma, 0))

            d_eta, d_alpha = nb_partial_sums(
                obs,
                exposure,
                rates,
                dispersions,
                np.broadcast_to(weight, (rates.size, obs.size)),
            )

            return -d_eta, -d_alpha

        return self._shifted_depth(encoder, gamma, rates, dispersions, *shifted)

    def _shift_inputs(self) -> tuple[np.ndarray, tuple[int, ...]] | None:
        """The decode and clone lengths the shifted emission uses, or `None`.

        The same four conditions `compute_emission_probability_nb_betabinom_coded`
        reads, so the gradient is of the objective actually scored.
        """
        from port.patch.hmm_nophasing.shifted_emission import (
            current_clone_lengths,
            shifted,
        )

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
        lengths = current_clone_lengths(
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
        from port.patch.hmm_nophasing.shifted_emission import (
            clone_count_triples,
            stacked_log_lambda,
        )

        n_states = rates.size
        triples = clone_count_triples(encoder.obs_count, encoder.total_count, lengths)
        n_codes = triples.obs.size
        n_clones = len(lengths)

        # NB the per-clone softmax over segments, `P[c, j]`, and `S_c` with it.
        terms = rates[decode] + stacked_log_lambda(self.normal_log_lambda, lengths)
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

        # NB the clone's shift enters as the exposure `total exp(-S_c)`, the rate `exp(log_mu)` per state
        exposure = triples.total * np.exp(-shifts[code_clone])

        d_eta, d_alpha = nb_partials(triples.obs, exposure, rates, dispersions)
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
