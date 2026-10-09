r"""The M step's gradient in closed form, where `cnaster`'s BFGS differences it (#433).

.. math::
    f(x) = -\sum_{i,u} W_{iu} \left[\ell^{NB}_{iu} + \ell^{BB}_{iu}\right],
    \qquad W_{iu} = \sum_{g \to u} \gamma_{ig},

with partials from `port.patch.emission` (T- #776). Under the shift the rate
is `exp(log_mu_i - S_c)` and `d eta / d log_mu_j = delta_ij - P_cj`. `cnaster`'s
`cost_fn` and callback still drive the fit. Referee: `port.qa.jax_hmm`.
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
    """The EM objective's gradient for one fit, in the fit's packed coordinates, at the model's current posteriors."""

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
        """Read the fit's settings, defaulting from `_run_optimization_pipeline`'s own signature."""
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(
                type(model)._run_optimization_pipeline
            ).parameters.items()
        }
        setting = defaults | kwargs

        base = np.array(base_nb_mean, dtype=np.float64)
        max_rdr = setting["max_rdr"]

        # NB `hmm_nophasing.py:822-826`: bins above `max_rdr` lose their baseline.
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

        # NB zeros where depth is not fitted; `_pack` leaves those blocks out.
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
        """The decode and clone lengths the shifted emission uses, or `None` under the same conditions."""
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
    """A `scipy.optimize.minimize` `method`: BFGS with `gradient` as `jac`.

    Keeps `cnaster`'s `cost_fn`, callback and options; value and gradient taken together.
    """

    def method(
        fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **kwargs: Any
    ) -> scipy.optimize.OptimizeResult:
        callback = kwargs.pop("callback", None)
        # NB BFGS's own options only (#448): drops `cnaster`'s `ftol`, which scipy warns on.
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
    """`hmm.solver` and the `em_*` options `cnaster` pairs with it, or `("BFGS", {})` unconfigured (#448)."""
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
    """The M step as the configuration states it, with `gradient` where used.

    Not installed (#448): `cnaster` runs BFGS whatever `hmm.solver` says.
    Supports `BFGS`, `L-BFGS-B` and `Nelder-Mead`; raises `ValueError` otherwise.
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
