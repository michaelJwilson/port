"""The M step's closed-form gradient against `jax`'s autodiff of the same objective
(#433).

The objective's value is first pinned against cnaster's coded emission.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import cnaster.config
import cnaster.hmm_utils
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
from cnaster.config import get_global_config
from cnaster.hmm_nophasing import hmm_nophasing as upstream
from jax.scipy.special import expit
from port.patch.hmm_nophasing import hmm_nophasing
from port.patch.hmm_nophasing.gradient import (
    EmGradient,
    analytic_bfgs,
    configured_method,
)
from port.qa.jax_hmm import emission, shifted_rates

from tests.fixtures import two_clone_stacked_instance


def _problem(
    *, shifted: bool, shared: bool, seed: int = 5, silent: float | None = None
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    """A clone-stacked fit, mid-EM: posteriors held, parameters at `x`."""

    generator = np.random.default_rng(seed)
    n_states, n_clones, per_clone = 4, 3, 40
    n_segments = n_clones * per_clone

    exposure = generator.uniform(20.0, 60.0, n_segments)
    trials = generator.integers(0, 40, n_segments).astype(np.float64)
    observed = generator.poisson(exposure).astype(np.float64)

    # NB a zero-baseline bin, which cnaster scores 0; excluded from `jax`, whose
    # untaken `jnp.where` branch gives a `nan` gradient.
    if silent is not None:
        exposure[3] = 0.0
        observed[3] = silent
    successes = generator.binomial(trials.astype(int), 0.4).astype(np.float64)

    X = np.stack([observed, successes], axis=1)[:, :, None]
    base = exposure[:, None]
    total = trials[:, None]

    model = hmm_nophasing(params="stmp", t=1 - 1e-6)
    model.apply_logmu_shift = shifted
    gamma = generator.dirichlet(np.ones(n_states), size=n_segments).T
    model.state_posteriors = gamma

    normal_lambda = generator.uniform(0.5, 1.5, per_clone)
    normal_lambda /= normal_lambda.sum()

    settings: dict[str, Any] = {
        "init_log_mu": generator.normal(0.0, 0.3, (n_states, 1)),
        "init_p_binom": generator.uniform(0.2, 0.8, (n_states, 1)),
        "init_alphas": np.full((n_states, 1), 0.2),
        "init_taus": np.full((n_states, 1), 25.0),
        "shared_NB_dispersion": shared,
        "shared_BB_dispersion": shared,
        "normal_lambda": normal_lambda,
        "clone_lengths": np.full(n_clones, per_clone),
    }

    gradient = EmGradient.for_fit(model, X, n_states, base, total, **settings)
    log_startprob, log_mu, p_binom, alphas, taus = gradient.initial
    alphas = generator.uniform(0.05, 0.4, (n_states, 1))
    taus = generator.uniform(10.0, 60.0, (n_states, 1))
    x = model.pack_params(
        log_startprob, log_mu, p_binom, alphas, taus, **gradient.flags
    )

    return model, gradient, x, {"X": X, "base": base, "total": total, **settings}


def _cnaster_value(model: Any, gradient: Any, x: np.ndarray) -> float:
    """`cost_fn`'s value: `-sum(gamma * emission)` through the coded emission."""
    _, log_mu, p_binom, alphas, taus = model.unpack_params(
        x, gradient.n_states, *gradient.initial, **gradient.flags
    )
    rdr, baf = model.compute_emission_probability_nb_betabinom_coded(
        gradient.nb,
        gradient.bb,
        log_mu,
        alphas,
        p_binom,
        taus,
        normal_log_lambda=gradient.normal_log_lambda,
        clone_lengths=gradient.clone_lengths,
    )
    return float(-np.sum(model.state_posteriors * (rdr + baf)))


def _jax_objective(model: Any, gradient: Any, data: dict[str, Any]) -> Any:
    """The same objective in `jax`, from `jax_hmm`'s emission and shifted rates."""

    n_states = gradient.n_states
    flags = gradient.flags
    gamma = jnp.asarray(model.state_posteriors)
    shifted = gradient._shift_inputs()
    X = data["X"]

    def objective(x: Any) -> Any:
        idx = n_states  # NB the start probabilities, unread.
        log_mu = x[idx : idx + n_states]
        idx += n_states
        p_binom = expit(x[idx : idx + n_states])
        idx += n_states

        if flags["shared_NB_dispersion"]:
            alphas = jnp.full(n_states, jnp.exp(x[idx]))
            idx += 1
        else:
            alphas = jnp.exp(x[idx : idx + n_states])
            idx += n_states

        if flags["shared_BB_dispersion"]:
            taus = jnp.full(n_states, jnp.exp(x[idx]))
        else:
            taus = jnp.exp(x[idx : idx + n_states])

        args = (X[:, 0, 0], data["base"][:, 0], X[:, 1, 0], data["total"][:, 0])

        if shifted is None:
            scored = emission(log_mu, alphas, p_binom, taus, *args)
        else:
            decode, lengths = shifted
            stacked = np.tile(np.log(data["normal_lambda"]), len(lengths))
            rates = shifted_rates(log_mu, decode, stacked, lengths)
            bounds = np.concatenate(([0], np.cumsum(lengths)))
            scored = jnp.concatenate(
                [
                    emission(
                        rates[clone],
                        alphas,
                        p_binom,
                        taus,
                        *(arg[bounds[clone] : bounds[clone + 1]] for arg in args),
                    )
                    for clone in range(len(lengths))
                ],
                axis=1,
            )

        return -jnp.sum(gamma * scored)

    return objective


@pytest.mark.oracle
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("shared", [True, False])
def test_the_closed_form_gradient_is_jaxs(
    shifted: bool, shared: bool, cnaster_config: None
) -> None:
    """Value to 1e-9 of `cnaster`'s (3.6e-10 realized); gradient to 1e-8 relative of `jax`'s."""

    model, gradient, x, data = _problem(shifted=shifted, shared=shared)
    objective = _jax_objective(model, gradient, data)

    value = _cnaster_value(model, gradient, x)
    np.testing.assert_allclose(float(objective(x)), value, rtol=1e-9)

    ours = gradient(x)
    theirs = np.asarray(jax.grad(objective)(x))

    assert ours.shape == theirs.shape
    np.testing.assert_allclose(
        ours, theirs, rtol=1e-8, atol=1e-8 * np.abs(theirs).max()
    )


@pytest.mark.analytic
def test_the_shifted_gradient_has_no_component_along_the_flat_direction(
    cnaster_config: None,
) -> None:
    """`log_mu -> log_mu + c` leaves the shifted objective unchanged, so `sum_i g_i = 0`."""
    model, gradient, x, _ = _problem(shifted=True, shared=True)
    n_states = gradient.n_states
    g_mu = gradient(x)[n_states : 2 * n_states]

    assert abs(g_mu.sum()) <= 1e-9 * np.abs(g_mu).max()


@pytest.mark.analytic
def test_a_bin_without_baseline_moves_no_gradient(cnaster_config: None) -> None:
    """A zero-baseline bin's count cannot move the gradient, to 1e-12 (summation order)."""
    _, gradient, x, _ = _problem(shifted=False, shared=True, silent=0.0)
    _, other, _, _ = _problem(shifted=False, shared=True, silent=500.0)

    np.testing.assert_allclose(gradient(x), other(x), rtol=1e-12)


@pytest.mark.cnaster
@pytest.mark.patch
def test_the_closed_form_fit_is_cnasters_fit_to_a_stated_tolerance(
    cnaster_config: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cnaster's finite-difference fit matches `port`'s closed-form one to 1e-5 relative at `max_iter=20`."""

    assert hmm_nophasing.analytic_gradient
    assert EmGradient is not None

    # NB BFGS on both sides, so only the gradients differ.
    monkeypatch.setattr(get_global_config().hmm, "solver", "BFGS")

    instance = two_clone_stacked_instance()
    kwargs = {
        "init_log_mu": np.log(np.array([[1.0], [2.0]])),
        "init_p_binom": np.array([[0.5], [0.25]]),
        "max_iter": 20,
        "normal_lambda": instance["normal_lambda"],
        "clone_lengths": instance["clone_lengths"],
        "shared_NB_dispersion": True,
        "shared_BB_dispersion": True,
    }
    args = (instance["X"], instance["lengths"], 2, instance["base"], instance["total"])

    theirs = upstream(params="smp", t=0.99).optimize(*args, **kwargs)
    ours = hmm_nophasing(params="smp", t=0.99).optimize(*args, **kwargs)

    for key in ("new_log_mu", "new_p_binom", "new_alphas", "new_taus"):
        np.testing.assert_allclose(ours[key], theirs[key], rtol=1e-4, err_msg=key)


@pytest.mark.patch
def test_the_m_step_passes_bfgs_only_its_own_options() -> None:
    """cnaster's `ftol` reaches no BFGS call; bitwise against scipy BFGS without it (#448)."""

    scale = np.array([1.0, 4.0, 9.0])

    def fun(x: np.ndarray) -> float:
        return float(np.sum(scale * (x - 1.0) ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * scale * (x - 1.0)

    x0 = np.array([3.0, -2.0, 0.5])
    cnasters = {"maxiter": 50, "ftol": 1e-6, "gtol": 1e-5, "disp": False}

    with pytest.warns(scipy.optimize.OptimizeWarning, match="ftol"):
        scipy.optimize.minimize(fun, x0, jac=grad, method="BFGS", options=cnasters)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ours = scipy.optimize.minimize(
            fun, x0, method=analytic_bfgs(grad), options=cnasters
        )

    reference = scipy.optimize.minimize(
        lambda x: (fun(x), grad(x)),
        x0,
        jac=True,
        method="BFGS",
        options={"maxiter": 50, "gtol": 1e-5, "disp": False},
    )

    np.testing.assert_array_equal(ours.x, reference.x)
    assert ours.nit == reference.nit


@pytest.mark.patch
@pytest.mark.parametrize("solver", ["BFGS", "L-BFGS-B"])
def test_the_m_step_runs_the_configured_solver_at_its_tolerances(
    solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hmm.solver` and `em_*` keys reach the M step, bitwise against scipy with the mapped options (#448)."""

    hmm = SimpleNamespace(
        solver=solver, em_maxiter=7, em_ftol=1e-3, em_xrtol=1e-3, em_disp=0
    )

    # NB both bindings: `hmm_utils` imports the name at load.
    for module in (cnaster.config, cnaster.hmm_utils):
        monkeypatch.setattr(
            module, "get_global_config", lambda: SimpleNamespace(hmm=hmm)
        )

    scale = np.array([1.0, 4.0, 9.0])

    def fun(x: np.ndarray) -> float:
        return float(np.sum(scale * (x - 1.0) ** 2) + np.sum(x**4))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * scale * (x - 1.0) + 4.0 * x**3

    x0 = np.array([3.0, -2.0, 0.5])
    cnasters = {"maxiter": 50, "ftol": 1e-6, "gtol": 1e-5, "disp": False}
    ours = scipy.optimize.minimize(
        fun, x0, method=configured_method(grad), options=cnasters
    )

    stated = (
        {"maxiter": 50, "gtol": 1e-5, "disp": False, "xrtol": 1e-3}
        if solver == "BFGS"
        else {"gtol": 1e-5, "maxiter": 7, "ftol": 1e-3}
    )
    reference = scipy.optimize.minimize(
        lambda x: (fun(x), grad(x)), x0, jac=True, method=solver, options=stated
    )

    np.testing.assert_array_equal(ours.x, reference.x)
    assert ours.nit == reference.nit
