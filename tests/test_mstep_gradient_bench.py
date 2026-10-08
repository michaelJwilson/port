"""One `hmm_nophasing.optimize` fit, closed-form gradient against `cnaster`'s finite
differences (#433).

Eight states, shared dispersions, `max_iter=3`, shift on. Stress (`release`): 10 clones
of 3,000 bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

GATE = {"n_clones": 3, "n_obs": 300}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_clones": 10, "n_obs": 3_000}
"""30,000 clone-stacked segments."""

N_STATES = 8


def _fit_inputs(n_clones: int, n_obs: int) -> dict[str, Any]:
    """A clone-stacked instance from eight planted states, and a start near them."""
    generator = np.random.default_rng(11)
    n_segments = n_clones * n_obs

    rates = np.linspace(-0.7, 0.7, N_STATES)
    probabilities = np.linspace(0.15, 0.5, N_STATES)
    planted = generator.integers(0, N_STATES, n_segments)

    exposure = generator.uniform(200.0, 400.0, n_segments)
    trials = generator.integers(10, 60, n_segments).astype(np.float64)
    observed = generator.negative_binomial(
        10.0, 10.0 / (10.0 + exposure * np.exp(rates[planted]))
    )
    successes = generator.binomial(trials.astype(int), probabilities[planted])

    X = np.stack([observed, successes], axis=1).astype(np.float64)[:, :, None]
    normal_lambda = generator.uniform(0.5, 1.5, n_obs)

    return {
        "X": X,
        "lengths": np.full(n_clones, n_obs),
        "n_states": N_STATES,
        "base_nb_mean": exposure[:, None],
        "total_bb_RD": trials[:, None],
        "log_sitewise_transmat": np.full(n_segments, np.log(1e-4)),
        "shared_NB_dispersion": True,
        "shared_BB_dispersion": True,
        "init_log_mu": (rates + 0.1)[:, None],
        "init_p_binom": (probabilities + 0.03)[:, None],
        "init_alphas": np.full((N_STATES, 1), 0.2),
        "init_taus": np.full((N_STATES, 1), 30.0),
        "max_iter": 3,
        "normal_lambda": normal_lambda / normal_lambda.sum(),
        "clone_lengths": np.full(n_clones, n_obs),
    }


def _fit(inputs: dict[str, Any], *, analytic: bool) -> dict[str, Any]:
    from port.patch.hmm_nophasing import hmm_nophasing
    from port.pipeline import with_attributes

    arguments = dict(inputs)
    X = arguments.pop("X")
    lengths = arguments.pop("lengths")
    n_states = arguments.pop("n_states")
    base = arguments.pop("base_nb_mean")
    total = arguments.pop("total_bb_RD")

    model = with_attributes(
        hmm_nophasing, apply_logmu_shift=True, analytic_gradient=analytic
    )
    result: dict[str, Any] = model(params="smp", t=1 - 1e-6).optimize(
        X, lengths, n_states, base, total, **arguments
    )
    return result


@pytest.mark.benchmark
@pytest.mark.parametrize("analytic", [True, False], ids=["analytic", "finite"])
def test_one_m_step_fit_at_the_gate_size(
    analytic: bool, benchmark: BenchmarkFixture, cnaster_config: None
) -> None:
    inputs = _fit_inputs(**GATE)
    _fit(inputs, analytic=analytic)  # NB compile outside the timer.
    benchmark(_fit, inputs, analytic=analytic)


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("analytic", [True, False], ids=["analytic", "finite"])
def test_one_m_step_fit_at_the_stress_size(
    analytic: bool, benchmark: BenchmarkFixture, cnaster_config: None
) -> None:
    inputs = _fit_inputs(**STRESS)
    _fit(inputs, analytic=analytic)
    benchmark(_fit, inputs, analytic=analytic)
