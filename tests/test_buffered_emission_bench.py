"""One buffered emission entry point against `cnaster`'s two, at gate and stress sizes (#205, #90).

Stress (`release`): K = 7, G = 3,000, S = 2,000; bitwise evidence in
`test_buffered_emission.py`.
"""

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers

Inputs = dict[str, Any]
"""The arrays one arm is handed, shared by both arms."""

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300}
"""Gate size; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3_000, "n_spots": 2_000}
"""Stress size: 1.34 GB allocated per phased call in `cnaster`'s arm."""


def _inputs(n_states: int, n_obs: int, n_spots: int) -> Inputs:
    """Both parameter shapes, built outside the timer (#278)."""
    generator = np.random.default_rng(29)

    exposure = generator.integers(20, 45, (n_obs, n_spots)).astype(np.float64)
    trials = generator.integers(5, 25, (n_obs, n_spots)).astype(np.float64)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = generator.poisson(exposure)
    single_X[:, 1, :] = generator.binomial(trials.astype(int), 0.42)

    parameters = {
        "log_mu": np.linspace(-0.35, 0.35, n_states),
        "alphas": np.linspace(0.12, 0.55, n_states),
        "p_binom": np.linspace(0.22, 0.78, n_states),
        "taus": np.linspace(8.0, 28.0, n_states),
    }

    return {
        "single_X": single_X,
        "base_nb_mean": exposure,
        "total_bb_RD": trials,
        **parameters,
        **{f"{name}_column": values[:, None] for name, values in parameters.items()},
    }


def _run_cnaster(inputs: Inputs) -> tuple[np.ndarray, np.ndarray]:
    from cnaster.hmm_nophasing import hmm_nophasing

    scored: tuple[np.ndarray, np.ndarray]
    scored = hmm_nophasing.compute_emission_probability_nb_betabinom(
        inputs["single_X"],
        inputs["base_nb_mean"],
        inputs["log_mu_column"],
        inputs["alphas_column"],
        inputs["total_bb_RD"],
        inputs["p_binom_column"],
        inputs["taus_column"],
    )
    return scored


def _run_buffered(inputs: Inputs, buffers: tuple[np.ndarray, np.ndarray]) -> None:
    from port.sandbox.patch.emission import emission_into

    emission_into(
        inputs["single_X"][:, 0, :],
        inputs["base_nb_mean"],
        inputs["single_X"][:, 1, :],
        inputs["total_bb_RD"],
        inputs["log_mu"],
        inputs["alphas"],
        inputs["p_binom"],
        inputs["taus"],
        buffers[0],
        buffers[1],
        False,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("implementation", ["cnaster", "buffered"])
def test_the_emission(
    benchmark: BenchmarkFixture, implementation: str, size: dict[str, int]
) -> None:
    """Both arms, warm, at gate and stress sizes (#204)."""
    from port.sandbox.patch.emission import emission_buffers

    inputs = _inputs(**size)

    if implementation == "cnaster":
        _run_cnaster(inputs)
        benchmark(_run_cnaster, inputs)
    else:
        buffers = emission_buffers(
            size["n_states"], size["n_obs"], size["n_spots"], phased=False
        )
        _run_buffered(inputs, buffers)
        benchmark(_run_buffered, inputs, buffers)
