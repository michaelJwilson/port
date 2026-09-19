"""What one emission entry point costs against `cnaster`'s two (#205).

**The claim is the allocation, not the ratio.** `cnaster` builds
`(n_states, n_obs, n_spots)` per channel on every call and the phased class
doubles the state axis; `port.patch.emission.emission_into` writes into
buffers the caller keeps. At the stress size below that is 1.34 GB the
phased entry point allocates per call and this one does not, twice per outer
iteration (#90).

So these rows are here to catch the case where writing into a buffer cost
something, not to argue a speedup. `tests/test_buffered_emission.py` carries
the bitwise evidence that makes it a simplification.

The stress pair carries `release`: the phased arm at `K = 7`, `G = 3,000`,
`S = 2,000` allocates 1.34 GB in `cnaster`'s arm alone.
"""

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

Inputs = dict[str, Any]
"""The arrays one arm is handed, named so both arms take the same thing."""

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3_000, "n_spots": 2_000}
"""336 MB per channel unphased, 1.34 GB across both channels phased."""


def _inputs(n_states: int, n_obs: int, n_spots: int, *, per_spot: bool) -> Inputs:
    generator = np.random.default_rng(29)

    exposure = generator.integers(20, 45, (n_obs, n_spots)).astype(np.float64)
    trials = generator.integers(5, 25, (n_obs, n_spots)).astype(np.float64)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = generator.poisson(exposure)
    single_X[:, 1, :] = generator.binomial(trials.astype(int), 0.42)

    def column(values: np.ndarray) -> np.ndarray:
        stacked = np.asarray(values)[:, None]
        return np.tile(stacked, (1, n_spots)) if per_spot else stacked

    return {
        "single_X": single_X,
        "base_nb_mean": exposure,
        "total_bb_RD": trials,
        "log_mu": column(np.linspace(-0.35, 0.35, n_states)),
        "alphas": column(np.linspace(0.12, 0.55, n_states)),
        "p_binom": column(np.linspace(0.22, 0.78, n_states)),
        "taus": column(np.linspace(8.0, 28.0, n_states)),
    }


def _run_cnaster(inputs: Inputs) -> tuple[np.ndarray, np.ndarray]:
    from cnaster.hmm_nophasing import hmm_nophasing

    scored: tuple[np.ndarray, np.ndarray]
    scored = hmm_nophasing.compute_emission_probability_nb_betabinom(
        inputs["single_X"],
        inputs["base_nb_mean"],
        inputs["log_mu"],
        inputs["alphas"],
        inputs["total_bb_RD"],
        inputs["p_binom"],
        inputs["taus"],
    )
    return scored


def _run_buffered(inputs: Inputs, buffers: tuple[np.ndarray, np.ndarray]) -> None:
    from port.patch.emission import emission_into

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


def _bench(
    benchmark: BenchmarkFixture, size: dict[str, int], implementation: str
) -> None:
    from port.patch.emission import emission_buffers

    inputs = _inputs(**size, per_spot=False)

    if implementation == "cnaster":
        _run_cnaster(inputs)
        benchmark(_run_cnaster, inputs)
    else:
        buffers = emission_buffers(
            size["n_states"], size["n_obs"], size["n_spots"], phased=False
        )
        _run_buffered(inputs, buffers)
        benchmark(_run_buffered, inputs, buffers)


@pytest.mark.benchmark
@pytest.mark.parametrize("implementation", ["cnaster", "buffered"])
def test_the_gate_emission(benchmark: BenchmarkFixture, implementation: str) -> None:
    """A baseline at a gate size, which argues nothing either way."""
    _bench(benchmark, GATE, implementation)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("implementation", ["cnaster", "buffered"])
def test_the_stress_emission(benchmark: BenchmarkFixture, implementation: str) -> None:
    """The size the allocation tells at, warm.

    Both arms are called once outside the timer: both are `numba` kernels and
    a first call on a cold cache is compilation rather than work (#204).
    """
    _bench(benchmark, STRESS, implementation)
