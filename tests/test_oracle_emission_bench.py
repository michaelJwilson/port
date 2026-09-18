"""Baselines for the two **matched** emission families, on the dev fixture (#128).

`emissions.py` is the largest module on the oracle surface and most of it is
unreachable: `cnaster` implements a negative binomial and a beta-binomial and
nothing else, so those two families are the only ones a rung can referee
against, and the only ones it is meaningful to time. That is the one-way rule
from `tests/test_oracle_correspondence.py` showing up in a benchmark table --
a family with no counterpart has no baseline because it has no comparison.

Measured on `dev_instance`, which is what `CLAUDE.md`'s *develop against*
instance is for: the key instance does not fit (#90) and the critical instance
is sized to gate in seconds, not to measure. The gate pair reduces its bin
axis and keeps everything else; the stress pair is the whole of it.

These assert no ratio. `CLAUDE.md` puts the bar for acting on one at a stress
size and on a claim someone is making; what these are is the number #128's
rungs report against, so that "the fit-level rung costs X" is a measurement
rather than an impression.
"""

from typing import Any

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import from_core_inference_truth
from tests.fixtures import _emission_families, dev_instance

GATE_OBS = 200
"""The dev instance's bin axis, reduced. Its `M`, `K` and `S` are untouched."""


def _cnaster_inputs(truth: Any) -> dict[str, Any]:
    """`compute_emission_probability_nb_betabinom`'s arguments, as `hmrf` passes them."""
    kwargs = from_core_inference_truth(truth).as_kwargs()
    return {
        "X": kwargs["single_X"],
        "base_nb_mean": kwargs["single_base_nb_mean"],
        "log_mu": truth.log_mu.reshape(-1, 1),
        "alphas": truth.alphas.reshape(-1, 1),
        "total_bb_RD": kwargs["single_total_bb_RD"],
        "p_binom": truth.p_binom.reshape(-1, 1),
        "taus": truth.taus.reshape(-1, 1),
    }


def _upstream_inputs(truth: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """The same instance as upstream's two-channel family and its observations.

    The exposure and the trial counts ride as the covariate, which is where
    upstream puts a per-observation quantity the family does not carry.
    """
    kwargs = from_core_inference_truth(truth).as_kwargs()
    single_X = np.asarray(kwargs["single_X"])
    exposure = np.asarray(kwargs["single_base_nb_mean"])
    trials = np.asarray(kwargs["single_total_bb_RD"])
    observations = torch.as_tensor(
        np.stack([single_X[:, 0, :], single_X[:, 1, :]], axis=-1), dtype=torch.float64
    )
    covariate = torch.as_tensor(
        np.stack([exposure, trials], axis=-1), dtype=torch.float64
    )
    family = _emission_families(truth.log_mu, truth.alphas, truth.p_binom, truth.taus)
    return family, observations, covariate


def _cnaster_emission(inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from cnaster.hmm_nophasing import hmm_nophasing

    scored: tuple[np.ndarray, np.ndarray] = (
        hmm_nophasing.compute_emission_probability_nb_betabinom(
            inputs["X"],
            inputs["base_nb_mean"],
            inputs["log_mu"],
            inputs["alphas"],
            inputs["total_bb_RD"],
            inputs["p_binom"],
            inputs["taus"],
        )
    )
    return scored


@pytest.fixture(scope="module")
def gate_instance() -> Any:
    """`dev_instance` over `GATE_OBS` bins: `M = 4`, `K = 10`, `S = 1,600`."""
    return dev_instance(n_obs=GATE_OBS)


@pytest.fixture(scope="module")
def stress_instance() -> Any:
    """The dev instance whole, `G = 1,000`."""
    return dev_instance()


@pytest.mark.benchmark
@pytest.mark.cnaster
def test_cnaster_emission_gate(benchmark: BenchmarkFixture, gate_instance: Any) -> None:
    """`cnaster`'s two matched families over the reduced dev instance.

    Realized **376 ms** minimum against upstream's 128 ms, a ratio of 2.9.
    """
    inputs = _cnaster_inputs(gate_instance)
    benchmark(_cnaster_emission, inputs)


@pytest.mark.benchmark
@pytest.mark.upstream
def test_upstream_emission_gate(
    benchmark: BenchmarkFixture, gate_instance: Any
) -> None:
    """Upstream's, on the same instance and the same parameters.

    `upstream` rather than `upstream_oracle`: this times the referee, it does
    not consult it. The agreement claim is #9's, at 2.5e-11.
    """
    family, observations, covariate = _upstream_inputs(gate_instance)
    benchmark(family.log_density, observations, covariate)


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.cnaster
def test_cnaster_emission_stress(
    benchmark: BenchmarkFixture, stress_instance: Any
) -> None:
    """The whole dev instance: five times the bins of the gate.

    Realized **1,977 ms** against upstream's 930 ms, a ratio of 2.1. The gap
    narrows with size, which is the encoder's constant factor being amortized.
    """
    inputs = _cnaster_inputs(stress_instance)
    benchmark(_cnaster_emission, inputs)


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.upstream
def test_upstream_emission_stress(
    benchmark: BenchmarkFixture, stress_instance: Any
) -> None:
    """Upstream's, at the size a ratio may be read at."""
    family, observations, covariate = _upstream_inputs(stress_instance)
    benchmark(family.log_density, observations, covariate)
