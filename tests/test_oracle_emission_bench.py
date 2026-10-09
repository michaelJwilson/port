"""Timings of `cnaster`'s NB and beta-binomial emission against upstream's on
`dev_instance` (#128).

Gate: `GATE_OBS` bins; stress: the whole instance. No ratio is asserted.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch
from cnaster.hmm_nophasing import hmm_nophasing
from port.sim.truth import dev_instance, emission_family
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import from_core_inference_truth
from tests.fixtures import tiers

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
    """The same instance as upstream's two-channel family, exposure and trials as the covariate."""
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
    family = emission_family(truth.log_mu, truth.alphas, truth.p_binom, truth.taus)
    return family, observations, covariate


def _cnaster_emission(inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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


def _cnaster_arm(truth: Any) -> Callable[[], Any]:
    return partial(_cnaster_emission, _cnaster_inputs(truth))


def _upstream_arm(truth: Any) -> Callable[[], Any]:
    family, observations, covariate = _upstream_inputs(truth)
    return lambda: family.log_density(observations, covariate)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers("gate_instance", "stress_instance"))
@pytest.mark.parametrize(
    "arm",
    [_cnaster_arm, _upstream_arm],
    ids=["cnaster", "upstream"],
)
def test_emission(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    arm: Callable[[Any], Callable[[], Any]],
    size: str,
) -> None:
    """`cnaster`'s two matched families against upstream's at the same parameters (agreement: #9)."""
    benchmark(arm(request.getfixturevalue(size)))
