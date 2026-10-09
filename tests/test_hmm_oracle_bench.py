"""Forward, backward and posterior timings, `cnaster` against upstream (#140).

Upstream loops over chains in Python, so the ratio is not a kernel ratio. No ratio is
asserted.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import (
    CnasterChainInputs,
    cnaster_lattice_arguments,
    cnaster_log_emission,
    cnaster_posterior,
    from_negative_binomial_chains,
    upstream_chain_densities,
)
from tests.fixtures import negative_binomial_chains, tiers

GATE = {"n_states": 5, "sequence_length": 200, "n_sequences": 8}
"""1,600 positions over eight chains: the per-pull-request size."""

STRESS = {"n_states": 7, "sequence_length": 3_000, "n_sequences": 8}
"""24,000 positions, the order `run_cnaster` reaches on the dev instance."""


def _cnaster_forward(inputs: CnasterChainInputs, emission: np.ndarray) -> np.ndarray:
    from cnaster.hmm_nophasing import hmm_nophasing

    forward: np.ndarray = hmm_nophasing.forward_lattice(
        *cnaster_lattice_arguments(inputs, emission)
    )
    return forward


_cnaster_both = cnaster_posterior


def _upstream_both(
    densities: list[np.ndarray], initial: np.ndarray, transition: np.ndarray
) -> float:
    from sal.likelihood.forward_backward import forward_backward

    return sum(
        float(forward_backward(density, initial, transition).log_evidence)
        for density in densities
    )


_densities = upstream_chain_densities


def _instance(settings: dict[str, int]) -> Any:
    fixture = negative_binomial_chains(**settings)
    inputs = from_negative_binomial_chains(fixture)
    return (
        fixture,
        inputs,
        cnaster_log_emission(inputs),
        _densities(fixture),
        np.log(np.asarray(fixture.dataset.initial, dtype=float)),
        np.log(np.asarray(fixture.dataset.transition, dtype=float)),
    )


@pytest.fixture(scope="module")
def gate() -> Any:
    return _instance(GATE)


@pytest.fixture(scope="module")
def stress() -> Any:
    return _instance(STRESS)


def _cnaster_arm(instance: Any) -> Callable[[], Any]:
    _, inputs, emission, _, _, _ = instance
    return partial(_cnaster_both, inputs, emission)


def _upstream_arm(instance: Any) -> Callable[[], Any]:
    _, _, _, densities, initial, transition = instance
    return partial(_upstream_both, densities, initial, transition)


@pytest.mark.benchmark
def test_cnaster_forward_only_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """`forward_lattice` over the concatenated batch, one call."""
    _, inputs, emission, _, _, _ = gate
    benchmark(_cnaster_forward, inputs, emission)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers("gate", "stress"))
@pytest.mark.parametrize(
    "arm", [_cnaster_arm, _upstream_arm], ids=["cnaster", "upstream"]
)
def test_forward_backward(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    arm: Callable[[Any], Callable[[], Any]],
    size: str,
) -> None:
    """Both passes and the posterior; upstream looped over chains."""
    benchmark(arm(request.getfixturevalue(size)))
