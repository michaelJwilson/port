"""One buffered emission entry point against `cnaster`'s two, at gate and stress sizes (#205, #90).

Stress (`release`): K = 7, G = 3,000, S = 2,000; bitwise evidence in
`test_buffered_emission.py`.
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.builders import (
    EmissionInputs,
    buffered_emission,
    cnaster_emission_pair,
    emission_inputs,
)
from tests.fixtures import tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300}
"""Gate size; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3_000, "n_spots": 2_000}
"""Stress size: 1.34 GB allocated per phased call in `cnaster`'s arm."""


def _inputs(n_states: int, n_obs: int, n_spots: int) -> EmissionInputs:
    """Built outside the timer (#278)."""
    return emission_inputs(n_states, n_obs, n_spots, seed=29)


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
        columns = inputs.columns()
        cnaster_emission_pair(columns)
        benchmark(cnaster_emission_pair, columns)
    else:
        buffers = emission_buffers(
            size["n_states"], size["n_obs"], size["n_spots"], phased=False
        )
        buffered_emission(inputs, buffers, False)
        benchmark(buffered_emission, inputs, buffers, False)
