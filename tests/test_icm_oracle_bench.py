"""Benchmark: `cnaster`'s `icm_sweep_deque` against upstream's ICM and alpha expansion (#140, #8).

Gate 400 nodes, stress 3,600; no ratio asserted. Agreement is `test_icm_oracle.py`'s.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import cnaster_icm_labelling, upstream_expansion, upstream_icm
from tests.fixtures import PottsLabels, potts_labels, scaled_graph, tiers

if TYPE_CHECKING:
    from sal.sim.graph import PottsGraph

GATE_SHAPE = (20, 20)
"""400 nodes, three clones: the per-pull-request size."""

STRESS_SHAPE = (60, 60)
"""3,600 nodes, an order of magnitude up."""


def _cnaster_sweep(fixture: PottsLabels, _: "PottsGraph", start: np.ndarray) -> Any:
    return cnaster_icm_labelling(fixture, start)


def _upstream_icm(fixture: PottsLabels, graph: "PottsGraph", _: np.ndarray) -> Any:
    return upstream_icm(fixture, graph)


def _upstream_expansion(
    fixture: PottsLabels, graph: "PottsGraph", _: np.ndarray
) -> Any:
    return upstream_expansion(fixture, graph)


def _sized(shape: tuple[int, int]) -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    fixture = potts_labels(shape=shape, n_clones=3)
    return fixture, scaled_graph(fixture), np.zeros(fixture.n_nodes, dtype=np.int64)


@pytest.fixture(scope="module")
def gate() -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    return _sized(GATE_SHAPE)


@pytest.fixture(scope="module")
def stress() -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    return _sized(STRESS_SHAPE)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers("gate", "stress"))
@pytest.mark.parametrize(
    "arm",
    [_cnaster_sweep, _upstream_icm, _upstream_expansion],
    ids=["cnaster-icm", "upstream-icm", "upstream-expansion"],
)
def test_labelling(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    arm: Callable[..., Any],
    size: str,
) -> None:
    """Time the three solvers on one instance and objective, warmed by the benchmark."""
    benchmark(arm, *request.getfixturevalue(size))
