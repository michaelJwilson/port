"""Benchmark: `cnaster`'s `icm_sweep_deque` against upstream's ICM and alpha expansion (#140, #8).

Gate 400 nodes, stress 3,600; no ratio asserted. Agreement is `test_icm_oracle.py`'s.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import cnaster_icm_labelling
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
    from sal.search.icm import iterated_conditional_modes

    return iterated_conditional_modes(graph, fixture.field, np.random.default_rng(0))


def _upstream_expansion(
    fixture: PottsLabels, graph: "PottsGraph", _: np.ndarray
) -> Any:
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion

    # NB PYTHON was the default before e0aeb19 made it RUST (#410)
    return alpha_expansion(graph, fixture.field, backend=Backend.PYTHON)


@pytest.fixture(scope="module")
def gate() -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    fixture = potts_labels(shape=GATE_SHAPE, n_clones=3)
    return fixture, scaled_graph(fixture), np.zeros(fixture.n_nodes, dtype=np.int64)


@pytest.fixture(scope="module")
def stress() -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    fixture = potts_labels(shape=STRESS_SHAPE, n_clones=3)
    return fixture, scaled_graph(fixture), np.zeros(fixture.n_nodes, dtype=np.int64)


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
