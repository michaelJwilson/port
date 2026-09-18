"""What the label solvers cost, `cnaster` against upstream's two (#140).

Three solvers on one instance and one objective, so the numbers are
comparable: `cnaster`'s `icm_sweep_deque`, upstream's
`iterated_conditional_modes` -- the same move set, which is what makes it a
fair baseline -- and `alpha_expansion`, a stronger one.

`tests/test_icm_oracle.py` establishes what each buys: the sweep is
single-site optimal, and expansion lowers its energy by 4.62 from the same
start. These say what that costs, so #8's question -- whether `cnaster`
should use a stronger solver -- has both halves.

No ratio is asserted. Upstream's ICM defaults to its `numba` backend and
`cnaster`'s is `njit` throughout, so a first call pays compilation; the
benchmark's own warmup absorbs it, which is why these are timed here rather
than inferred from a test's wall clock.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import cnaster_icm_labelling
from tests.fixtures import PottsLabels, potts_labels

if TYPE_CHECKING:
    from snakes_and_ladders.sim.graph import PottsGraph

GATE_SHAPE = (20, 20)
"""400 nodes, three clones: the per-pull-request size."""

STRESS_SHAPE = (60, 60)
"""3,600 nodes, an order of magnitude up, where a move set's cost shows."""


def _graph(fixture: PottsLabels) -> "PottsGraph":
    from tests.fixtures import _scaled_graph

    return _scaled_graph(fixture)


def _cnaster_sweep(fixture: PottsLabels, start: np.ndarray) -> Any:
    return cnaster_icm_labelling(fixture, start)


def _upstream_icm(fixture: PottsLabels, graph: "PottsGraph") -> Any:
    from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes

    return iterated_conditional_modes(
        graph, fixture.field, fixture.n_clones, np.random.default_rng(0)
    )


def _upstream_expansion(fixture: PottsLabels, graph: "PottsGraph") -> Any:
    from snakes_and_ladders.search.alpha_expansion import alpha_expansion

    return alpha_expansion(graph, fixture.field, fixture.n_clones)


@pytest.fixture(scope="module")
def gate() -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    fixture = potts_labels(shape=GATE_SHAPE, n_clones=3)
    return fixture, _graph(fixture), np.zeros(fixture.n_nodes, dtype=np.int64)


@pytest.fixture(scope="module")
def stress() -> tuple[PottsLabels, "PottsGraph", np.ndarray]:
    fixture = potts_labels(shape=STRESS_SHAPE, n_clones=3)
    return fixture, _graph(fixture), np.zeros(fixture.n_nodes, dtype=np.int64)


@pytest.mark.cnaster
@pytest.mark.benchmark
@pytest.mark.subject
def test_cnaster_icm_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """`icm_sweep_deque` over 400 nodes."""
    fixture, _, start = gate
    benchmark(_cnaster_sweep, fixture, start)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.upstream
def test_upstream_icm_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """The same move set, upstream's implementation.

    `upstream` rather than `upstream_oracle`: this times the baseline, it does
    not consult it. The agreement claims are in `test_icm_oracle.py`.
    """
    fixture, graph, _ = gate
    benchmark(_upstream_icm, fixture, graph)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.upstream
def test_upstream_expansion_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """The stronger move set, which is expected to cost more and find more."""
    fixture, graph, _ = gate
    benchmark(_upstream_expansion, fixture, graph)


@pytest.mark.cnaster
@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.subject
def test_cnaster_icm_stress(benchmark: BenchmarkFixture, stress: Any) -> None:
    """3,600 nodes, nine times the gate."""
    fixture, _, start = stress
    benchmark(_cnaster_sweep, fixture, start)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.upstream
def test_upstream_icm_stress(benchmark: BenchmarkFixture, stress: Any) -> None:
    fixture, graph, _ = stress
    benchmark(_upstream_icm, fixture, graph)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.upstream
def test_upstream_expansion_stress(benchmark: BenchmarkFixture, stress: Any) -> None:
    fixture, graph, _ = stress
    benchmark(_upstream_expansion, fixture, graph)
