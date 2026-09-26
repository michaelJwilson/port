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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import cnaster_icm_labelling
from tests.fixtures import PottsLabels, potts_labels, tiers

if TYPE_CHECKING:
    from snakes_and_ladders.sim.graph import PottsGraph

GATE_SHAPE = (20, 20)
"""400 nodes, three clones: the per-pull-request size."""

STRESS_SHAPE = (60, 60)
"""3,600 nodes, an order of magnitude up, where a move set's cost shows."""


def _graph(fixture: PottsLabels) -> "PottsGraph":
    from tests.fixtures import _scaled_graph

    return _scaled_graph(fixture)


def _cnaster_sweep(fixture: PottsLabels, _: "PottsGraph", start: np.ndarray) -> Any:
    return cnaster_icm_labelling(fixture, start)


def _upstream_icm(fixture: PottsLabels, graph: "PottsGraph", _: np.ndarray) -> Any:
    from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes

    return iterated_conditional_modes(
        graph, fixture.field, fixture.n_clones, np.random.default_rng(0)
    )


def _upstream_expansion(
    fixture: PottsLabels, graph: "PottsGraph", _: np.ndarray
) -> Any:
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
    """`icm_sweep_deque`, upstream's ICM on the same move set, and its expansion.

    The expansion is the stronger move set, expected to cost more and find
    more. `upstream` rather than `upstream_oracle`: this times the baseline,
    it does not consult it. The agreement claims are in `test_icm_oracle.py`.
    """
    benchmark(arm, *request.getfixturevalue(size))
