"""One recursion's cost against cnaster's four, at gate and stress (`K = 7` phased)
sizes (#205).

No speedup is claimed; equivalence is `test_unified_lattice.py`'s.
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.builders import (
    LatticeInputs,
    cnaster_lattice,
    random_lattice,
    unified_lattice,
)
from tests.fixtures import tiers

GATE = {"n_states": 3, "n_obs": 300, "n_spots": 4}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3_000, "n_spots": 50}
"""Where the `S^2` inner loop and the per-site transition tell."""


def _inputs(n_states: int, n_obs: int, n_spots: int, *, phased: bool) -> LatticeInputs:
    return random_lattice(n_states, [n_obs], n_spots, phased=phased, seed=31)


_run_cnaster, _run_unified = cnaster_lattice, unified_lattice


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
@pytest.mark.parametrize("implementation", ["cnaster", "unified"])
def test_the_recursion(
    benchmark: BenchmarkFixture,
    which: str,
    phased: bool,
    implementation: str,
    size: dict[str, int],
) -> None:
    """Both arms, warm (compilation outside the timer, #204), at gate and stress sizes."""
    inputs = _inputs(**size, phased=phased)

    if implementation == "cnaster":
        _run_cnaster(which, inputs, phased=phased)
        benchmark(_run_cnaster, which, inputs, phased=phased)
    else:
        _run_unified(which, inputs, phased=phased)
        benchmark(_run_unified, which, inputs, phased=phased)
