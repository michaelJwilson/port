"""`oxiport`'s lattices against `cnaster`'s `@njit` ones, warm (#318, #204).

Gate rows decide nothing; stress rows (`release`): K = 10, 10 x 1,000 bins, 20 spots.
"""

from typing import Any

import pytest
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmm_phased import hmm_phased
from port.patch import lattice
from pytest_benchmark.fixture import BenchmarkFixture

from tests.builders import random_lattice

GATE = {"n_states": 5, "n_contigs": 10, "per_contig": 100, "n_spots": 4}
"""Gate size; decides no ratio."""

STRESS = {"n_states": 10, "n_contigs": 10, "per_contig": 1_000, "n_spots": 20}
"""Stress size, where the ratio is read."""


def _inputs(
    n_states: int, n_contigs: int, per_contig: int, n_spots: int, *, phased: bool
) -> tuple[Any, ...]:
    return random_lattice(
        n_states,
        [per_contig] * n_contigs,
        n_spots,
        phased=phased,
        seed=31,
        dirichlet=True,
    ).arguments


def _recursion(which: str, *, phased: bool, implementation: str) -> Any:
    if implementation == "cnaster":
        return getattr(hmm_phased if phased else hmm_nophasing, which)

    name = which.replace("_lattice", "_lattice_phased") if phased else which
    return getattr(lattice, f"{name}_rust")


def _bench(
    benchmark: BenchmarkFixture,
    size: dict[str, int],
    which: str,
    phased: bool,
    implementation: str,
) -> None:
    arguments = _inputs(**size, phased=phased)
    recursion = _recursion(which, phased=phased, implementation=implementation)

    recursion(*arguments)
    benchmark(recursion, *arguments)


@pytest.mark.benchmark
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
@pytest.mark.parametrize("implementation", ["cnaster", "rust"])
def test_the_gate_lattice(
    benchmark: BenchmarkFixture, which: str, phased: bool, implementation: str
) -> None:
    """Gate-size baseline."""
    _bench(benchmark, GATE, which, phased, implementation)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
@pytest.mark.parametrize("implementation", ["cnaster", "rust"])
def test_the_stress_lattice(
    benchmark: BenchmarkFixture, which: str, phased: bool, implementation: str
) -> None:
    """Stress-size ratio, warm."""
    _bench(benchmark, STRESS, which, phased, implementation)
