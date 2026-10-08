"""`oxiport`'s lattices against `cnaster`'s `@njit` ones, warm (#318, #204).

Gate rows decide nothing; stress rows (`release`): K = 10, 10 x 1,000 bins, 20 spots.
"""

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

GATE = {"n_states": 5, "n_contigs": 10, "per_contig": 100, "n_spots": 4}
"""Gate size; decides no ratio."""

STRESS = {"n_states": 10, "n_contigs": 10, "per_contig": 1_000, "n_spots": 20}
"""Stress size, where the ratio is read."""


def _inputs(
    n_states: int, n_contigs: int, per_contig: int, n_spots: int, *, phased: bool
) -> tuple[Any, ...]:
    generator = np.random.default_rng(31)
    n_obs = n_contigs * per_contig
    rows = 2 * n_states if phased else n_states

    return (
        np.full(n_contigs, per_contig, dtype=np.int64),
        np.log(generator.dirichlet(np.ones(n_states), n_states)),
        np.log(generator.dirichlet(np.ones(n_states))),
        generator.normal(-5.0, 3.0, (rows, n_obs, n_spots)),
        np.log(generator.uniform(1e-4, 0.3, n_obs)),
    )


def _recursion(which: str, *, phased: bool, implementation: str) -> Any:
    if implementation == "cnaster":
        from cnaster.hmm_nophasing import hmm_nophasing
        from cnaster.hmm_phased import hmm_phased

        return getattr(hmm_phased if phased else hmm_nophasing, which)

    from port.patch import lattice

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
