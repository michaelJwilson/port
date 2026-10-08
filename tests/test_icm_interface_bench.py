"""The ICM interface reduction's cost, fold included, at 400 and 20,000 spots (#59 item
5).

No speedup is claimed; each round seeds the global RNG the sweep draws from.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from port.patch.icm.interface import CsrGraph, fold_unary, icm_sweep
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.sparse import csr_matrix

from tests.fixtures import tiers

GATE_SPOTS = 400
STRESS_SPOTS = 20_000
N_CLONES = 4
N_SAMPLES = 3
NEIGHBOURS = 6
BETA = 0.75
SEED = 6_803


@dataclass(frozen=True)
class Problem:
    """One labelling problem, so both columns are benchmarked on the same one."""

    field: np.ndarray
    graph: csr_matrix
    assignment: np.ndarray
    sample_ids: np.ndarray
    weights: np.ndarray


def _problem(n_spots: int) -> Problem:
    rng = np.random.default_rng(SEED)

    field = rng.normal(-50.0, 5.0, (n_spots, N_CLONES))

    rows = np.repeat(np.arange(n_spots), NEIGHBOURS)
    cols = rng.integers(0, n_spots, NEIGHBOURS * n_spots)
    graph = csr_matrix(
        (rng.uniform(0.5, 2.0, NEIGHBOURS * n_spots), (rows, cols)),
        shape=(n_spots, n_spots),
    )

    return Problem(
        field=field,
        graph=graph,
        assignment=rng.integers(0, N_CLONES, n_spots),
        sample_ids=rng.integers(0, N_SAMPLES, n_spots),
        weights=rng.normal(0.0, 2.0, (N_CLONES, N_SAMPLES)),
    )


def _cnaster_call(problem: Problem) -> None:
    """Fifteen arguments, with the weights indexed inside the inner loop."""
    from cnaster.icm import icm_sweep_deque

    # NPY002: cnaster's sweep shuffles with the legacy global RNG.
    np.random.seed(SEED)  # noqa: NPY002
    icm_sweep_deque(
        single_llf=problem.field,
        adj_indptr=problem.graph.indptr,
        adj_indices=problem.graph.indices,
        adj_weights=problem.graph.data,
        new_assignment=problem.assignment.copy(),
        spatial_weight=BETA,
        posterior=None,
        log_persample_weights=problem.weights,
        sample_ids=problem.sample_ids,
        min_clone_spots=0,
    )


def _patched_call(problem: Problem) -> None:
    """Eight parameters, with the fold charged to this column."""
    # NPY002: cnaster's sweep shuffles with the legacy global RNG.
    np.random.seed(SEED)  # noqa: NPY002
    icm_sweep(
        fold_unary(problem.field, problem.weights, problem.sample_ids),
        CsrGraph.from_matrix(problem.graph),
        problem.assignment.copy(),
        BETA,
        min_clone_spots=0,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_spots", tiers(GATE_SPOTS, STRESS_SPOTS))
@pytest.mark.parametrize(
    "arm", [_cnaster_call, _patched_call], ids=["cnaster", "patched"]
)
def test_sweep(
    benchmark: BenchmarkFixture, arm: Callable[[Problem], None], n_spots: int
) -> None:
    """One sweep over 4 clones; 20,000 spots is the scale a slice runs at."""
    benchmark(arm, _problem(n_spots))
