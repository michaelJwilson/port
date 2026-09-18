"""What the interface reduction costs (issue #59 item 5).

Recorded so the simplification is not mistaken for a free lunch, and not
mistaken for a regression either. Folding `log_persample_weights` into the
field replaces one indexed add per clone per spot **visit** with one
vectorized add per clone per spot **sweep**, so the trade depends on how many
times the solver revisits a spot.

Both columns include the fold, which is the honest comparison: the caller
pays it.

| spots | clones | `cnaster`, 15 arguments | reduced, 8 | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 4 | 4.74 ms | 3.97 ms | 1.19 |
| 20,000 | 4 | 234 ms | 169 ms | 1.38 |

**1.38x at the stress size does not clear `CLAUDE.md`'s 2x bar, so no
speedup is claimed.** This lands as a simplification, on the bitwise
equivalence in `test_icm_interface.py`; the ratio is reported because
measuring it was the only way to establish that the fold does not cost
anything, and it happens to be positive.

The sweep draws its queue order from the global `numpy` RNG, so each round
seeds it -- otherwise the benchmark measures a different number of epochs on
every round and reports the variance as noise.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from port.patch.icm_interface import CsrGraph, fold_unary, icm_sweep
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.sparse import csr_matrix

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

    # NPY002 is the finding, not the violation: `cnaster`'s sweep shuffles
    # its queue with the legacy global RNG, so a `Generator` cannot reach it.
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
    # NPY002 is the finding, not the violation: `cnaster`'s sweep shuffles
    # its queue with the legacy global RNG, so a `Generator` cannot reach it.
    np.random.seed(SEED)  # noqa: NPY002
    icm_sweep(
        fold_unary(problem.field, problem.weights, problem.sample_ids),
        CsrGraph.from_matrix(problem.graph),
        problem.assignment.copy(),
        BETA,
        min_clone_spots=0,
    )


@pytest.mark.benchmark
def test_cnaster_sweep_gate(benchmark: BenchmarkFixture) -> None:
    """400 spots, 4 clones."""
    benchmark(_cnaster_call, _problem(GATE_SPOTS))


@pytest.mark.benchmark
def test_patched_sweep_gate(benchmark: BenchmarkFixture) -> None:
    """The same, through the reduced interface."""
    benchmark(_patched_call, _problem(GATE_SPOTS))


@pytest.mark.benchmark
@pytest.mark.release
def test_cnaster_sweep_stress(benchmark: BenchmarkFixture) -> None:
    """20,000 spots, which is the scale a slice runs at."""
    benchmark(_cnaster_call, _problem(STRESS_SPOTS))


@pytest.mark.benchmark
@pytest.mark.release
def test_patched_sweep_stress(benchmark: BenchmarkFixture) -> None:
    """And the reduced interface at that scale."""
    benchmark(_patched_call, _problem(STRESS_SPOTS))
