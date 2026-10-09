"""Label solvers on sal's planted noisy tilings, certified by TRW-S's bound (#410, sal #1074/#1061).

Referees: the planted tile states and TRW-S's lower bound on the Potts energy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from port.extensions.label_solver import sweep_for
from port.patch.icm.alpha_expansion import potts_graph_from
from port.patch.icm.interface import CsrGraph
from port.sandbox.extensions.label_solvers import SWEEPS
from sal.search.trws import trws
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energy, smoothed_noise, tile_partition, tiling_field


def _instance(side: int, n_tiles: int, n_states: int, seed: int) -> Any:
    lattice = lattice_graph((side, side), BoundaryCondition.OPEN, 1.0)
    rng = np.random.default_rng(seed)
    tiles = tile_partition(lattice, n_tiles, rng)
    states = rng.permutation(n_states)[:n_tiles] % n_states
    if n_tiles > n_states:
        states = rng.integers(0, n_states, n_tiles)
    strengths = rng.uniform(1.0, 2.0, n_tiles)
    field = tiling_field(tiles, states, strengths, n_states) + 0.5 * smoothed_noise(
        lattice, n_states, 2, np.random.default_rng(seed + 1)
    )

    edges = lattice.edge_index
    n = lattice.n_nodes
    adjacency = sp.coo_matrix(
        (
            np.ones(2 * len(edges)),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(n, n),
    ).tocsr()

    return field, CsrGraph.from_matrix(adjacency), states[tiles]


def _solve(name: str, field: np.ndarray, graph: Any, beta: float) -> np.ndarray:
    labelling = np.zeros(field.shape[0], dtype=np.int64)
    # NB `cnaster`'s sweep draws its queue order from the legacy state (#45)
    np.random.seed(0)  # noqa: NPY002
    sweep = SWEEPS[name] if name in SWEEPS else sweep_for(name)  # type: ignore[arg-type]
    sweep(field, graph, labelling, beta, min_clone_spots=1)
    return labelling


def _gap_and_agreement(
    name: str, side: int, n_tiles: int, n_states: int
) -> tuple[float, float]:
    beta = 0.6
    field, graph, planted = _instance(side, n_tiles, n_states, seed=7)
    potts = potts_graph_from(graph, beta)
    bound = trws(potts, field).bound
    labelling = _solve(name, field, graph, beta)

    return float(energy(potts, field, labelling) - bound), float(
        np.mean(labelling == planted)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["alpha-rust", "alpha-rust-icm"])
def test_alpha_expansion_reaches_the_certified_ground_state(name: str) -> None:
    """Alpha expansion's energy equals TRW-S's bound, to 1e-6; >= 0.99 agree with planted."""
    gap, agreement = _gap_and_agreement(name, 40, 6, 4)

    assert abs(gap) < 1e-6, f"{name} is {gap:.3f} nats above the bound"
    assert agreement >= 0.99, f"{name} agrees with the planted at {agreement:.4f}"


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["icm", "icm-numba"])
def test_single_site_descent_stays_within_its_measured_gap(name: str) -> None:
    """ICM lies above the bound by under 10 nats; >= 0.98 agree with planted."""
    gap, agreement = _gap_and_agreement(name, 40, 6, 4)

    assert 0.0 <= gap < 10.0, f"{name} gap {gap:.3f}"
    assert agreement >= 0.98, f"{name} agrees with the planted at {agreement:.4f}"


@pytest.mark.oracle
@pytest.mark.release
def test_alpha_expansion_is_certified_at_the_stress_size() -> None:
    """At 10,000 sites alpha expansion is at the bound, to 1e-6; >= 0.99 agree with planted."""
    gap, agreement = _gap_and_agreement("alpha-rust", 100, 12, 6)

    assert abs(gap) < 1e-6, f"alpha-rust is {gap:.3f} nats above the bound"
    assert agreement >= 0.99
