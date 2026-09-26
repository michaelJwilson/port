"""The label solvers against planted noisy tilings, certified by TRW-S (#410, sal #1074/#1061).

sal's tiling fixture: a lattice split into Voronoi tiles, each rewarding one
planted state, with spatially smoothed noise over the field. The planted
states are the truth a labelling is read against, and TRW-S's lower bound
certifies how far a labelling's Potts energy can be from the ground state.
Neither referee is a solver under test.

Realized at the gate instance (1,600 sites, 6 tiles, 4 states, coupling
0.6, noise 0.5), energy minus the bound, and agreement with the planted:

| solver | gap (nats) | agreement |
| --- | --- | --- |
| `alpha-rust` | 0.0 | 0.9950 |
| `alpha-rust-icm` | 0.0 | 0.9950 |
| `icm` (`cnaster`'s) | 5.3 | 0.9850 |
| `icm-numba` | 6.3 | 0.9831 |

At 10,000 sites, 12 tiles and 6 states, alpha expansion's gap is still 0.0
(agreement 0.9975) and the ICMs' 60.9 to 78.6: the stress tier below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp


def _instance(side: int, n_tiles: int, n_states: int, seed: int) -> Any:
    from port.patch.icm.interface import CsrGraph
    from sal.sim.graph import BoundaryCondition, lattice_graph
    from sal.sim.potts import smoothed_noise, tile_partition, tiling_field

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
    from port.extensions.label_solver import sweep_for

    labelling = np.zeros(field.shape[0], dtype=np.int64)
    # NB `cnaster`'s sweep draws its queue order from the legacy state (#45).
    np.random.seed(0)  # noqa: NPY002
    sweep_for(name)(field, graph, labelling, beta, min_clone_spots=1)  # type: ignore[arg-type]
    return labelling


def _gap_and_agreement(
    name: str, side: int, n_tiles: int, n_states: int
) -> tuple[float, float]:
    from port.patch.icm.alpha_expansion import potts_graph_from
    from sal.search.trws import trws
    from sal.sim.potts import energy

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
    """Energy equal to TRW-S's lower bound, so no labelling is lower; 0.995 planted."""
    gap, agreement = _gap_and_agreement(name, 40, 6, 4)

    assert abs(gap) < 1e-6, f"{name} is {gap:.3f} nats above the bound"
    assert agreement >= 0.99, f"{name} agrees with the planted at {agreement:.4f}"


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["icm", "icm-numba"])
def test_single_site_descent_stays_within_its_measured_gap(name: str) -> None:
    """Above the bound, as a local search must be, and by the gap measured here."""
    gap, agreement = _gap_and_agreement(name, 40, 6, 4)

    assert 0.0 <= gap < 10.0, f"{name} gap {gap:.3f}"
    assert agreement >= 0.98, f"{name} agrees with the planted at {agreement:.4f}"


@pytest.mark.oracle
@pytest.mark.release
def test_alpha_expansion_is_certified_at_the_stress_size() -> None:
    """10,000 sites, 12 tiles, 6 states: still the ground state, 0.9975 planted."""
    gap, agreement = _gap_and_agreement("alpha-rust", 100, 12, 6)

    assert abs(gap) < 1e-6, f"alpha-rust is {gap:.3f} nats above the bound"
    assert agreement >= 0.99
