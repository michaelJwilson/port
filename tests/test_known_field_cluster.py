"""#559: field-weighted Swendsen-Wang and Wolff, against their kernels enumerated exactly.

On a 4-site graph at q = 3 every transition is enumerable: each subset of
like edges bonded, and each label of each cluster. The exact kernel is checked
to keep `sal`'s Boltzmann law ``exp(-beta E)``, and each move's empirical
transitions from one state are checked against the kernel's row.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from port.extensions import field_cluster as cluster
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy

EDGES = ((0, 1), (1, 2), (0, 2), (2, 3))
"""A triangle and a pendant: clusters of every size from 1 to 4."""
Q, BETA, DRAWS = 3, 0.7, 40_000


def _graph() -> PottsGraph:
    n = 4
    adjacency = np.zeros((n, n))
    for i, j in EDGES:
        adjacency[i, j] = adjacency[j, i] = 1.0
    indptr = np.concatenate([[0], np.cumsum((adjacency > 0).sum(axis=1))])
    indices = np.concatenate([np.flatnonzero(row) for row in adjacency])
    # NB unequal couplings, so a kernel that ignored J per edge would fail
    table = {(0, 1): 0.8, (0, 2): 0.5, (1, 2): 1.3, (2, 3): 1.0}
    coupling = np.array([table[(min(i, j), max(i, j))] for i in range(n) for j in indices[indptr[i] : indptr[i + 1]]])  # fmt: skip
    return PottsGraph.from_csr(indptr, indices, coupling)


def _states() -> list[tuple[int, ...]]:
    return list(itertools.product(range(Q), repeat=4))


def _components(n: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for i, j in bonds:
        parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(root(x), []).append(x)
    return list(groups.values())


def _kernel(graph: PottsGraph, rows: np.ndarray, move: str) -> np.ndarray:
    """The exact transition matrix over all Q^4 states."""
    states = _states()
    index = {s: k for k, s in enumerate(states)}
    kernel = np.zeros((len(states), len(states)))
    edges = [(int(e[0]), int(e[1])) for e in graph.edge_index]
    p = 1.0 - np.exp(-BETA * graph.edge_coupling)
    for s in states:
        like = [k for k, (i, j) in enumerate(edges) if s[i] == s[j]]
        for bonded in itertools.product((0, 1), repeat=len(like)):
            weight = float(
                np.prod(
                    [p[k] if b else 1 - p[k] for k, b in zip(like, bonded, strict=True)]
                )
            )
            clusters = _components(
                4, [edges[k] for k, b in zip(like, bonded, strict=True) if b]
            )
            # NB Swendsen-Wang redraws every cluster; Wolff the seed's, the seed uniform over the 4 sites
            if move == "swendsen-wang":
                moves, share = [clusters], 1.0
            else:
                moves, share = [[c] for c in clusters for _ in c], 0.25
            for moved in moves:
                law = []
                for c in moved:
                    logw = BETA * rows[c].sum(axis=0)
                    law.append(
                        np.exp(logw - logw.max()) / np.exp(logw - logw.max()).sum()
                    )
                for labels in itertools.product(range(Q), repeat=len(moved)):
                    t = list(s)
                    for c, label in zip(moved, labels, strict=True):
                        for site in c:
                            t[site] = label
                    kernel[index[s], index[tuple(t)]] += (
                        share
                        * weight
                        * float(
                            np.prod([law[k][label] for k, label in enumerate(labels)])
                        )
                    )
    return kernel


@pytest.fixture(scope="module")
def setup() -> tuple[PottsGraph, np.ndarray, np.ndarray]:
    graph = _graph()
    rows = np.random.default_rng(3).normal(scale=1.5, size=(4, Q))
    boltzmann = np.array(
        [np.exp(-BETA * energy(graph, rows, np.array(s))) for s in _states()]
    )
    return graph, rows, boltzmann / boltzmann.sum()


@pytest.mark.analytic
@pytest.mark.parametrize("move", ["swendsen-wang", "wolff"])
def test_the_field_weighted_kernel_keeps_the_boltzmann_law(
    setup: tuple[PottsGraph, np.ndarray, np.ndarray], move: str
) -> None:
    """``pi K = pi`` to 1e-12, with rows summing to one: the move is exact at every beta, not only accepted often."""
    graph, rows, pi = setup
    kernel = _kernel(graph, rows, move)
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(pi @ kernel, pi, atol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("move", ["swendsen-wang", "wolff"])
def test_each_move_draws_its_enumerated_kernel(
    setup: tuple[PottsGraph, np.ndarray, np.ndarray], move: str
) -> None:
    """From (0, 0, 0, 1), 40,000 steps' destinations against the kernel's row: total variation under 0.02."""
    graph, rows, _ = setup
    states = _states()
    index = {s: k for k, s in enumerate(states)}
    origin = (0, 0, 0, 1)
    exact = _kernel(graph, rows, move)[index[origin]]
    rng = np.random.default_rng(11)
    lists = cluster.neighbour_lists(graph)
    counts = np.zeros(len(states))
    for _ in range(DRAWS):
        state = np.array(origin, dtype=np.int64)
        if move == "swendsen-wang":
            cluster.field_weighted_sw(state, graph, rows, rng, BETA)
        else:
            cluster.field_weighted_wolff(state, rows, lists, rng, BETA)
        counts[index[tuple(int(v) for v in state)]] += 1
    assert 0.5 * np.abs(counts / DRAWS - exact).sum() < 0.02


@pytest.mark.analytic
def test_the_heat_bath_draws_the_softmax() -> None:
    """Gumbel-max against ``softmax(beta w)``: 200,000 draws within 0.005 of each probability."""
    weights = np.array([[0.0, 1.0, -2.0, 0.5]])
    rng = np.random.default_rng(0)
    draws = cluster.heat_bath_labels(np.repeat(weights, 200_000, axis=0), 0.8, rng)
    law = np.exp(0.8 * weights[0]) / np.exp(0.8 * weights[0]).sum()
    np.testing.assert_allclose(
        np.bincount(draws, minlength=4) / draws.size, law, atol=0.005
    )
