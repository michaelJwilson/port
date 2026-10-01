"""Field-weighted relabelling for Swendsen-Wang and Wolff, with a Glauber interleave (#559).

Ticket: #559 -- field-weighted Swendsen-Wang and Wolff moves, with a
  Glauber interleave.
Measurement: `docs/study-field-strength.md`, 25 realizations x 50 starts on
  `dev_tree_1s_hard`: 0.01 / 0.06 nats from TRW-S's bound with the
  interleave, at 2.7-3.8x annealed Glauber's runtime.
Exit: graduate to `extensions/` as a `label_solver` row if it beats the
  admitted row end to end (#570); else retire.

`sal`'s cluster moves recolour a Fortuin-Kasteleyn cluster by proposing a
label uniformly and accepting on the field (`sal.sample.potts_mcmc.sweeps._recolour`).
In a field the size of #556's -- 17.6 nats per spot on `dev_tree_1s` -- a
cluster of tens of spots almost never accepts a uniform proposal, so the move
freezes. Given the bonds, the clusters are independent and cluster ``C`` at
label ``c`` has weight ``exp(beta sum_C h[i, c])``: the coupling cancels, as
`sal`'s `label_directed_sweep` states. Drawing ``c`` from that weight is the
exact heat bath on the joint measure, so there is no accept step to fail.

- **Swendsen-Wang** (`field_weighted_sw`): every cluster redrawn from its
  field weight.
- **Wolff** (`field_weighted_wolff`): one cluster, grown from a uniform seed
  through like neighbours as `sal.sample.potts_mcmc.sweeps.wolff_sweep` grows it,
  redrawn from its field weight over all ``q`` labels, its own included. The
  cluster's construction probability ratio between the two configurations
  cancels the coupling term for every label, so the heat bath on what
  remains, the field, keeps detailed balance.
- **Glauber interleave** (`anneal`, ``glauber=True``): one `sal` single-site
  heat-bath sweep after each cluster move, which moves the boundary sites a
  cluster move cannot split off.

The bonds are `sal`'s: ``1 - exp(-beta J)`` on like edges
(`sal.sample.potts_mcmc.sweeps.bond_probability`), merged by
`sal.sample.potts_mcmc.sweeps.bond_roots`. `sal` expresses neither move today:
it would need a heat-bath recolouring beside `_recolour` and a `PottsMove` for
each, run by `anneal_potts` (#559). `sal`'s ghost-spin move (its #1041) puts
the field inside the bond measure instead, and is the follow-up.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "anneal",
    "field_weighted_sw",
    "field_weighted_wolff",
    "heat_bath_labels",
    "neighbour_lists",
]


def heat_bath_labels(
    weights: np.ndarray, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """One label per row of `weights`, drawn with probability proportional to ``exp(beta * weights)``, by Gumbel-max."""
    gumbel = -np.log(-np.log(rng.random(weights.shape)))
    return np.asarray(np.argmax(beta * weights + gumbel, axis=1), dtype=np.int64)


def neighbour_lists(graph: Any) -> Any:
    """`sal`'s per-site neighbour lists of `graph`, the form :func:`field_weighted_wolff` walks."""
    from sal.sample.potts_mcmc.sweeps import adjacency_lists

    return adjacency_lists(*graph.compressed_adjacency())


def field_weighted_sw(
    state: np.ndarray,
    graph: Any,
    rows: np.ndarray,
    rng: np.random.Generator,
    beta: float,
) -> None:
    """One Swendsen-Wang pass in place: `sal`'s bonds, each cluster's label drawn from its field weight."""
    from sal.sample.potts_mcmc.sweeps import bond_probability, bond_roots

    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(first.size) < bond_probability(graph, beta))
    roots = bond_roots(graph.n_nodes, graph.edge_index[active])
    sums = np.zeros_like(rows)
    np.add.at(sums, roots, rows)
    heads = np.unique(roots)
    labels = np.empty(graph.n_nodes, dtype=np.int64)
    labels[heads] = heat_bath_labels(sums[heads], beta, rng)
    state[:] = labels[roots]


def field_weighted_wolff(
    state: np.ndarray,
    rows: np.ndarray,
    lists: Any,
    rng: np.random.Generator,
    beta: float,
) -> int:
    """One Wolff cluster in place, its label drawn from its field weight; returns its size.

    `lists` is :func:`neighbour_lists` of the graph.
    """
    bounds, incident, weights = lists.bounds, lists.incident, lists.weights
    seed = int(rng.integers(state.shape[0]))
    colour = int(state[seed])
    cluster, frontier = [seed], [seed]
    inside = np.zeros(state.shape[0], dtype=bool)
    inside[seed] = True
    while frontier:
        node = frontier.pop()
        for position in range(bounds[node], bounds[node + 1]):
            neighbour = incident[position]
            if inside[neighbour] or state[neighbour] != colour:
                continue
            if rng.random() < 1.0 - np.exp(-beta * weights[position]):
                inside[neighbour] = True
                cluster.append(neighbour)
                frontier.append(neighbour)
    members = np.asarray(cluster, dtype=np.int64)
    state[members] = heat_bath_labels(rows[members].sum(axis=0)[None, :], beta, rng)[0]
    return members.size


def anneal(
    graph: Any,
    rows: np.ndarray,
    start: np.ndarray,
    rng: np.random.Generator,
    temperatures: np.ndarray,
    move: str,
    glauber: bool,
) -> tuple[np.ndarray, float]:
    """The lowest-energy labelling an annealed chain visits, and its energy.

    One step per temperature: one cluster move (`move` is ``"swendsen-wang"``
    or ``"wolff"``), then, with `glauber`, one `sal` heat-bath sweep. A Wolff
    step is one cluster, as `sal.search.ground_state.run_annealed` counts it.
    """
    from sal.backend import Backend
    from sal.sample.potts_mcmc.sweeps import sweep_at
    from sal.sim.potts import energy

    state = np.array(start, dtype=np.int64)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    lists = neighbour_lists(graph)
    sweep = (
        sweep_at(rows, offsets, neighbours, couplings, Backend.RUST)
        if glauber
        else None
    )
    best, best_energy = state.copy(), float(energy(graph, rows, state))
    for temperature in np.asarray(temperatures, dtype=np.float64):
        beta = 1.0 / float(temperature)
        if move == "swendsen-wang":
            field_weighted_sw(state, graph, rows, rng, beta)
        elif move == "wolff":
            field_weighted_wolff(state, rows, lists, rng, beta)
        else:
            msg = f"move {move!r}: swendsen-wang or wolff"
            raise ValueError(msg)
        if sweep is not None:
            sweep(state, rng, beta)
        current = float(energy(graph, rows, state))
        if current < best_energy:
            best, best_energy = state.copy(), current
    return best, best_energy
