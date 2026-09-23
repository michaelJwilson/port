"""The BAF stage's start, and its coupling, chosen from the data's depth (#362).

`run_cnaster` starts the BAF-stage clone search from the phasing grid
(`run_cnaster.py:276`), `npart_phasing^2` rectangles, and runs the HMRF at
the configured `spatial_weight`. On CalicoST's easy simulated sample the
search stays near that start: the fitted BAF clones agree with the grid at
ARI 0.663 and with the truth at 0.723, and started from the planted labels
the same run reaches 0.990. So the start, not the model, loses the clones.

Here the start is a draw of the prior itself: a zero-field `q`-state Potts
sample at coupling `J` over the spots' adjacency, by `snakes_and_ladders`'
Swendsen-Wang move (or Wolff). A Potts sample is a local prior only: one colour recurs over
disjoint patches. So each connected patch of one colour is **recoloured** as
its own clone (`recolour`), and `J` sets how large patches are.

`umi_grow` is the second start, and the one that works: clones grown from
the deepest spots, each to total SNP UMIs over `max_clones`. Measured on the
two samples with `--sal`, 8 clones and the HMRF at `J = 0.45`: clone ARI
0.957 (easy) and 0.959 (hard), against 0.698 and 0.982 from the grid.

For the Potts start, `J` is chosen as the least coupling at which the patches are deep enough to
fit, by `cnaster`'s own floors, `hmrf.min_spots_per_clone` spots and
`hmrf.min_avgumi_per_clone` SNP UMIs per bin (`merge_by_minspots`). Over
`realizations` independent draws per `J`, the share of spots lying in a
patch under either floor is averaged, and `J` is the least at which that
share is at most `tolerance`: one bad patch in one draw does not set it.
The draw used is the one at that `J` with the fewest spots under the floors,
and its under-floor patches are merged, smallest first, into their
smallest neighbouring patch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = [
    "MAX_CLONES",
    "draw",
    "Floors",
    "choose_coupling",
    "recolour",
    "umi_grow",
    "wolff_start",
]

MAX_CLONES = 16
"""The most initial clones the BAF stage is started from."""


@dataclass(frozen=True)
class Floors:
    """What a patch needs to be fitted as a clone."""

    spots: int
    umis: float


def recolour(labels: np.ndarray, adjacency: sp.csr_matrix) -> np.ndarray:
    """Each connected patch of one label, as its own label `0..k-1`."""
    upper = sp.triu(adjacency, k=1).tocoo()
    same = labels[upper.row] == labels[upper.col]
    bonds = sp.coo_matrix(
        (np.ones(int(same.sum())), (upper.row[same], upper.col[same])),
        shape=adjacency.shape,
    )
    _, patches = connected_components(bonds, directed=False)
    patches_: np.ndarray = patches.astype(np.int64)
    return patches_


def _under(patches: np.ndarray, umis: np.ndarray, floors: Floors) -> np.ndarray:
    """Per spot: whether its patch is under either floor."""
    spots = np.bincount(patches)
    depth = np.bincount(patches, weights=umis)
    shallow = (spots < floors.spots) | (depth < floors.umis)
    under: np.ndarray = shallow[patches]
    return under


def _graph(adjacency: sp.csr_matrix, coupling: float) -> Any:
    from snakes_and_ladders.sim.graph import PottsGraph

    upper = sp.triu(adjacency, k=1).tocoo()
    return PottsGraph(
        n_nodes=adjacency.shape[0],
        edges=tuple(zip(upper.row.tolist(), upper.col.tolist(), strict=True)),
        coupling=tuple((coupling * upper.data).tolist()),
    )


def draw(
    adjacency: sp.csr_matrix,
    coupling: float,
    n_colours: int,
    rng: np.random.Generator,
    steps: int,
    move: str = "swendsen-wang",
) -> np.ndarray:
    """One zero-field Potts sample at `coupling`, recoloured into patches.

    `steps` sweeps of `move`, `snakes_and_ladders`' Swendsen-Wang by default:
    one Wolff step flips one cluster, which near and above the transition
    spans most of the lattice, so a Wolff burn-in of whole-lattice clusters
    costs a Swendsen-Wang sweep each. Both leave the same law invariant.
    """
    from snakes_and_ladders.sample.potts_mcmc import PottsMove, sample_potts

    chain = sample_potts(
        _graph(adjacency, coupling),
        np.zeros(n_colours),
        PottsMove(move),
        rng,
        n_sweeps=1,
        burn_in=steps,
    )
    return recolour(np.asarray(chain.states[-1], dtype=np.int64), adjacency)


def choose_coupling(
    adjacency: sp.csr_matrix,
    umis: np.ndarray,
    floors: Floors,
    rng: np.random.Generator,
    *,
    n_colours: int = 4,
    realizations: int = 8,
    tolerance: float = 0.05,
    largest: float = 0.5,
    low: float = 0.05,
    high: float = 3.0,
    bisections: int = 8,
    steps: int = 50,
) -> tuple[float, list[np.ndarray], list[float]]:
    """The least `J` whose draws leave at most `tolerance` of spots under floors.

    A `J` whose draws' largest patch holds more than `largest` of the spots,
    averaged over realizations, is too high whatever its floors say: one
    clone spanning most of the lattice is not a start. Returns `J`, its
    draws, and the under-floor share of each.
    """

    def ensemble(coupling: float) -> tuple[list[np.ndarray], list[float]]:
        draws = [
            draw(adjacency, coupling, n_colours, rng, steps)
            for _ in range(realizations)
        ]
        shares = [float(_under(p, umis, floors).mean()) for p in draws]
        return draws, shares

    best = ensemble(high)

    for _ in range(bisections):
        middle = 0.5 * (low + high)
        draws, shares = ensemble(middle)

        spanning = np.mean([np.bincount(p).max() / p.size for p in draws])

        if spanning > largest:
            high = middle
        elif np.mean(shares) <= tolerance:
            high, best = middle, (draws, shares)
        else:
            low = middle

    return high, best[0], best[1]


def _merge_under(
    patches: np.ndarray,
    umis: np.ndarray,
    floors: Floors,
    adjacency: sp.csr_matrix,
    max_clones: int = MAX_CLONES,
) -> np.ndarray:
    """Merge patches into their smallest neighbour, smallest patch first.

    Until every patch clears `floors` and there are at most `max_clones`.
    """
    labels = patches.copy()

    while True:
        under = _under(labels, umis, floors)
        n_patches = np.unique(labels).size

        if n_patches > max_clones and not under.any():
            under = np.ones_like(under)

        if not under.any() or n_patches == 1:
            _, dense = np.unique(labels, return_inverse=True)
            merged: np.ndarray = dense.astype(np.int64)
            return merged

        spots = np.bincount(labels)
        small = min(np.unique(labels[under]).tolist(), key=lambda p: (spots[p], p))
        members = labels == small
        upper = adjacency.tocoo()
        crossing = members[upper.row] & ~members[upper.col]
        neighbours = labels[upper.col[crossing]]

        if neighbours.size == 0:
            _, dense = np.unique(labels, return_inverse=True)
            isolated: np.ndarray = dense.astype(np.int64)
            return isolated

        # NB into the *smallest* neighbour, not the one sharing most edges:
        #    that one is usually the patch already absorbing the rest, and
        #    merging into it snowballs -- measured at a floor of 40 per bin,
        #    405 patches ended as 3 clones where 16 of the floor were due.
        touching = np.unique(neighbours)
        labels[members] = touching[np.argmin(spots[touching])]


def start(
    adjacency: sp.csr_matrix,
    umis: np.ndarray,
    floors: Floors,
    rng: np.random.Generator,
    coupling: float | None = None,
    **options: Any,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """`J`, the initial labels, and what chose them.

    With `coupling` given, its `realizations` draws are taken at that `J`
    rather than a `J` searched for.
    """
    max_clones = int(options.pop("max_clones", MAX_CLONES))

    if coupling is None:
        coupling, draws, shares = choose_coupling(
            adjacency, umis, floors, rng, **options
        )
    else:
        n_colours = int(options.get("n_colours", 4))
        steps = int(options.get("steps", 50))
        draws = [
            draw(adjacency, coupling, n_colours, rng, steps)
            for _ in range(int(options.get("realizations", 8)))
        ]
        shares = [float(_under(p, umis, floors).mean()) for p in draws]
    chosen = int(np.argmin(shares))
    labels = _merge_under(
        draws[chosen],
        umis,
        floors,
        adjacency,
        max_clones,
    )
    record = {
        "coupling": coupling,
        "shares": shares,
        "patches": int(np.unique(draws[chosen]).size),
        "clones": int(np.unique(labels).size),
    }
    return coupling, labels, record


def umi_grow(
    adjacency: sp.csr_matrix,
    umis: np.ndarray,
    rng: np.random.Generator,
    max_clones: int = MAX_CLONES,
) -> np.ndarray:
    """Clones grown from the deepest spots, each to an equal share of the UMIs.

    The target per clone is the total SNP UMIs over `max_clones`. A clone is rooted at the unassigned spot with the most
    UMIs and grows by adding one unassigned neighbour of the clone at a time,
    drawn with probability proportional to its UMIs, until it holds the
    target; its spots then leave the heap and the next clone is rooted at the
    deepest spot left. Clones hemmed in under the target leave more than
    `max_clones`; the smallest is merged into its smallest neighbour until
    `max_clones` remain.
    """
    n_spots = umis.size
    target = float(umis.sum()) / max_clones
    labels = np.full(n_spots, -1, dtype=np.int64)
    order = np.argsort(-umis, kind="stable")
    indptr, indices = adjacency.indptr, adjacency.indices
    clone = 0

    for root in order:
        if labels[root] >= 0:
            continue

        labels[root] = clone
        depth = float(umis[root])
        frontier = {int(n) for n in indices[indptr[root] : indptr[root + 1]]}
        frontier = {n for n in frontier if labels[n] < 0}

        while depth < target and frontier:
            candidates = np.fromiter(frontier, dtype=np.int64)
            weights = umis[candidates].astype(np.float64) + 1e-12
            chosen = int(rng.choice(candidates, p=weights / weights.sum()))
            labels[chosen] = clone
            depth += float(umis[chosen])
            frontier.discard(chosen)
            frontier.update(
                int(n)
                for n in indices[indptr[chosen] : indptr[chosen + 1]]
                if labels[n] < 0
            )

        clone += 1

    return _merge_under(labels, umis, Floors(spots=0, umis=0.0), adjacency, max_clones)


@contextmanager
def wolff_start(
    floors: Floors | None = None,
    seed: int = 0,
    umis_per_bin: float | None = None,
    method: str = "potts",
    **options: Any,
) -> Iterator[dict[str, Any]]:
    """Start the BAF stage from :func:`start`, at the `J` it chose, for the block.

    Wraps `port.patch.hmrf.run_core_inference` for its first `params="sp"`
    call: the initial clone index becomes the patches and `spatial_weight`
    the chosen `J`. `floors` defaults to the configuration's
    `hmrf.min_spots_per_clone` and `hmrf.min_avgumi_per_clone`.
    """
    import port.patch.hmrf as patch

    record: dict[str, Any] = {}
    original = patch.run_core_inference

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("params") == "sp" and "labels" not in record:
            from cnaster.config import get_global_config

            config = get_global_config()
            totals = np.asarray(args[3], dtype=np.float64)
            chosen = floors or (
                Floors(spots=0, umis=umis_per_bin * totals.shape[0])
                if umis_per_bin is not None
                else Floors(
                    spots=int(config.hmrf.min_spots_per_clone),
                    umis=float(config.hmrf.min_avgumi_per_clone) * totals.shape[0],
                )
            )
            adjacency = sp.csr_matrix(kwargs["adjacency_mat"])
            rng = np.random.default_rng(seed)
            coupling: float | None

            if method == "grow":
                labels = umi_grow(
                    adjacency,
                    totals.sum(axis=0),
                    rng,
                    int(options.get("max_clones", MAX_CLONES)),
                )
                coupling = options.get("coupling")
                found = {"coupling": coupling, "clones": int(labels.max()) + 1}
            else:
                coupling, labels, found = start(
                    adjacency, totals.sum(axis=0), chosen, rng, **options
                )
            record.update(found, labels=labels, floors=chosen)
            args = (
                *args[:5],
                [np.flatnonzero(labels == c) for c in range(int(labels.max()) + 1)],
                *args[6:],
            )
            if coupling is not None:
                kwargs["spatial_weight"] = coupling
        return original(*args, **kwargs)

    patch.run_core_inference = wrapped

    try:
        yield record
    finally:
        patch.run_core_inference = original
