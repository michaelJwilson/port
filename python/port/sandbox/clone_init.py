"""Clone initializers for the BAF stage, trialled against the planted truth (#358).

`run_cnaster` starts its BAF-stage clone search from the phasing grid
(`run_cnaster.py:276`), `npart_phasing^2` rectangles, whatever the data say.
This module holds the candidates #358 trials in its place; `trial()` installs
one into a run by replacing the initial clone index `run_core_inference` is
handed for `params="sp"`, and `tests/clone_init_trial.py` scores them.

Every candidate takes an :class:`InitContext` -- the spot coordinates and
adjacency, the BAF counts and the coupling `J` the HMRF assumes -- the number
of clones, and a generator, and returns one label per spot in `0..n-1`.

- `grid`: `cnaster`'s rectangles at `ceil(sqrt(n))^2`, blocks folded onto
  `n` labels; the baseline.
- `best_equal`: `cnaster.spatial.best_equal_partition` at the same size,
  1,000 trials.
- `rectangles`: Dirichlet blocks, as `initialize_rectangular_clones` draws
  them, redrawn until every clone holds `floor` spots -- it terminates where
  upstream's loop cannot (#248).
- `voronoi`: k-means on the coordinates.
- `sw_prior`: a Swendsen-Wang sample of the `n`-state Potts prior at the `J`
  the HMRF assumes, bond probability `1 - exp(-J w)`: the labelling the prior
  itself would draw.
- `sw_umi`: the same, with `J` raised by bisection until every label's
  realized SNP-covering UMIs reach the depth the HMM needs (`20` per bin, the
  threshold `initialize_rdr_clone_refininement` applies): the coupling
  tailored to the data's depth rather than assumed.
- `features`: k-means on neighbour-smoothed per-bin minor-allele fractions,
  reduced to 8 components, with the coordinates.
- `spectral`: spectral clustering of the adjacency reweighted by the
  similarity of those features.
- `ward`: Ward agglomeration of the features with the adjacency as
  connectivity, so every clone is spatially connected.
- `best_sw_umi`, `best_features`: the best of `realizations` draws by
  :func:`objective`, the BAF pseudobulk log-likelihood plus the Potts term.
- `oracle`: the planted labels, an upper bound for the table only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

__all__ = ["CANDIDATES", "InitContext", "objective", "trial"]

DEPTH_PER_BIN = 20
"""SNP-covering UMIs per bin a clone's pseudobulk needs (`spatial.py:392`)."""


@dataclass
class InitContext:
    """What an initializer may read."""

    coords: np.ndarray
    adjacency: sp.csr_matrix
    b_counts: np.ndarray
    """`(n_bins, n_spots)` B-allele counts."""
    totals: np.ndarray
    """`(n_bins, n_spots)` SNP-covering UMIs."""
    coupling: float
    """The HMRF's `spatial_weight`, the `J` it assumes."""
    truth: np.ndarray | None = None


Initializer = Callable[[InitContext, int, np.random.Generator], np.ndarray]


def _fold(labels: np.ndarray, n: int) -> np.ndarray:
    _, dense = np.unique(labels, return_inverse=True)
    folded: np.ndarray = (dense % n).astype(np.int64)
    return folded


def grid(ctx: InitContext, n: int, _rng: np.random.Generator) -> np.ndarray:
    from cnaster.spatial import rectangle_partition

    side = int(np.ceil(np.sqrt(n)))
    _, blocks = rectangle_partition(ctx.coords, side, side)
    return _fold(blocks, n)


def best_equal(ctx: InitContext, n: int, _rng: np.random.Generator) -> np.ndarray:
    from cnaster.spatial import best_equal_partition

    side = int(np.ceil(np.sqrt(n)))
    _, blocks = best_equal_partition(ctx.coords, side, side, n_trials=1_000)
    return _fold(blocks, n)


def rectangles(
    ctx: InitContext, n: int, rng: np.random.Generator, tries: int = 1_000
) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n)))
    floor = 0.2 * len(ctx.coords) / n

    for _ in range(tries):
        digits = []
        for axis in (0, 1):
            share = rng.dirichlet(np.ones(side) * 10)
            low, high = np.percentile(ctx.coords[:, axis], [5, 95])
            edges = low + (high - low) * np.cumsum(share)
            edges[-1] = ctx.coords[:, axis].max() + 1
            digits.append(np.digitize(ctx.coords[:, axis], edges, right=True))
        mapping = rng.permutation(np.arange(side * side) % n)
        labels: np.ndarray = mapping[digits[0] * side + digits[1]]
        if np.bincount(labels, minlength=n).min() > floor:
            return labels.astype(np.int64)

    return grid(ctx, n, rng)


def voronoi(ctx: InitContext, n: int, rng: np.random.Generator) -> np.ndarray:
    from sklearn.cluster import KMeans

    seed = int(rng.integers(2**31))
    labels: np.ndarray = KMeans(n_clusters=n, n_init=4, random_state=seed).fit_predict(
        ctx.coords
    )
    return labels.astype(np.int64)


def swendsen_wang(
    ctx: InitContext, n: int, rng: np.random.Generator, coupling: float, sweeps: int
) -> np.ndarray:
    """A zero-field `n`-state Potts sample at `coupling`, by Swendsen-Wang."""
    from scipy.sparse.csgraph import connected_components

    upper = sp.triu(ctx.adjacency, k=1).tocoo()
    probability = 1.0 - np.exp(-coupling * upper.data)
    labels = rng.integers(n, size=len(ctx.coords))

    for _ in range(sweeps):
        bonded = (labels[upper.row] == labels[upper.col]) & (
            rng.random(upper.data.size) < probability
        )
        bonds = sp.coo_matrix(
            (np.ones(bonded.sum()), (upper.row[bonded], upper.col[bonded])),
            shape=ctx.adjacency.shape,
        )
        n_clusters, cluster = connected_components(bonds, directed=False)
        labels = rng.integers(n, size=n_clusters)[cluster]

    return labels.astype(np.int64)


def sw_prior(ctx: InitContext, n: int, rng: np.random.Generator) -> np.ndarray:
    return swendsen_wang(ctx, n, rng, ctx.coupling, sweeps=50)


def _deep_enough(ctx: InitContext, labels: np.ndarray, n: int) -> bool:
    per_spot = ctx.totals.sum(axis=0)
    depth = np.bincount(labels, weights=per_spot, minlength=n)
    return bool(depth.min() >= DEPTH_PER_BIN * ctx.totals.shape[0])


def sw_umi(ctx: InitContext, n: int, rng: np.random.Generator) -> np.ndarray:
    """The prior's sample, at the least `J` whose labels are each deep enough."""
    low, high = ctx.coupling, max(4.0 * ctx.coupling, 4.0)
    best = swendsen_wang(ctx, n, rng, high, sweeps=50)

    for _ in range(8):
        middle = 0.5 * (low + high)
        labels = swendsen_wang(ctx, n, rng, middle, sweeps=50)
        if _deep_enough(ctx, labels, n):
            high, best = middle, labels
        else:
            low = middle

    return best


def _features(ctx: InitContext, components: int = 8) -> np.ndarray:
    from sklearn.decomposition import PCA

    smooth = ctx.adjacency + sp.eye(ctx.adjacency.shape[0])
    b = np.asarray(smooth @ ctx.b_counts.T)
    t = np.asarray(smooth @ ctx.totals.T)
    fraction = np.where(t > 0, b / np.maximum(t, 1), 0.5)
    minor = np.minimum(fraction, 1.0 - fraction)
    reduced = PCA(n_components=min(components, minor.shape[1])).fit_transform(minor)
    scaled: np.ndarray = reduced / (reduced.std(axis=0) + 1e-12)
    return scaled


def features(ctx: InitContext, n: int, rng: np.random.Generator) -> np.ndarray:
    from sklearn.cluster import KMeans

    coords = (ctx.coords - ctx.coords.mean(0)) / (ctx.coords.std(0) + 1e-12)
    stacked = np.hstack([_features(ctx), 0.5 * coords])
    seed = int(rng.integers(2**31))
    labels: np.ndarray = KMeans(n_clusters=n, n_init=4, random_state=seed).fit_predict(
        stacked
    )
    return labels.astype(np.int64)


def spectral(ctx: InitContext, n: int, rng: np.random.Generator) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    f = _features(ctx)
    upper = ctx.adjacency.tocoo()
    similarity = np.exp(-np.sum((f[upper.row] - f[upper.col]) ** 2, axis=1) / 2.0)
    affinity = sp.coo_matrix(
        (upper.data * similarity + 1e-6, (upper.row, upper.col)),
        shape=ctx.adjacency.shape,
    ).tocsr()
    seed = int(rng.integers(2**31))
    model = SpectralClustering(
        n_clusters=n, affinity="precomputed", random_state=seed, assign_labels="kmeans"
    )
    labels: np.ndarray = model.fit_predict(affinity)
    return labels.astype(np.int64)


def ward(ctx: InitContext, n: int, _rng: np.random.Generator) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=n, linkage="ward", connectivity=ctx.adjacency
    )
    labels: np.ndarray = model.fit_predict(_features(ctx))
    return labels.astype(np.int64)


def oracle(ctx: InitContext, n: int, _rng: np.random.Generator) -> np.ndarray:
    if ctx.truth is None:
        msg = "the oracle needs the planted labels"
        raise ValueError(msg)
    return _fold(ctx.truth, n)


def objective(ctx: InitContext, labels: np.ndarray, n: int) -> float:
    """BAF pseudobulk log-likelihood at each clone's MLE, plus the Potts term.

    What a labelling explains of the allele counts, summed over clones and
    bins, and what the prior pays for it, `J` per agreeing edge weight.
    """
    loglik = 0.0

    for clone in range(n):
        members = labels == clone
        if not members.any():
            continue
        b = ctx.b_counts[:, members].sum(axis=1)
        t = ctx.totals[:, members].sum(axis=1)
        p = np.clip(np.where(t > 0, b / np.maximum(t, 1), 0.5), 1e-9, 1 - 1e-9)
        loglik += float(np.sum(b * np.log(p) + (t - b) * np.log(1 - p)))

    upper = sp.triu(ctx.adjacency, k=1).tocoo()
    agree = labels[upper.row] == labels[upper.col]
    return loglik + ctx.coupling * float(upper.data[agree].sum())


def best_of(candidate: Initializer, realizations: int = 5) -> Initializer:
    """The best of `realizations` draws of `candidate`, by :func:`objective`."""

    def best(ctx: InitContext, n: int, rng: np.random.Generator) -> np.ndarray:
        draws = [candidate(ctx, n, rng) for _ in range(realizations)]
        return max(draws, key=lambda labels: objective(ctx, labels, n))

    return best


CANDIDATES: dict[str, Initializer] = {
    "grid": grid,
    "best_equal": best_equal,
    "rectangles": rectangles,
    "voronoi": voronoi,
    "sw_prior": sw_prior,
    "sw_umi": sw_umi,
    "features": features,
    "spectral": spectral,
    "ward": ward,
    "best_sw_umi": best_of(sw_umi),
    "best_features": best_of(features),
    "best_rectangles": best_of(rectangles),
    "oracle": oracle,
}


@contextmanager
def trial(
    name: str, truth: np.ndarray | None = None, seed: int = 0
) -> Iterator[dict[str, Any]]:
    """Start the BAF stage from `CANDIDATES[name]` for the block.

    Wraps `port.patch.hmrf.run_core_inference`, which the shift table
    installs, before a run patches it in; the coordinates are kept from
    `construct_multislice_lattice_adjacency`, which is handed them first.
    Yields a record of the initial labels used.
    """
    import cnaster.scripts.run_cnaster as pipeline

    import port.patch.hmrf as patch

    record: dict[str, Any] = {}
    original, adjacency_builder = (
        patch.run_core_inference,
        (pipeline.construct_multislice_lattice_adjacency),
    )

    def keep_coords(
        sample_ids: Any, sample_list: Any, coords: Any, *a: Any, **k: Any
    ) -> Any:
        record["coords"] = np.asarray(coords, dtype=np.float64)
        return adjacency_builder(sample_ids, sample_list, coords, *a, **k)

    def start(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("params") == "sp" and "labels" not in record:
            n = len(args[5])
            ctx = InitContext(
                coords=record["coords"],
                adjacency=sp.csr_matrix(kwargs["adjacency_mat"]),
                b_counts=np.asarray(args[0][:, 1, :], dtype=np.float64),
                totals=np.asarray(args[3], dtype=np.float64),
                coupling=float(kwargs.get("spatial_weight", 1.0)),
                truth=truth,
            )
            labels = CANDIDATES[name](ctx, n, np.random.default_rng(seed))
            record["labels"] = labels
            args = (
                *args[:5],
                [np.flatnonzero(labels == c) for c in range(n)],
                *args[6:],
            )
        return original(*args, **kwargs)

    patch.run_core_inference = start
    pipeline.construct_multislice_lattice_adjacency = keep_coords

    try:
        yield record
    finally:
        patch.run_core_inference = original
        pipeline.construct_multislice_lattice_adjacency = adjacency_builder
