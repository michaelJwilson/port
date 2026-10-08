"""Set aside (#541): clone-label starts, existing and new, behind one signature.

Ticket: #541 -- which clone-label start places the spots best before a Potts
  solver, and which solver from it.
Measurement: `docs/nb/clone_label_study.ipynb`, on dev_tree 60 x 50 r0 (`3381575a`).
Exit: a start graduates to `patch/` if, end to end under `--sal`, it is at
  least as accurate as `grid2` on dev_tree, easy, hard and
  `dev_shared_unique` and no slower; else it stays here with its numbers.

Every start is `start(capture, rng, field=None, k=K) -> labels`: one label per
spot. `capture` is `problem.Capture`, the spots' counts, coordinates,
adjacency and UMIs; `field` is `(spots, q)` spot log-likelihoods under the
clone profiles a labelling in hand gives -- `grid2`'s, in the study -- and is
read only by the starts marked `needs_field`. Each docstring states the
hypothesis the study tests.
"""

# ruff: noqa: ARG001 -- every start takes one signature; not every start reads each argument

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

K = 4
"""Clones a clustering start asks for: `grid2`'s, 2 x 2 rectangles per slice."""

__all__ = ["STARTS", "K", "Start"]


class Start(NamedTuple):
    """One start: its function, whether it draws, and whether it reads the field."""

    name: str
    run: Callable[..., np.ndarray]
    stochastic: bool
    needs_field: bool
    source: str


def _adjacency(capture: Any) -> Any:
    import scipy.sparse as sp

    n = capture.n_spots
    return sp.csr_matrix(
        (capture.weights, capture.indices, capture.indptr), shape=(n, n)
    )


def _labels_of(index: list[np.ndarray], n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int64)
    for clone, spots in enumerate(index):
        labels[np.asarray(spots, dtype=np.int64)] = clone
    return labels


def _grid(capture: Any, npart: int, spots: np.ndarray | None = None) -> np.ndarray:
    """`cnaster`'s rectangles, `npart x npart` per slice, over `spots` (all by default)."""
    from cnaster.spatial import initialize_clones

    keep = np.arange(capture.n_spots) if spots is None else np.flatnonzero(spots)
    index = initialize_clones(
        capture.coords[keep], np.asarray(capture.sample_ids)[keep], npart, npart
    )
    labels = np.full(capture.n_spots, -1, dtype=np.int64)
    labels[keep] = _labels_of(index, keep.size)
    return labels


def grid2(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """`cnaster`'s default: 2 x 2 rectangles. Data-blind; holds while clones are compact, fails where a rectangle straddles two."""
    return _grid(capture, 2)


def grid3(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """3 x 3 rectangles: more clones than planted, for a solver or the floor to merge (#490: over-partitions)."""
    return _grid(capture, 3)


def normal_first(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """The run's normal candidates as one clone, `grid2` over the rest, without #490's second run.

    Hypothesis: the normal candidates (`determine_normal_candidates`) are the
    normal clone, so starting it whole removes the rectangles' largest error.
    """
    normal = np.asarray(capture.normal_candidates, dtype=bool)
    labels = _grid(capture, 2, ~normal) + 1
    labels[normal] = 0
    return labels


def umi_grow(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """#491's growth from the deepest spots to equal UMI shares. Seed-dependent: its seeds come from depth, not signal."""
    from port.sandbox.wolff_init import umi_grow as grow

    return grow(_adjacency(capture), capture.umis, rng, max_clones=k)


def _wolff(coupling: float) -> Callable[..., np.ndarray]:
    def wolff(
        capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
    ) -> np.ndarray:
        from port.sandbox.wolff_init import Floors, start

        _, labels, _ = start(
            _adjacency(capture),
            capture.umis,
            Floors(spots=capture.floor, umis=0.0),
            rng,
            coupling=coupling,
            max_clones=k,
        )
        return labels

    wolff.__doc__ = (
        f"#358's Potts draw at J = {coupling:g}, patches floored: data-blind, "
        "its patch sizes set by J."
    )
    return wolff


def field_argmax(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """Each spot's most likely clone. Salt-and-pepper wherever a spot's UMIs cannot separate clones."""
    return np.asarray(np.argmax(field, axis=1), dtype=np.int64)


def uniform(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """A clone per spot, uniformly: the floor on what a start is worth."""
    return np.asarray(rng.integers(0, k, capture.n_spots), dtype=np.int64)


# -- new starts ----------------------------------------------------------------


def _smoothed(capture: Any, values: np.ndarray, hops: int) -> np.ndarray:
    """`values` averaged over each spot's `hops`-hop neighbourhood, itself included."""
    import scipy.sparse as sp

    walk = _adjacency(capture).astype(bool).astype(np.float64) + sp.identity(
        capture.n_spots
    )
    out = np.asarray(values, dtype=np.float64)
    for _ in range(hops):
        out = np.asarray(walk.dot(out)) / np.asarray(walk.sum(axis=1))
    return out


def _centred(field: np.ndarray) -> np.ndarray:
    """Each spot's field less its maximum: its log-likelihood ratios, scale-free across spots."""
    return np.asarray(field - field.max(axis=1, keepdims=True))


def grid_relabel(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """`grid3`'s rectangles, each given the clone whose profile explains its spots best.

    Hypothesis: a rectangle straddling two clones is where `grid2` fails;
    finer rectangles, each relabelled by the summed field, merge where they
    agree and so place the boundary at the rectangles' resolution.
    """
    rectangles = _grid(capture, 3)
    labels = np.empty(capture.n_spots, dtype=np.int64)
    for rectangle in np.unique(rectangles):
        spots = rectangles == rectangle
        labels[spots] = int(np.argmax(field[spots].sum(axis=0)))
    return labels


def smoothed_argmax(hops: int) -> Callable[..., np.ndarray]:
    def start(
        capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
    ) -> np.ndarray:
        return np.asarray(
            np.argmax(_smoothed(capture, _centred(field), hops), axis=1), dtype=np.int64
        )

    start.__doc__ = (
        f"The field pooled over {hops}-hop neighbourhoods, then its argmax. "
        "Hypothesis: pooling lifts a low-UMI spot's field over the noise that "
        "salts the plain argmax."
    )
    return start


def mean_field(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """Ten mean-field iterations of the Potts posterior, then its argmax.

    Hypothesis: the posterior marginals at the run's coupling are the
    field argmax corrected by the neighbours, without a solver's cost.
    """
    from scipy.special import softmax

    adjacency = _adjacency(capture)
    q = softmax(field, axis=1)
    for _ in range(10):
        q = softmax(
            field + capture.spatial_weight * np.asarray(adjacency.dot(q)), axis=1
        )
    return np.asarray(np.argmax(q, axis=1), dtype=np.int64)


def agglomerative(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """Ward clustering of the 1-hop pooled field, joined only across graph edges, cut at `k`.

    Hypothesis: contiguity is a hard constraint a clone mostly meets, so
    clustering along the graph finds clone-shaped regions without a coupling.
    """
    from sklearn.cluster import AgglomerativeClustering

    features = _smoothed(capture, _centred(field), 1)
    fitted = AgglomerativeClustering(
        n_clusters=k, connectivity=_adjacency(capture), linkage="ward"
    ).fit(features)
    return np.asarray(fitted.labels_, dtype=np.int64)


def spectral(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """k-means in the spectral embedding of the graph, each edge weighted by its spots' field similarity.

    Hypothesis: edges inside a clone join spots of like field, so the
    graph's `k` weakest cuts are the clone boundaries.
    """
    import scipy.sparse as sp
    from sklearn.cluster import SpectralClustering

    features = _smoothed(capture, _centred(field), 1)
    upper = sp.triu(_adjacency(capture), k=1).tocoo()
    distance = np.square(features[upper.row] - features[upper.col]).sum(axis=1)
    scale = float(np.median(distance)) or 1.0
    affinity = sp.coo_matrix(
        (np.exp(-distance / scale), (upper.row, upper.col)),
        shape=(capture.n_spots,) * 2,
    )
    affinity = (affinity + affinity.T).tocsr()
    fitted = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=int(rng.integers(2**31)),
        assign_labels="kmeans",
    ).fit(affinity)
    return np.asarray(fitted.labels_, dtype=np.int64)


def posterior_draw(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """One Swendsen-Wang draw from `p(labels | field, beta)` after 50 sweeps (`sal`'s sampler).

    Hypothesis: a draw at the run's temperature sits in the posterior's bulk
    rather than at a rectangle's edges, so a solver from it starts near the mode.
    """
    from sal.backend import Backend
    from sal.sample.potts_mcmc import PottsMove, Recolour, sample_potts

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph

    graph = potts_graph_from(
        CsrGraph(capture.indptr, capture.indices, capture.weights),
        capture.spatial_weight,
    )
    # NB `sal`'s Python Swendsen-Wang pass, its default before
    #    michaelJwilson/snakes_and_ladders#1283, as the move alone with the
    #    uniform recolour, a bare move's meaning before #1323: the draws
    #    `docs/nb/clone_label_study.ipynb` records replay at their seeds (T- #781).
    chain = sample_potts(
        graph,
        field,
        [PottsMove.SWENDSEN_WANG],
        rng,
        n_sweeps=50,
        recolour=Recolour.UNIFORM,
        cluster_backend=Backend.PYTHON,
    )
    draws = np.asarray(chain.states)
    return np.asarray(draws[-1], dtype=np.int64).reshape(-1)


def baf_agglomerative(
    capture: Any, rng: np.random.Generator, field: Any = None, k: int = K
) -> np.ndarray:
    """Field-free: Ward clustering, along the graph, of each spot's B-allele deviation per 100 bins, pooled 2 hops.

    Hypothesis: allelic imbalance alone separates the clones before any
    profile exists, so this could replace the BAF stage's rectangles.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA

    chunk = np.arange(capture.b.shape[0]) // 100
    trials = np.array(
        [capture.trials[chunk == c].sum(axis=0) for c in np.unique(chunk)]
    )
    b = np.array([capture.b[chunk == c].sum(axis=0) for c in np.unique(chunk)])
    deviation = np.abs(b - 0.5 * trials) / np.sqrt(np.maximum(trials, 1.0))
    features = _smoothed(capture, deviation.T, 2)
    reduced = PCA(n_components=min(10, features.shape[1])).fit_transform(features)
    fitted = AgglomerativeClustering(
        n_clusters=k, connectivity=_adjacency(capture), linkage="ward"
    ).fit(reduced)
    return np.asarray(fitted.labels_, dtype=np.int64)


STARTS: dict[str, Start] = {
    s.name: s
    for s in (
        Start("grid2", grid2, False, False, "cnaster"),
        Start("grid3", grid3, False, False, "cnaster"),
        Start("normal-first", normal_first, False, False, "port #490, one run"),
        Start("umi_grow", umi_grow, True, False, "port #491"),
        Start("wolff J=0.5", _wolff(0.5), True, False, "port #358"),
        Start("wolff J=1 (beta)", _wolff(1.0), True, False, "port #358"),
        Start("wolff J=2", _wolff(2.0), True, False, "port #358"),
        Start("field argmax", field_argmax, False, True, "sal"),
        Start("uniform", uniform, True, False, "floor"),
        Start("grid3 relabelled", grid_relabel, False, True, "new"),
        Start("smoothed argmax 1", smoothed_argmax(1), False, True, "new"),
        Start("smoothed argmax 2", smoothed_argmax(2), False, True, "new"),
        Start("mean field", mean_field, False, True, "new"),
        Start("agglomerative", agglomerative, False, True, "new"),
        Start("spectral", spectral, False, True, "new"),
        Start("posterior draw", posterior_draw, True, True, "new"),
        Start("BAF agglomerative", baf_agglomerative, False, False, "new"),
    )
}
"""Every start, in the study's order."""
