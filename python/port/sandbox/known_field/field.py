"""The RDR + BAF field of a drawn realization at its planted copy states and clone profiles (#556).

Ticket: #556 -- the clone field of a drawn realization at its planted law,
  the problem the study's solvers are compared on.
Measurement: `docs/study-field-strength.md`, the field margins: median 9.2-9.8
  nats per spot on dev_tree against 0.8 and 0.2 on CalicoST easy and hard.
Exit: stays the study's, since the pipeline never knows the planted law;
  retire with the study.

- RDR: a spot's gene UMIs are DM(T_s, kappa q_k), q_k the baseline lambda
  times clone k's depth factor at its planted copies, normalized
  (`port.sim.entries.dirichlet_multinomial`). The log-likelihood less its
  clone-free terms is sum_{g: c_g > 0} [lgamma(c_g + kappa q_gk) -
  lgamma(kappa q_gk)]; a Dirichlet aggregates, so the gene-level law is exact.
- BAF: haplotype-A reads (the written A, or B where the truth phase switched)
  are BetaBinomial(n, p_k s, (1 - p_k) s), s = 1 / rho - 1, at clone k's
  planted haplotype-A share under the manifest's admixture law, rho its
  `bb_overdispersion` (`port.sim.draw._alleles`); Binomial(n, p_k) at rho = 0.
  p is clipped to [`CLIP`, 1 - `CLIP`], so a read against an LOH costs 9.2
  nats at rho = 0 rather than excluding the clone.
- Graph: the hex array's nearest neighbours at weight 1, coupling 1.0, the
  pipeline's `spatial_weight`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "CLIP",
    "KnownProblem",
    "baf_field",
    "hex_graph",
    "missed",
    "overdispersion",
    "problems",
    "rdr_field",
]

CLIP = 1e-4
"""The allele share's clip: a read against an LOH costs log(1 / CLIP) = 9.2 nats."""

BETA = 1.0
"""The pipeline's `spatial_weight`."""


class KnownProblem(NamedTuple):
    """One realization's labelling problem."""

    realization: int
    field: np.ndarray
    """`(spots, clones)` log-likelihood, clone-free terms dropped."""
    planted: np.ndarray
    """`(spots,)` the drawn labels, `normal` 0."""
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    spatial_weight: float
    draw_seconds: float
    field_seconds: float

    @property
    def n_spots(self) -> int:
        return int(self.field.shape[0])


def hex_graph(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR `(indptr, indices, weights)` of each point's nearest neighbours at weight 1."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    spacing = float(np.min(tree.query(points, k=2)[0][:, 1]))
    pairs = tree.query_pairs(spacing * 1.01, output_type="ndarray")
    n = len(points)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    matrix = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    matrix.sort_indices()
    return matrix.indptr, matrix.indices, matrix.data


def _segment_sum(values: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    out = np.zeros(indptr.size - 1)
    nonempty = np.flatnonzero(np.diff(indptr) > 0)
    out[nonempty] = np.add.reduceat(values, indptr[nonempty])
    return out


def rdr_field(counts: Any, alpha: np.ndarray) -> np.ndarray:
    """`(spots, clones)` DM log-likelihood less its clone-free terms; `counts` spots x genes, `alpha` `(genes, clones)`.

    `alpha` is kappa q, every column summing to the same kappa: only then are
    lgamma(kappa) - lgamma(n + kappa) and the multinomial coefficient free of
    the clone, and dropped.
    """
    from scipy.special import gammaln

    counts = counts.tocsr()
    c = counts.data.astype(np.float64)
    out = np.empty((counts.shape[0], alpha.shape[1]))
    for k in range(alpha.shape[1]):
        a = alpha[counts.indices, k]
        out[:, k] = _segment_sum(gammaln(c + a) - gammaln(a), counts.indptr)
    return out


def baf_field(
    a: Any, b: Any, switched: np.ndarray, share: np.ndarray, rho: float = 0.0
) -> np.ndarray:
    """`(spots, clones)` log-likelihood of haplotype-A reads less its clone-free terms; `share` `(clones, snps)`.

    Beta-binomial at overdispersion `rho` (lgamma(s) - lgamma(n + s) and the
    binomial coefficient dropped, s shared by every clone); binomial at 0.
    """
    from scipy.special import gammaln

    a, b = a.tocsr(), b.tocsr()
    total = (a + b).tocsr()
    rows = np.repeat(np.arange(total.shape[0]), np.diff(total.indptr))
    cols = total.indices
    # NB haplotype-A reads: the written A, or the written B where the phase switched
    a_at = np.asarray(a[rows, cols]).ravel()
    b_at = np.asarray(b[rows, cols]).ravel()
    hap_a = np.where(switched[cols], b_at, a_at).astype(np.float64)
    n = total.data.astype(np.float64)
    p = np.clip(share, CLIP, 1 - CLIP)
    out = np.empty((total.shape[0], share.shape[0]))
    for k in range(share.shape[0]):
        pk = p[k, cols]
        if rho > 0:
            s = 1.0 / rho - 1.0
            terms = (
                gammaln(hap_a + pk * s)
                + gammaln(n - hap_a + (1 - pk) * s)
                - gammaln(pk * s)
                - gammaln((1 - pk) * s)
            )
        else:
            terms = hap_a * np.log(pk) + (n - hap_a) * np.log1p(-pk)
        out[:, k] = _segment_sum(terms, total.indptr)
    return out


def missed(label: np.ndarray, truth: np.ndarray) -> int:
    """Labels unlike the planted ones under the 1-1 matching of fitted to planted labels that misses fewest.

    A solver's label k is clone k's field column, so the identity is usually
    the best matching; after a color merge a clone's spots can sit under
    another's label, and the matching does not count that as a miss.
    """
    from scipy.optimize import linear_sum_assignment

    n = int(max(label.max(), truth.max())) + 1
    agree = np.zeros((n, n), dtype=np.int64)
    np.add.at(agree, (label, truth), 1)
    rows, cols = linear_sum_assignment(-agree)
    return int(label.size - agree[rows, cols].sum())


def overdispersion(
    successes: np.ndarray, trials: np.ndarray, share: np.ndarray
) -> float:
    """Beta-binomial rho by moments: sum[(b - n p)^2 - n p (1 - p)] / sum[n (n - 1) p (1 - p)], over entries of n >= 2.

    Unbiased for rho when `share` is each entry's true p: E[(b - n p)^2] =
    n p (1 - p) (1 + (n - 1) rho). An entry of one read carries no rho.
    """
    keep = trials >= 2
    n, b, p = trials[keep], successes[keep], share[keep]
    excess = ((b - n * p) ** 2 - n * p * (1 - p)).sum()
    return float(excess / (n * (n - 1) * p * (1 - p)).sum())


def problems(
    manifest_path: Path, n: int | None = None, realizations: int | None = None
) -> Iterator[KnownProblem]:
    """Each realization's problem in turn, `n` of them if given; the next draws only when asked for.

    `realizations` overrides the manifest's `[sample] realizations`, as
    `python -m port.sim.draw --seed` overrides its seed: the same clones,
    layout and phase law, more count draws.
    """
    import pandas as pd

    from port.sim import draw as d
    from port.sim.laws import allele_share

    manifest = d.read_manifest(manifest_path)
    if realizations is not None:
        from dataclasses import replace

        more = {"sample": {"realizations": int(realizations)}}
        manifest = replace(manifest, tables=d._merge(manifest.tables, more))
    tree_rng = np.random.default_rng(np.random.SeedSequence(manifest.seed).spawn(3)[0])
    tree = d.draw_tree(manifest, tree_rng)
    clones = ("normal", *manifest.tumour)
    baseline = pd.read_csv(
        manifest.resolve(manifest.reference["baseline"]), sep="\t", comment="#"
    )
    gene_chrom = baseline["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    gene_pos = ((baseline["cdsStart"] + baseline["cdsEnd"]) // 2).to_numpy()
    copies = d.clone_copies(tree, clones, gene_chrom, gene_pos)
    weights = baseline["lambda"].to_numpy()[:, None] * d._depth(
        manifest, clones, copies
    )
    kappa = float(manifest.model["dirichlet_concentration"])
    alpha = np.maximum(kappa * weights / weights.sum(axis=0, keepdims=True), 1e-300)
    _, snp_chrom, snp_pos = d._snps(manifest)
    snp_copies = d.clone_copies(tree, clones, snp_chrom, snp_pos)
    share = np.stack(
        [
            allele_share(
                snp_copies[:, k, 0].astype(np.float64),
                snp_copies[:, k, 1].astype(np.float64),
                manifest.normal_frac(clone),
                manifest.model["admixture"],
            )
            for k, clone in enumerate(clones)
        ]
    )
    rho = float(manifest.model["bb_overdispersion"])
    _, _, points = d.hex_array(
        int(manifest.array["rows"]), int(manifest.array["columns"])
    )
    indptr, indices, graph_weights = hex_graph(points)

    opened = time.perf_counter()
    for realized in d.realize(manifest, None):
        if n is not None and realized.index >= n:
            return
        drawn = time.perf_counter()
        (labels,) = realized.truth.labels
        (counts,) = realized.counts
        field = rdr_field(counts, alpha) + baf_field(
            realized.a, realized.b, realized.phase, share, rho
        )
        yield KnownProblem(
            realized.index,
            field,
            np.asarray(labels, dtype=np.int64),
            indptr,
            indices,
            graph_weights,
            BETA,
            drawn - opened,
            time.perf_counter() - drawn,
        )
        opened = time.perf_counter()
