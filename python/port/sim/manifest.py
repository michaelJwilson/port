"""A manifest of what a `run_cnaster` run's inputs were, as data.

**Proposed for `cnaster`, written here.** #116: a run says a great deal about
its inputs and records almost none of it as data, so an equivalent instance
cannot be planted from a real dataset -- only from guesses. `port` cannot land
this (`CLAUDE.md`, "Working against a repository you do not own"), so it is
written as a patch with its validation beside it.

The package already writes a structured record of how well a run recovered a
**known** truth: `scripts/run_sim_analysis.py`'s `save_validation_stats_yaml`.
This is the other half -- what the data *was* -- and it is deliberately in the
same shape.

What the log carries today is prose: percentiles, medians and an `AnnData`
repr. Six quantiles is not a law anyone can draw from, which is why every
count summary here is a **fitted family with its parameters and a fit
statistic** rather than a set of order statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MANIFEST_VERSION = 1
"""Bumped when a field changes meaning, so a consumer can refuse an old one."""

MIN_DISPERSION = 1e-6
"""Below this the negative binomial is a Poisson, and `r` is unidentifiable."""


@dataclass(frozen=True)
class FittedCounts:
    """A count distribution, named and parameterized.

    `family` is `negative_binomial` wherever the data are overdispersed and
    `poisson` where they are not, because a negative binomial fitted to
    equidispersed counts has an unbounded `r` and reports a number that means
    nothing.

    Parameters
    ----------
    family : str
        `negative_binomial` or `poisson`.
    mean, variance : float
        The moments the fit was taken from, reported so a consumer can check
        the fit rather than trust it.
    dispersion : float | None
        `alpha` in `var = mu + alpha mu^2`. `None` for a Poisson.
    dispersion_index : float
        `variance / mean`. One is Poisson; the distance from one is the whole
        reason the family is chosen rather than assumed.
    """

    family: str
    mean: float
    variance: float
    dispersion: float | None
    dispersion_index: float


def fit_counts(values: np.ndarray) -> FittedCounts:
    """Fit a count law by moments, and say which one it is.

    Moments rather than maximum likelihood on purpose: this runs over every
    spot and every gene of a real dataset, the estimate feeds a simulator
    rather than an inference, and `var = mu + alpha mu^2` inverts in closed
    form. A consumer that wants the likelihood estimate has the moments here
    to start it from.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(flat.mean())
    variance = float(flat.var(ddof=1)) if flat.size > 1 else 0.0

    if mean <= 0.0:
        return FittedCounts("poisson", mean, variance, None, float("nan"))

    dispersion = (variance - mean) / mean**2
    if dispersion <= MIN_DISPERSION:
        return FittedCounts("poisson", mean, variance, None, variance / mean)

    return FittedCounts(
        "negative_binomial", mean, variance, float(dispersion), variance / mean
    )


@dataclass(frozen=True)
class Manifest:
    """Everything needed to plant an instance of the same shape.

    Not everything needed to plant the *same* instance: the manifest carries
    laws and shapes, not the data. Two runs of a generator against one
    manifest differ, and that is what a fixture is for.
    """

    version: int = MANIFEST_VERSION
    shapes: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)
    segmentation: dict[str, Any] = field(default_factory=dict)
    clones: dict[str, Any] = field(default_factory=dict)
    emission: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, Any] = field(default_factory=dict)


def summarize_inputs(
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shapes and count laws for the three matrices a run reads.

    The allele matrices' shapes are nowhere in the log today, and their
    sparsity is what decides whether a planted instance is the same problem:
    a dense fixture over ninety per cent zeros is a different dataset.
    """
    expression = np.asarray(
        adata.layers["count"].todense()
        if hasattr(adata.layers["count"], "todense")
        else adata.layers["count"]
    )
    allele_a = _dense(cell_snp_Aallele)
    allele_b = _dense(cell_snp_Ballele)

    shapes = {
        "n_spots": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "n_snps": int(allele_a.shape[1]),
        "expression_zero_fraction": float(np.mean(expression == 0)),
        "allele_zero_fraction": float(np.mean((allele_a + allele_b) == 0)),
    }

    counts = {
        "per_spot_umi": asdict(fit_counts(expression.sum(axis=1))),
        "per_gene_umi": asdict(fit_counts(expression.sum(axis=0))),
        "per_spot_snp_umi": asdict(fit_counts((allele_a + allele_b).sum(axis=1))),
        # The pair's trial count, which is what a beta-binomial draw needs and
        # what only a per-clone mean reaches the log as.
        "snp_trials": asdict(fit_counts(allele_a + allele_b)),
    }
    return shapes, counts


def _dense(matrix: Any) -> np.ndarray:
    """A sparse or dense matrix as an array, without assuming which it is."""
    return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix)


def summarize_layout(df_gene_snp: Any, n_reference_genes: int) -> dict[str, Any]:
    """The gene and SNP layout: what matched, what did not, and how dense.

    `form_gene_snp_table` logs none of this, and it is the half of realism
    that decides whether a fixture's binning resembles a real one.
    """
    genes = df_gene_snp[df_gene_snp.is_interval]
    snps = df_gene_snp[~df_gene_snp.is_interval]

    per_gene = snps.groupby("gene").size() if len(snps) else None

    return {
        "n_reference_genes": int(n_reference_genes),
        "n_assay_genes": int(len(genes)),
        "reference_not_in_assay": int(max(0, n_reference_genes - len(genes))),
        "n_snps": int(len(snps)),
        "snps_per_gene": None
        if per_gene is None
        else asdict(fit_counts(np.asarray(per_gene, dtype=float))),
        "genes_with_no_snp": 0
        if per_gene is None
        else int(np.count_nonzero(np.asarray(per_gene) == 0)),
    }


def summarize_segmentation(lengths: np.ndarray, df_gene_snp: Any) -> dict[str, Any]:
    """The binning outcome, which `create_bin_ranges` logs nothing about."""
    lengths = np.asarray(lengths, dtype=int)

    # NB `bin_id` exists only after `create_bin_ranges`. A manifest taken
    #    before it still carries the segmentation `lengths` declares, and says
    #    nothing about what fell in each bin rather than guessing.
    if "bin_id" not in df_gene_snp.columns:
        per_bin = None
    else:
        bins = df_gene_snp[df_gene_snp.bin_id.notna()]
        per_bin = bins.groupby("bin_id").size() if len(bins) else None

    return {
        "n_chromosomes": int(lengths.size),
        "n_bins": int(lengths.sum()),
        "bins_per_chromosome": [int(x) for x in lengths],
        "rows_per_bin": None
        if per_bin is None
        else asdict(fit_counts(np.asarray(per_bin, dtype=float))),
    }


def simulation_manifest(
    *,
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
    df_gene_snp: Any,
    lengths: np.ndarray,
    n_reference_genes: int,
    clone_sizes: list[int] | None = None,
    log_mu: np.ndarray | None = None,
    p_binom: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    taus: np.ndarray | None = None,
    random_state: int = 0,
) -> Manifest:
    """Assemble the manifest from what a run already holds.

    Every argument is something `run_cnaster` has in scope at the end of a
    run; nothing here is recomputed from disk.
    """
    shapes, counts = summarize_inputs(adata, cell_snp_Aallele, cell_snp_Ballele)

    return Manifest(
        shapes=shapes,
        counts=counts,
        layout=summarize_layout(df_gene_snp, n_reference_genes),
        segmentation=summarize_segmentation(lengths, df_gene_snp),
        clones={
            "n_clones": len(clone_sizes) if clone_sizes else 0,
            "spots_per_clone": list(clone_sizes) if clone_sizes else [],
        },
        emission={
            "n_states": int(np.size(log_mu)) if log_mu is not None else 0,
            "log_mu": _listed(log_mu),
            "p_binom": _listed(p_binom),
            "alphas": _listed(alphas),
            "taus": _listed(taus),
        },
        seeds={"random_state": int(random_state)},
    )


def _listed(values: np.ndarray | None) -> list[float] | None:
    """An array as plain floats, or `None`, so the YAML carries no numpy types."""
    return (
        None if values is None else [float(x) for x in np.asarray(values).reshape(-1)]
    )


def write_simulation_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Write the manifest beside the run's other outputs."""
    destination = Path(path)
    destination.write_text(yaml.safe_dump(asdict(manifest), sort_keys=False))
    return destination


def read_simulation_manifest(path: str | Path) -> Manifest:
    """Read one back, refusing a version this code does not know."""
    loaded = yaml.safe_load(Path(path).read_text())
    version = int(loaded.get("version", 0))

    if version != MANIFEST_VERSION:
        msg = f"manifest version {version}, this reads {MANIFEST_VERSION}"
        raise ValueError(msg)

    return Manifest(**loaded)
