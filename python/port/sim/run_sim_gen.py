"""Generate `run_cnaster`'s inputs from a simulation manifest.

**Proposed for `cnaster`, written here.** The other half of #116: a manifest
says what a dataset was, and this turns one into a dataset of the same shape,
beside a configuration that points at it.

    python -m port.sim.run_sim_gen manifest.yaml
    python -m port.sim.run_sim_gen sim/manifests/easy.toml

A `.toml` manifest (#382, `port.sim.toml_manifest`) is generative: it writes
one sample directory in the format of CalicoST's `sim/<name>/` -- counts,
alleles, positions and the truth -- pure or admixed as its `[model]` says. A
`.yaml` manifest (#116) is a record of a run's inputs and writes the
`run_cnaster` input tree below.

By default it writes into the **current directory** and the configuration it
writes names every path relative to that directory, so the tree can be moved
or copied and still run:

    ./spaceranger/…            the assay and the spatial positions
    ./snp/…                    barcodes, SNP ids, the two allele matrices
    ./hgtable.tsv              the reference gene table
    ./genetic_map.tab          the recombination map
    ./sample_sheet.tsv         what `load_input_data` reads first
    ./new_sim_config.yaml      what `run_cnaster` reads

**What it does not do is reproduce a dataset.** The manifest carries laws and
shapes, so two runs of this against one manifest differ, and neither is the
data that produced the manifest. That is the point: a fixture is a draw, and
what is being held fixed is the problem rather than the sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from port.sim.manifest import Manifest, read_simulation_manifest
from port.sim.toml_manifest import (
    SimManifest,
    allele_share,
    copies_at,
    read_manifest,
    segments,
    with_normal_frac,
)

GENE_SPACING = 200_000
"""Base pairs between gene intervals. Wide enough to leave each its own block."""

GENE_LENGTH = 20_000
"""Each interval's extent."""

MARKER_SPACING = 500_000
"""Base pairs between genetic-map markers."""

CENTIMORGANS_PER_MEGABASE = (0.3, 2.5)
"""The rate per marker interval. Varying, or the interpolation is exact."""

SAMPLE_ID = "S1"
FILTERED_FEATURE_NAME = "filtered_feature_bc_matrix"


@dataclass(frozen=True)
class Generated:
    """Where the generated instance was written."""

    root: Path
    config: Path
    sample_sheet: Path
    n_spots: int
    n_genes: int
    n_snps: int


def _draw_counts(law: dict[str, Any], size: tuple[int, ...], rng: Any) -> np.ndarray:
    """Draw from a fitted law, by the family it names.

    The manifest records `negative_binomial` or `poisson` and the moments the
    fit came from, so this is the inverse of `fit_counts` and nothing else: a
    consumer that wants a different family fits its own.
    """
    mean = float(law["mean"])
    if law["family"] == "poisson" or not law.get("dispersion"):
        return np.asarray(rng.poisson(max(mean, 0.0), size=size), dtype=np.int64)

    dispersion = float(law["dispersion"])
    number = 1.0 / dispersion
    probability = number / (number + mean)
    return np.asarray(rng.negative_binomial(number, probability, size=size))


def generate(manifest: Manifest, root: Path, *, seed: int | None = None) -> Generated:
    """Write a full input tree under `root`, and the configuration for it."""
    import anndata
    import scipy.sparse

    rng = np.random.default_rng(
        manifest.seeds.get("random_state", 0) if seed is None else seed
    )

    n_spots = int(manifest.shapes["n_spots"])
    n_genes = int(manifest.shapes["n_genes"])
    lengths = [int(x) for x in manifest.segmentation["bins_per_chromosome"]]
    n_bins = sum(lengths)

    snp_dir, spaceranger = root / "snp", root / "spaceranger"
    (spaceranger / "spatial").mkdir(parents=True, exist_ok=True)
    snp_dir.mkdir(parents=True, exist_ok=True)

    barcodes = [f"BC{spot:05d}-1" for spot in range(n_spots)]
    (snp_dir / "barcodes.txt").write_text("\n".join(barcodes) + "\n")

    # One gene interval per bin, and the bins laid out along the chromosomes
    # the manifest declares.
    chromosome_of, index_within = [], []
    for chromosome, extent in enumerate(lengths, start=1):
        chromosome_of += [chromosome] * extent
        index_within += list(range(extent))

    genes = [f"gene_{index:06d}" for index in range(n_genes)]
    pd.DataFrame(
        [
            {
                "name": gene,
                "name2": gene,
                "chrom": f"chr{chromosome_of[index % n_bins]}",
                "cdsStart": index_within[index % n_bins] * GENE_SPACING,
                "cdsEnd": index_within[index % n_bins] * GENE_SPACING + GENE_LENGTH,
            }
            for index, gene in enumerate(genes)
        ]
    ).set_index("name").to_csv(root / "hgtable.tsv", sep="\t")

    expression = _draw_counts(
        manifest.counts["per_gene_umi"], (n_spots, n_genes), rng
    ) // max(n_spots, 1)
    zero_fraction = float(manifest.shapes.get("expression_zero_fraction", 0.0))
    expression[rng.random((n_spots, n_genes)) < zero_fraction] = 0

    sparse_expression = scipy.sparse.csr_matrix(expression.astype(np.float32))
    adata = anndata.AnnData(
        X=sparse_expression,
        obs=pd.DataFrame(index=barcodes).assign(sample=SAMPLE_ID),
        var=pd.DataFrame(index=genes),
    )
    # NB from the local matrix rather than `adata.X`, whose type `anndata`
    #    declares as a union wide enough to include `None`.
    adata.layers["count"] = sparse_expression.copy()
    adata.write_h5ad(spaceranger / f"{FILTERED_FEATURE_NAME}.h5ad")

    n_snps = int(manifest.shapes["n_snps"])
    snp_ids = np.array(
        [f"{chromosome_of[index % n_bins]}_{1 + index}_A_G" for index in range(n_snps)]
    )
    np.save(snp_dir / "unique_snp_ids.npy", snp_ids)

    trials = _draw_counts(manifest.counts["snp_trials"], (n_spots, n_snps), rng)
    b_allele = rng.binomial(np.maximum(trials, 0), 0.5)
    scipy.sparse.save_npz(
        snp_dir / "cell_snp_Aallele.npz",
        scipy.sparse.csr_matrix(b_allele.astype(np.int32)),
    )
    scipy.sparse.save_npz(
        snp_dir / "cell_snp_Ballele.npz",
        scipy.sparse.csr_matrix((trials - b_allele).astype(np.int32)),
    )

    rows, columns = np.unravel_index(np.arange(n_spots), _lattice(n_spots))
    pd.DataFrame(
        {
            "barcode": barcodes,
            "in_tissue": 1,
            "x": rows,
            "y": columns,
            "pixel_row": rows * 100,
            "pixel_col": columns * 100,
        }
    ).to_csv(spaceranger / "spatial" / "tissue_positions.csv", index=False)

    _write_genetic_map(lengths, root / "genetic_map.tab", rng)

    sample_sheet = root / "sample_sheet.tsv"
    pd.DataFrame(
        [
            {
                "bam": "none",
                "sample_id": SAMPLE_ID,
                # Relative, so the tree moves as one.
                "spaceranger_dir": "spaceranger",
                "snp_dir": "snp",
            }
        ]
    ).to_csv(sample_sheet, index=False, sep="\t")

    config = _write_config(manifest, root)

    return Generated(root, config, sample_sheet, n_spots, n_genes, n_snps)


def _lattice(n_spots: int) -> tuple[int, int]:
    """The squarest grid holding `n_spots`, since a manifest records no geometry.

    A gap in what #116 asks for, and stated rather than hidden: the spatial
    layout of a real slide is not a rectangle, and a manifest that carried the
    coordinate extent and the neighbour-degree distribution would let this
    draw one. It does not yet.
    """
    side = int(np.ceil(np.sqrt(n_spots)))
    return side, side


def _write_genetic_map(lengths: list[int], path: Path, rng: Any) -> None:
    """A recombination map spanning every gene interval the tree carries."""
    rows: list[dict[str, object]] = []

    for chromosome, extent in enumerate(lengths, start=1):
        span = extent * GENE_SPACING + GENE_LENGTH
        positions = np.arange(0, span + MARKER_SPACING, MARKER_SPACING)
        rates = rng.uniform(*CENTIMORGANS_PER_MEGABASE, positions.size)
        centimorgans = np.concatenate(
            ([0.0], np.cumsum(rates[1:] * np.diff(positions) / 1e6))
        )
        rows += [
            {"chrom": f"chr{chromosome}", "pos": int(pos), "pos_cm": float(cm)}
            for pos, cm in zip(positions, centimorgans, strict=True)
        ]

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_config(manifest: Manifest, root: Path) -> Path:
    """`new_sim_config.yaml`, in `zenodo_sim_config.yaml`'s shape.

    Every path is relative to `root`, so the directory is the unit: copy it,
    `cd` into it, and `run_cnaster new_sim_config.yaml` runs.
    """
    n_states = int(manifest.emission.get("n_states") or 5)
    n_clones = int(manifest.clones.get("n_clones") or 2)

    config = {
        "paths": {
            "sample_sheet": "sample_sheet.tsv",
            "output_dir": "output",
            "perf_path": "cnaster.perf",
        },
        "preprocessing": {"normalidx_file": "None", "tumorprop_file": "None"},
        "visium": {"filtered_feature_name": FILTERED_FEATURE_NAME},
        "annotation": {"clone_label": "None", "clone_ranges": "None"},
        "run": {"legacy": True, "cache": False, "bafonly": False, "pause": False},
        "references": {
            "geneticmap_file": "genetic_map.tab",
            "hgtable_file": "hgtable.tsv",
            "annotation_file": "unused.gtf.gz",
            "filtergenelist_file": "None",
            "filterregion_file": "None",
        },
        "quality": {
            "phasing_min_snp_umis": 1,
            "spot_min_snp_umis": 1,
            "min_percent_expressed_spots": 0.0,
            "secondary_min_umi": 1,
            "secondary_min_snp_umi": 1,
            "secondary_min_normal_umi": 0,
            "max_binlength": 5_000_000,
            "local_outlier_filter": False,
            "filter_normal_diffexp": False,
            "normalize_gene_outliers": False,
            "normal_allele_specific_confidence": "(0.01, 0.99)",
            "min_normal_count_perbin": 1,
        },
        "phasing": {
            "run": True,
            "nu": 1.0,
            "logphase_shift": -2.0,
            "npart_phasing": 2,
            "baf_change_threshold": 0.05,
            "min_new_segment_size": 10,
            "min_prob": 1.0e-2,
        },
        "hmrf": {
            "n_clones": n_clones,
            "n_clones_rdr": n_clones,
            "min_spots_per_clone": 100,
            "min_avgumi_per_clone": 1,
            "tumorprop_threshold": 0.5,
            "max_iter_outer": 2,
            "ari_tolerance": 1.0,
            "spatial_weight": 1.0,
            "inertia": 0,
            "fixed_assignment": False,
            "unit_xsquared": 1,
            "unit_ysquared": 1,
            "random_state": int(manifest.seeds.get("random_state", 0)),
        },
        "hmm": {
            "solver": "L-BFGS-B",
            "n_states": n_states,
            "params": "smp",
            "t": 0.9999999,
            "t_phaseing": 0.99999,
            "fix_NB_dispersion": False,
            "shared_NB_dispersion": True,
            "fix_BB_dispersion": False,
            "shared_BB_dispersion": True,
            "compression_decimals": 0,
            "max_iter": 10,
            "tol": 0.001,
            "gmm_random_state": 0,
            "gmm_maxiter": 30,
            "gmm_min_binom_prob": 0.0,
            "gmm_max_binom_prob": 1.0,
            "em_maxiter": 100,
            "em_xtol": 1e-4,
            "em_ftol": 1e-4,
            "em_xrtol": 1e-4,
            "em_disp": 0,
        },
        "betabinom": {
            "run_default": False,
            "start_params": ",".join(["0.5"] * n_states),
            "start_disp": 1000.0,
        },
        "int_copy_num": {
            "rdr_weight": 0.5,
            "nonbalance_bafdist": 1.0,
            "nondiploid_rdrdist": 10.0,
            "ploidy": "diploid",
        },
    }

    path = root / "new_sim_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@dataclass(frozen=True)
class GeneratedSample:
    """Where a TOML manifest's sample was written, and what it holds."""

    path: Path
    n_spots: int
    n_genes: int
    n_snps: int
    n_segments: int
    labels: np.ndarray
    """`(n_spots,)` index into the manifest's clones."""


SPOT_CHUNK = 64
"""Spots drawn per block: 64 x 35,299 genes is 18 MB of float64."""


def generate_sample(
    manifest: SimManifest,
    into: Path | None = None,
    *,
    gene_table: Path | None = None,
) -> GeneratedSample:
    """Draw one sample from `manifest` and write it as `<into>/<name>/`.

    The generative model, per spot `s` of clone `c` with normal fraction `f`:

    - total UMI `N_s` and total SNP reads `M_s` are independent draws from
      `[coverage] spot_umi` and `spot_snp_umi` (lognormal, rounded, at
      least 1): CalicoST's two correlate at r = 0.010 and 0.002;
    - gene counts are `Poisson(N_s q_g)` with `q` normalized and
      `p_g d_g G_sg`: `p` the gene profile drawn once from
      `[coverage] gene_profile`, `d_g` the depth factor of the admixture law
      at the gene's planted `(A, B)`, and `G_sg ~ Gamma(1/alpha, alpha)` with
      `alpha = [model] nb_dispersion` (`G = 1` at 0);
    - SNP reads are `Poisson(M_s w_sj)`, `w` normalized and proportional to
      `v_j H_sj`: `v_j ~ Gamma(1/a, a)` with `a` the `[coverage] snp_total`
      dispersion, times `d_j` if `snp_depth_follows_copies`, and
      `H_sj ~ Gamma(1/b, b)` with `b = [model] snp_dispersion`;
    - the haplotype-A count of each nonzero entry is
      `BetaBinomial(n, share, rho)`, `share` the admixture law's, binomial at
      `rho = 0` and wherever the share is 0 or 1.

    Poisson rather than multinomial draws, so a spot's total is `N_s` up to
    Poisson noise, `sqrt(N_s) / N_s` = 1.8% at 3,000 UMI: 106M binomials are
    what a 35,299-way multinomial costs per 3,000 spots.

    The normal clone is drawn at `(1, 1)` with `f` irrelevant, so a pure
    sample (`f = 0`) differs from an admixed one only in the tumour spots.
    """
    rng = np.random.default_rng(manifest.seed)
    out = into if into is not None else manifest.resolve(manifest.output)
    out = out / manifest.name
    (out / "spatial").mkdir(parents=True, exist_ok=True)

    n_spots = int(manifest.size["n_spots"])
    barcodes = np.array([f"spot_{i}" for i in range(n_spots)])
    rows, cols = _hex_lattice(manifest)
    labels = _labels(manifest, barcodes, rows, cols)

    genes, gene_chrom, gene_pos = _genes(manifest, gene_table)
    snp_ids, snp_chrom, snp_pos = _snps(manifest, rng)

    _write_expression(manifest, out, rng, barcodes, labels, genes, gene_chrom, gene_pos)
    _write_alleles(manifest, out, rng, labels, snp_chrom, snp_pos)

    (out / "barcodes.txt").write_text("\n".join(barcodes))
    np.save(out / "unique_snp_ids.npy", snp_ids.astype(object))
    pd.DataFrame({0: barcodes, 1: 1, 2: rows, 3: cols, 4: rows, 5: cols}).to_csv(
        out / "spatial" / "tissue_positions_list.csv", header=False, index=False
    )
    names = np.array([c.name for c in manifest.clones])
    pd.DataFrame(
        {"labels": names[labels], "x": rows, "y": cols}, index=barcodes
    ).to_csv(out / "truth_clone_labels.tsv", sep="\t")
    profile = _profile(manifest)
    profile.to_csv(out / "truth_acn_profile.tsv", sep="\t", index=False)

    return GeneratedSample(out, n_spots, len(genes), len(snp_ids), len(profile), labels)


def _hex_lattice(manifest: SimManifest) -> tuple[np.ndarray, np.ndarray]:
    """Visium's hex packing, row-major: `x = row`, `y = 2 col + row % 2`."""
    index = np.arange(int(manifest.size["n_spots"]))
    columns = int(manifest.size["columns"])
    rows = index // columns
    return rows, 2 * (index % columns) + rows % 2


def _labels(
    manifest: SimManifest, barcodes: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Clone per spot: a truth file's, or `voronoi` from each clone's center.

    `voronoi` hands each clone, in manifest order, its `spots` nearest spots
    not yet taken, so the counts are the manifest's exactly.
    """
    source = manifest.layout.get("labels", "voronoi")
    index = {clone.name: i for i, clone in enumerate(manifest.clones)}

    if source != "voronoi":
        table = pd.read_csv(manifest.resolve(source), sep="\t", index_col=0)
        labels: np.ndarray = np.asarray(table["labels"].map(index).to_numpy())
        if table.shape[0] != barcodes.size or np.isnan(labels.astype(float)).any():
            msg = f"{source} does not label the manifest's {barcodes.size} spots"
            raise ValueError(msg)
        return np.asarray(labels, dtype=np.int64)

    labels = np.full(barcodes.size, -1, dtype=np.int64)
    points = np.column_stack([rows, cols]).astype(np.float64)
    for label, clone in enumerate(manifest.clones):
        if clone.center is None:
            msg = f"clone {clone.name} has no center for a voronoi layout"
            raise ValueError(msg)
        free = np.flatnonzero(labels < 0)
        distance = ((points[free] - np.asarray(clone.center)) ** 2).sum(axis=1)
        labels[free[np.argsort(distance, kind="stable")[: clone.spots]]] = label
    return labels


def _spread(manifest: SimManifest, count: int, rng: Any | None) -> tuple[Any, Any]:
    """`count` positions over the genome, by chromosome length.

    Evenly spaced when `rng` is None, uniform draws otherwise; sorted.
    """
    lengths = np.asarray(manifest.size["chromosome_lengths"], dtype=np.float64)
    genome = np.concatenate([[0.0], np.cumsum(lengths)])
    if rng is None:
        offsets = (np.arange(count) + 0.5) * genome[-1] / count
    else:
        offsets = np.sort(rng.uniform(0, genome[-1], count))
    chromosome = np.searchsorted(genome, offsets, side="right")
    position = (offsets - genome[chromosome - 1]).astype(np.int64)
    return chromosome.astype(str), position


def _genes(
    manifest: SimManifest, gene_table: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gene names, and each one's chromosome and midpoint (-1 where unplaced)."""
    import anndata

    n_genes = int(manifest.size["n_genes"])
    source = manifest.layout.get("genes", "synthetic")

    if source == "synthetic":
        chromosome, position = _spread(manifest, n_genes, None)
        return np.array([f"gene_{i:06d}" for i in range(n_genes)]), chromosome, position

    genes = np.asarray(anndata.read_h5ad(manifest.resolve(source)).var_names).astype(
        str
    )
    table_path = gene_table or manifest.resolve(manifest.layout["gene_table"])
    if not table_path.exists():
        msg = (
            f"gene table {table_path} not found: set $PORT_GRCH38 to CalicoST's "
            "GRCh38_resources, or pass --gene-table"
        )
        raise FileNotFoundError(msg)
    table = pd.read_csv(table_path, sep="\t", index_col=0)
    table = table.drop_duplicates("name2").set_index("name2")
    known = np.asarray(pd.Index(genes).isin(table.index))
    chromosome = np.full(genes.size, "", dtype=object)
    position = np.full(genes.size, -1, dtype=np.int64)
    rows = table.loc[genes[known]]
    chromosome[known] = rows["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    position[known] = ((rows["cdsStart"] + rows["cdsEnd"]) // 2).to_numpy()
    return genes, chromosome.astype(str), position


def _snps(manifest: SimManifest, rng: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SNP ids as CalicoST writes them, `chr_pos_N_N`, with their loci."""
    source = manifest.layout.get("snps", "synthetic")

    if source == "synthetic":
        chromosome, position = _spread(manifest, int(manifest.size["n_snps"]), rng)
        ids = np.array(
            [f"{c}_{p}_N_N" for c, p in zip(chromosome, position, strict=True)]
        )
        return ids, chromosome, position

    ids = np.load(manifest.resolve(source), allow_pickle=True).astype(str)
    chromosome = np.array([s.split("_")[0] for s in ids])
    position = np.array([int(s.split("_")[1]) for s in ids])
    return ids, chromosome, position


def _depth_factor(manifest: SimManifest, copies: np.ndarray) -> np.ndarray:
    """`(n_loci, n_clones)` read-depth factor of the admixture law; normal 1."""
    total = copies.sum(axis=2).astype(np.float64)
    factor = np.ones_like(total)
    for label, clone in enumerate(manifest.clones):
        if clone.name == "normal":
            continue
        tumour = 1.0 - manifest.normal_frac(clone.name)
        factor[:, label] = tumour * total[:, label] / 2.0 + (1.0 - tumour)
    return factor


def _lognormal_counts(manifest: SimManifest, law: str, size: int, rng: Any) -> Any:
    parameters = manifest.coverage[law].parameters
    drawn = rng.lognormal(parameters["mu"], parameters["sigma"], size)
    return np.maximum(np.rint(drawn), 1).astype(np.int64)


def _write_expression(
    manifest: SimManifest,
    out: Path,
    rng: Any,
    barcodes: np.ndarray,
    labels: np.ndarray,
    genes: np.ndarray,
    chromosome: np.ndarray,
    position: np.ndarray,
) -> None:
    """`filtered_feature_bc_matrix.h5ad`: counts as int64 CSR, `obs.labels`."""
    import anndata
    import scipy.sparse

    law = manifest.coverage["gene_profile"]
    expressed = rng.random(genes.size) < float(law.expressed or 1.0)
    profile = np.where(
        expressed,
        rng.lognormal(law.parameters["mu"], law.parameters["sigma"], genes.size),
        0.0,
    )
    placed = position >= 0
    copies = np.ones((genes.size, len(manifest.clones), 2), dtype=np.int64)
    copies[placed] = copies_at(manifest, chromosome[placed], position[placed])
    weights = profile[:, None] * _depth_factor(manifest, copies)

    depth = _lognormal_counts(manifest, "spot_umi", barcodes.size, rng)
    alpha = float(manifest.model.nb_dispersion)
    blocks = []

    for start in range(0, barcodes.size, SPOT_CHUNK):
        spots = np.arange(start, min(start + SPOT_CHUNK, barcodes.size))
        q = weights[:, labels[spots]].T
        if alpha > 0:
            q = q * rng.gamma(1.0 / alpha, alpha, q.shape)
        q = q / q.sum(axis=1, keepdims=True)
        blocks.append(scipy.sparse.csr_matrix(rng.poisson(depth[spots, None] * q)))

    counts = scipy.sparse.vstack(blocks).tocsr().astype(np.int64)
    names = np.array([c.name for c in manifest.clones])
    assay = anndata.AnnData(
        X=counts,
        obs=pd.DataFrame({"labels": names[labels]}, index=barcodes),
        var=pd.DataFrame(index=genes),
    )
    assay.write_h5ad(out / "filtered_feature_bc_matrix.h5ad")


def _write_alleles(
    manifest: SimManifest,
    out: Path,
    rng: Any,
    labels: np.ndarray,
    chromosome: np.ndarray,
    position: np.ndarray,
) -> None:
    """`cell_snp_{A,B}allele.npz`: int64 CSR, spots by SNPs, A the haplotype."""
    import scipy.sparse

    n_spots, n_snps = labels.size, position.size
    copies = copies_at(manifest, chromosome, position)
    dispersion = manifest.coverage["snp_total"].parameters.get("dispersion", 0.0)
    weights = (
        rng.gamma(1.0 / dispersion, dispersion, n_snps)
        if dispersion > 0
        else np.ones(n_snps)
    )
    follows = manifest.model.snp_depth_follows_copies
    factor = (
        _depth_factor(manifest, copies)
        if follows
        else np.ones_like(copies[..., 0], dtype=np.float64)
    )
    depth = _lognormal_counts(manifest, "spot_snp_umi", n_spots, rng)
    entry = float(manifest.model.snp_dispersion)

    blocks = []
    for start in range(0, n_spots, SPOT_CHUNK):
        spots = np.arange(start, min(start + SPOT_CHUNK, n_spots))
        w = weights[None, :] * factor[:, labels[spots]].T
        if entry > 0:
            w = w * rng.gamma(1.0 / entry, entry, w.shape)
        w = w / w.sum(axis=1, keepdims=True)
        blocks.append(scipy.sparse.csr_matrix(rng.poisson(depth[spots, None] * w)))
    trials = scipy.sparse.vstack(blocks).tocoo()

    share = np.stack(
        [
            allele_share(
                copies[:, label, 0].astype(np.float64),
                copies[:, label, 1].astype(np.float64),
                manifest.normal_frac(clone.name),
                manifest.model.admixture,
            )
            for label, clone in enumerate(manifest.clones)
        ]
    )
    p = share[labels[trials.row], trials.col]

    rho = float(manifest.model.bb_overdispersion)
    if rho > 0:
        inner = (p > 0) & (p < 1)
        scale = 1.0 / rho - 1.0
        p = p.copy()
        p[inner] = rng.beta(p[inner] * scale, (1 - p[inner]) * scale)
    hits = rng.binomial(trials.data, p)

    shape = (n_spots, n_snps)
    first = scipy.sparse.csr_matrix((hits, (trials.row, trials.col)), shape=shape)
    second = scipy.sparse.csr_matrix(
        (trials.data - hits, (trials.row, trials.col)), shape=shape
    )
    scipy.sparse.save_npz(out / "cell_snp_Aallele.npz", first.astype(np.int64))
    scipy.sparse.save_npz(out / "cell_snp_Ballele.npz", second.astype(np.int64))


def _profile(manifest: SimManifest) -> pd.DataFrame:
    """`truth_acn_profile.tsv`: `chr start end`, then `A`, `B` per clone."""
    rows = segments(manifest)
    frame = pd.DataFrame(rows, columns=["chr", "start", "end"])
    chromosome = frame["chr"].to_numpy().astype(str)
    middle = ((frame["start"] + frame["end"]) // 2).to_numpy()
    copies = copies_at(manifest, chromosome, middle)
    frame["chr"] = frame["chr"].astype(int)
    # NB CalicoST's order: tumour clones, then `normal` last.
    order = sorted(
        range(len(manifest.clones)), key=lambda i: manifest.clones[i].name == "normal"
    )
    for label in order:
        clone = manifest.clones[label]
        frame[f"{clone.name}_A_copy"] = copies[:, label, 0]
        frame[f"{clone.name}_B_copy"] = copies[:, label, 1]
    return frame


def main(argv: list[str] | None = None) -> int:
    """Read a manifest, write an instance: a sample for TOML, a tree for YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="a manifest, `.toml` (#382) or `.yaml` (#116)")
    parser.add_argument(
        "--into",
        default=None,
        help="where to write; TOML: the manifest's `output`, YAML: here",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    parser.add_argument(
        "--normal-frac",
        default=None,
        help="TOML: every tumour clone at this fraction, or `fitted`",
    )
    parser.add_argument("--name", default=None, help="TOML: override the sample name")
    parser.add_argument(
        "--gene-table", default=None, help="TOML: override `[layout] gene_table`"
    )
    arguments = parser.parse_args(argv)

    if Path(arguments.manifest).suffix == ".toml":
        manifest = read_manifest(arguments.manifest)
        if arguments.normal_frac is not None:
            manifest = with_normal_frac(manifest, arguments.normal_frac)
        if arguments.seed is not None:
            manifest = replace(manifest, seed=arguments.seed)
        if arguments.name is not None:
            manifest = replace(manifest, name=arguments.name)
        sample = generate_sample(
            manifest,
            None if arguments.into is None else Path(arguments.into),
            gene_table=None
            if arguments.gene_table is None
            else Path(arguments.gene_table),
        )
        print(
            f"wrote {sample.n_spots} spots, {sample.n_genes} genes, "
            f"{sample.n_snps} snps and {sample.n_segments} segments to {sample.path}"
        )
        return 0

    generated = generate(
        read_simulation_manifest(arguments.manifest),
        Path(arguments.into or "."),
        seed=arguments.seed,
    )
    print(
        f"wrote {generated.n_spots} spots, {generated.n_genes} genes and "
        f"{generated.n_snps} snps to {generated.root}; "
        f"run with `cd {generated.root} && run_cnaster {generated.config.name}`"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
