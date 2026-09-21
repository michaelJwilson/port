"""Generate `run_cnaster`'s inputs from a simulation manifest.

**Proposed for `cnaster`, written here.** The other half of #116: a manifest
says what a dataset was, and this turns one into a dataset of the same shape,
beside a configuration that points at it.

    python -m port.run_sim_gen manifest.yaml

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from port.simulation_manifest import Manifest, read_simulation_manifest

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


def main(argv: list[str] | None = None) -> int:
    """Read a manifest, write an instance beside it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="a simulation manifest, as YAML")
    parser.add_argument(
        "--into", default=".", help="where to write the tree; the default is here"
    )
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    arguments = parser.parse_args(argv)

    generated = generate(
        read_simulation_manifest(arguments.manifest),
        Path(arguments.into),
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
