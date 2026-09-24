"""CalicoST's simulated samples, loaded as fixtures (#362).

`sim/<name>/` holds one sample of the simulation study `cnaster`'s
`scripts/run_sim_analysis.py` scores, named by what generated it:

    numcnas{global}.{local}_cnasize{size}_ploidy{ploidy}_random{realization}

`global` CNAs are shared by every tumour clone and `local` ones by one clone
each; `size` is each event's length in base pairs. `sample_name` builds the
name from those numbers, and :data:`EASY` and :data:`HARD` are the two
samples committed here: few long events, and many short ones.

`load_simulated` reads what `run_cnaster` reads from the directory, in place,
and the truth `run_sim_analysis` reads from it (`get_sample_truth`):

- `truth_clone_labels.tsv`: `barcode label x y`, `normal` or `clone_k`,
  renumbered as `remap_clone_num` does, so `normal` is 0 and `clone_k` is
  `k + 1`;
- `truth_acn_profile.tsv`: `chr start end` and each clone's `A` and `B`
  copies over segments.

`write_sim_inputs` writes the sample sheet and the configuration beside
them: `zenodo_sim_config.yaml`, the configuration shipped for these samples,
with its paths pointed here. The gene table and genetic map are CalicoST's
`GRCh38_resources`, which the sample directory does not carry; `references`
finds them, and a test that needs them skips where they are absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
SIM_ROOT = REPOSITORY / "sim"
"""Where the committed samples live."""

INPUTS = (
    "barcodes.txt",
    "unique_snp_ids.npy",
    "cell_snp_Aallele.npz",
    "cell_snp_Ballele.npz",
    "filtered_feature_bc_matrix.h5ad",
    "spatial/tissue_positions_list.csv",
)
"""What `cnaster.io.load_input_data` reads from a sample directory."""

TRUTH = ("truth_clone_labels.tsv", "truth_acn_profile.tsv")


def _size(bases: float) -> str:
    """`5e7` as the sample names write it: mantissa, `e`, exponent."""
    mantissa, exponent = f"{bases:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def sample_name(
    n_global: int = 1,
    n_local: int = 2,
    cna_size: float = 5e7,
    ploidy: int = 2,
    realization: int = 0,
) -> str:
    """The directory name the simulation gave these parameters.

    The defaults are :data:`EASY`'s.
    """
    return (
        f"numcnas{n_global}.{n_local}_cnasize{_size(cna_size)}"
        f"_ploidy{ploidy}_random{realization}"
    )


EASY = sample_name(n_global=1, n_local=2, cna_size=5e7, ploidy=2, realization=0)
"""One shared and two clone-specific events per clone, 50 Mb each."""

HARD = sample_name(n_global=6, n_local=3, cna_size=1e7, ploidy=2, realization=0)
"""Six shared and three clone-specific events per clone, 10 Mb each."""


@dataclass(frozen=True)
class SimulatedSample:
    """One sample directory, with its truth read."""

    name: str
    path: Path
    barcodes: np.ndarray
    """`(n_spots,)` in `truth_clone_labels.tsv`'s order."""
    labels: np.ndarray
    """`(n_spots,)` planted clone, `normal` as 0 and `clone_k` as `k + 1`."""
    coords: np.ndarray
    """`(n_spots, 2)` the truth file's `x`, `y`."""
    clones: tuple[str, ...]
    """The truth's clone names, indexed by label."""
    profile: pd.DataFrame
    """`chr start end`, then `A` and `B` per clone name, one row per segment."""

    @property
    def n_clones(self) -> int:
        return len(self.clones)

    def copies_at(self, chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
        """`(n_positions, n_clones, 2)` planted `(A, B)` at each position.

        A position no segment covers is `-1`.
        """
        chrom = self.profile["chr"].astype(str).str.removeprefix("chr").to_numpy()
        starts = self.profile["start"].to_numpy()
        ends = self.profile["end"].to_numpy()
        found = np.full(position.size, -1)
        query = np.asarray(chromosome).astype(str)

        for row in range(len(self.profile)):
            covered = (query == chrom[row]) & (position >= starts[row])
            covered &= position < ends[row]
            found[covered] = row

        copies = np.full((position.size, self.n_clones, 2), -1, dtype=np.int64)
        kept = found >= 0

        for label, clone in enumerate(self.clones):
            for allele, column in enumerate(("A", "B")):
                values = self.profile[f"{clone}_{column}_copy"].to_numpy()
                copies[kept, label, allele] = values[found[kept]]

        return copies


def _clone_order(names: pd.Series) -> tuple[str, ...]:
    """`normal` first, then `clone_k` by `k`: `remap_clone_num`'s order."""
    others = sorted(
        (n for n in names.unique() if n != "normal"),
        key=lambda n: int(n.removeprefix("clone_")),
    )
    return ("normal", *others)


def load_simulated(name: str = EASY, root: Path = SIM_ROOT) -> SimulatedSample:
    """Read one sample's truth, and check it carries the inputs `cnaster` reads."""
    path = root / name
    missing = [f for f in (*INPUTS, *TRUTH) if not (path / f).exists()]

    if missing:
        msg = f"{path} is missing {missing}"
        raise FileNotFoundError(msg)

    table = pd.read_csv(
        path / "truth_clone_labels.tsv",
        sep="\t",
        names=["barcode", "clone", "x", "y"],
        skiprows=1,
    )
    clones = _clone_order(table["clone"])
    index = {clone: label for label, clone in enumerate(clones)}
    profile = pd.read_csv(path / "truth_acn_profile.tsv", sep="\t")

    return SimulatedSample(
        name=name,
        path=path,
        barcodes=table["barcode"].to_numpy(),
        labels=table["clone"].map(index).to_numpy(dtype=np.int64),
        coords=table[["x", "y"]].to_numpy(dtype=np.float64),
        clones=clones,
        profile=profile,
    )


RESOURCE_FILES = ("hgTables_hg38_gencode.txt", "genetic_map_GRCh38_merged.tab.gz")


def references() -> Path | None:
    """CalicoST's `GRCh38_resources`: `$PORT_GRCH38`, else the uv git checkout."""
    candidates = [os.environ.get("PORT_GRCH38", "")]
    candidates += sorted(
        str(p)
        for p in (Path.home() / ".cache/uv/git-v0/checkouts").glob(
            "*/*/GRCh38_resources"
        )
    )

    for candidate in candidates:
        if candidate and all((Path(candidate) / f).exists() for f in RESOURCE_FILES):
            return Path(candidate)

    return None


ZENODO_CONFIG = Path(__file__).parent / "data" / "zenodo_sim_config.yaml"


def sim_config(sample: SimulatedSample, root: Path, resources: Path) -> dict[str, Any]:
    """`zenodo_sim_config.yaml`, pointed at `sample` and writing under `root`."""
    document: dict[str, Any] = yaml.safe_load(ZENODO_CONFIG.read_text())
    document["paths"] = {
        "sample_sheet": str(root / "sample_sheet.tsv"),
        "output_dir": str(root / "output"),
        "perf_path": str(root / "cnaster.perf"),
    }
    document["references"].update(
        {
            "hgtable_file": str(resources / "hgTables_hg38_gencode.txt"),
            "geneticmap_file": str(resources / "genetic_map_GRCh38_merged.tab.gz"),
            "annotation_file": str(root / "unused.gtf.gz"),
            "filtergenelist_file": str(resources / "ig_gene_list.txt"),
            "filterregion_file": str(resources / "HLA_regions.bed"),
        }
    )
    return document


def write_sim_inputs(
    sample: SimulatedSample,
    root: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write the sample sheet and configuration for `sample`; return the config.

    `overrides` maps `section.key` to a value, as `recovery_audit --set` does.
    """
    resources = references()

    if resources is None:
        msg = "CalicoST's GRCh38_resources not found; set $PORT_GRCH38"
        raise FileNotFoundError(msg)

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "bam": ["unused.bam"],
            "sample_id": [sample.name],
            "spaceranger_dir": [str(sample.path)],
            "snp_dir": [str(sample.path)],
        }
    ).to_csv(root / "sample_sheet.tsv", sep="\t", index=False)

    document = sim_config(sample, root, resources)

    for key, value in (overrides or {}).items():
        section, _, name = key.partition(".")
        document[section][name] = value

    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(document))
    return config


PURE_SEED = 362
"""The draw :func:`purify` makes, so a pure sample is one fixture, not many."""


def _gene_copies(
    sample: SimulatedSample, genes: np.ndarray, resources: Path
) -> np.ndarray:
    """`(n_genes, n_clones)` planted `A + B` at each gene's midpoint; 2 off-table."""
    table = pd.read_csv(resources / RESOURCE_FILES[0], sep="\t", index_col=0)
    table = table.drop_duplicates("name2").set_index("name2")
    known = np.isin(genes, table.index)
    total = np.full((genes.size, sample.n_clones), 2, dtype=np.int64)
    rows = table.loc[genes[known]]
    chromosome = rows["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    middle = ((rows["cdsStart"] + rows["cdsEnd"]) // 2).to_numpy()
    copies = sample.copies_at(chromosome, middle)
    covered = copies[:, 0, 0] >= 0
    placed = np.flatnonzero(known)[covered]
    total[placed] = copies[covered].sum(axis=2)
    return total


def purify(sample: SimulatedSample, root: Path, seed: int = PURE_SEED) -> Path:
    """Write `sample` with every tumour spot pure; return the new directory.

    The simulated spots carry about 8 per cent normal admixture: at planted
    LOH the phased pseudobulk BAF is 0.072 to 0.082 rather than 0, the same
    in every clone. This redraws each tumour spot as pure tumour at its
    planted copies and keeps everything else:

    - read depth: the spot's total UMI (its exposure) is kept and its genes
      are a multinomial draw over the normal spots' pooled profile scaled by
      the planted `(A + B) / 2` at each gene;
    - alleles: each SNP's total reads (its trials) are kept and the
      haplotype-A count is `Binomial(n, A / (A + B))`, 0.5 where `A + B` is
      0;
    - normal spots, positions, barcodes and the truth: unchanged.
    """
    import anndata as ad
    import scipy.sparse as sp

    resources = references()

    if resources is None:
        msg = "CalicoST's GRCh38_resources not found; set $PORT_GRCH38"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)
    out = root / f"{sample.name}_pure"
    (out / "spatial").mkdir(parents=True, exist_ok=True)

    for name in (*INPUTS, *TRUTH):
        if not name.startswith("cell_snp") and not name.endswith(".h5ad"):
            (out / name).write_bytes((sample.path / name).read_bytes())

    by_barcode = dict(zip(sample.barcodes.astype(str), sample.labels, strict=True))

    assay = ad.read_h5ad(sample.path / "filtered_feature_bc_matrix.h5ad")
    labels = np.array([by_barcode[b] for b in assay.obs_names.astype(str)])
    counts = sp.csr_matrix(assay.X)
    genes = np.asarray(assay.var_names).astype(str)
    total = _gene_copies(sample, genes, resources)
    normal = np.asarray(counts[labels == 0].sum(axis=0)).ravel().astype(np.float64)
    depth = np.asarray(counts.sum(axis=1)).ravel()
    rows = []

    for spot in range(counts.shape[0]):
        clone = int(labels[spot])

        if clone == 0:
            rows.append(counts[spot])
            continue

        weights = normal * total[:, clone] / 2.0
        drawn = rng.multinomial(int(depth[spot]), weights / weights.sum())
        rows.append(sp.csr_matrix(drawn[None, :]))

    assay.X = sp.vstack(rows).tocsr().astype(counts.dtype)
    assay.write_h5ad(out / "filtered_feature_bc_matrix.h5ad")

    snps = np.load(sample.path / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    chromosome = np.array([s.split("_")[0].removeprefix("chr") for s in snps])
    position = np.array([int(s.split("_")[1]) for s in snps])
    copies = sample.copies_at(chromosome, position)
    barcodes = (sample.path / "barcodes.txt").read_text().split()
    spot_labels = np.array([by_barcode[b] for b in barcodes])
    first = sp.load_npz(sample.path / "cell_snp_Aallele.npz").tocsr()
    second = sp.load_npz(sample.path / "cell_snp_Ballele.npz").tocsr()
    trials = (first + second).tocoo()
    clone = spot_labels[trials.row]
    pair = copies[trials.col, clone]
    tumour = (clone > 0) & (pair[:, 0] >= 0)
    share = np.where(
        pair.sum(axis=1) > 0, pair[:, 0] / np.maximum(pair.sum(axis=1), 1), 0.5
    )
    a_count = np.asarray(first[trials.row, trials.col]).ravel()
    a_count = np.where(tumour, rng.binomial(trials.data, share), a_count)
    shape = first.shape
    new_a = sp.csr_matrix((a_count, (trials.row, trials.col)), shape=shape)
    new_b = sp.csr_matrix(
        (trials.data - a_count, (trials.row, trials.col)), shape=shape
    )
    sp.save_npz(out / "cell_snp_Aallele.npz", new_a.astype(first.dtype))
    sp.save_npz(out / "cell_snp_Ballele.npz", new_b.astype(second.dtype))

    return out


DECORRELATED_KEEP = 0.2
"""The share of tested genes :func:`decorrelate` keeps (#372)."""

ADMIXED_PURITY = 0.92
"""The simulated tumour spots' tumour fraction (:func:`purify`: about 8 per cent normal)."""


@dataclass(frozen=True)
class GeneAgreement:
    """Per tested gene, how far the tumour clones' expression is from baseline x copy number."""

    genes: np.ndarray
    """`(n_genes,)` column indices into the assay of the tested genes."""
    offset: np.ndarray
    """`(n_genes,)` mean over tumour clones of `log(x / e)`, less its median: the shared tumour program."""
    realized: np.ndarray
    """`(n_genes, n_tumour)` each tumour clone's share of its pseudobulk."""
    expected: np.ndarray
    """`(n_genes, n_tumour)` the normal baseline times the planted copy factor, normalized."""


def gene_agreement(
    counts: Any, labels: np.ndarray, total: np.ndarray, purity: float = ADMIXED_PURITY
) -> GeneAgreement:
    """Tumour pseudobulk against normal baseline x planted copy factor, per tested gene.

    `total` is `(n_genes, n_clones)` planted `A + B`. A gene is tested when the
    normal pseudobulk holds at least 20 counts and every tumour clone at
    least 5, so the log ratio is not Poisson noise. The expected share is
    `b_g f_gc`, normalized per clone, with `f = rho (A + B) / 2 + 1 - rho`.

    The offset is centred on its median. A program that raises some genes
    raises the clone's total, so every gene that follows the model sits at
    one common negative offset rather than at 0; uncentred, `|o_g|` near 0
    selects program genes (`tests/test_decorrelated_fixtures.py`).
    """
    n_clones = total.shape[1]
    normal = np.asarray(counts[labels == 0].sum(axis=0)).ravel().astype(np.float64)
    bulk = np.stack(
        [
            np.asarray(counts[labels == c].sum(axis=0)).ravel()
            for c in range(1, n_clones)
        ],
        axis=1,
    ).astype(np.float64)
    tested = np.flatnonzero((normal >= 20) & (bulk.min(axis=1) >= 5))
    factor = purity * total[tested, 1:] / 2.0 + 1.0 - purity
    expected = normal[tested, None] * factor
    expected /= expected.sum(axis=0)
    realized = bulk[tested] / bulk[tested].sum(axis=0)
    offset = np.log(realized / expected).mean(axis=1)
    offset -= np.median(offset)
    return GeneAgreement(tested, offset, realized, expected)


def decorrelate(
    sample: SimulatedSample, root: Path, keep: float = DECORRELATED_KEEP
) -> Path:
    """Write `sample` with its decorrelated genes zeroed; return the new directory.

    Of the genes :func:`gene_agreement` tests, the `keep` share with the
    smallest shared tumour offset `|o_g|` is kept, and every other tested gene
    is zeroed in every spot, normal spots included, as if never measured.
    Untested genes, the SNP counts, positions and the truth are unchanged.
    The selection reads the truth, so this is a fixture, not a method: it is
    where the read-depth model holds by construction (#372).
    """
    import anndata as ad
    import scipy.sparse as sp

    resources = references()

    if resources is None:
        msg = "CalicoST's GRCh38_resources not found; set $PORT_GRCH38"
        raise FileNotFoundError(msg)

    out = root / f"{sample.name}_decorrelated"
    (out / "spatial").mkdir(parents=True, exist_ok=True)

    for name in (*INPUTS, *TRUTH):
        if not name.endswith(".h5ad"):
            (out / name).write_bytes((sample.path / name).read_bytes())

    by_barcode = dict(zip(sample.barcodes.astype(str), sample.labels, strict=True))
    assay = ad.read_h5ad(sample.path / "filtered_feature_bc_matrix.h5ad")
    labels = np.array([by_barcode[b] for b in assay.obs_names.astype(str)])
    counts = sp.csc_matrix(assay.X)
    genes = np.asarray(assay.var_names).astype(str)
    agreement = gene_agreement(counts, labels, _gene_copies(sample, genes, resources))
    order = np.argsort(np.abs(agreement.offset), kind="stable")
    dropped = agreement.genes[order[int(round(keep * order.size)) :]]
    scale = np.ones(genes.size)
    scale[dropped] = 0.0
    zeroed = (counts @ sp.diags(scale)).tocsr()
    zeroed.eliminate_zeros()
    assay.X = zeroed.astype(counts.dtype)
    assay.write_h5ad(out / "filtered_feature_bc_matrix.h5ad")

    return out
