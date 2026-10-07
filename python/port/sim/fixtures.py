"""CalicoST's simulated samples, loaded as fixtures (#362).

`sim/<name>/` holds one sample of the simulation study `cnaster`'s
`scripts/run_sim_analysis.py` scores, named by what generated it:

    numcnas{global}.{local}_cnasize{size}_ploidy{ploidy}_random{realization}

`global` CNAs are shared by every tumour clone and `local` ones by one clone
each; `size` is each event's length in base pairs. `sample_name` builds the
name from those numbers, and :data:`EASY` and :data:`HARD` are the two
samples committed here: few long events, and many short ones.

The text inputs, the SNP ids and the truth are committed gzip-compressed
(`port.sim.files`, #460); the AnnData and allele matrices already are.
`load_simulated` reads the truth `run_sim_analysis` reads (`get_sample_truth`)
from the compressed files:

- `truth_clone_labels.tsv`: `barcode label x y`, `normal` or `clone_k`,
  renumbered as `remap_clone_num` does, so `normal` is 0 and `clone_k` is
  `k + 1`;
- `truth_acn_profile.tsv`: `chr start end` and each clone's `A` and `B`
  copies over segments.

`write_sim_inputs` stages what `run_cnaster` reads under the run's root --
the compressed inputs decompressed, the rest linked -- and writes the sample
sheet, pointed at the stage, and the configuration: `zenodo_sim_config.yaml`, the configuration shipped for these samples,
with its paths pointed here. The gene table and genetic map are CalicoST's
`GRCh38_resources`, which the sample directory does not carry; `references`
finds them, and a test that needs them skips where they are absent.

`realization_hash` names a sample by its content and `r0` draws `dev_tree`'s
realization 0, refused unless it hashes to `R0_HASH`: the fixture identity
the ledger records (#588, #595). Moved from `tests.sim_fixtures` and
`tests.sim_stages` (T- #673 G6), so a shipped audit can load and name its
samples; `tests.sim_stages.realization_hash`, the name `definitions.tsv`
cites, is this function imported.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from port.sim.files import decompress, load_ids, located, read_bytes, truth_labels
from port.sim.inputs import reference_files, run_paths

REPOSITORY = Path(__file__).resolve().parents[3]
"""The checkout: the committed samples and their configuration are its files."""
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
"""What `cnaster.io.load_input_data` reads from a sample directory, by these
names; committed as `<name>.gz` where the format is not compressed already."""

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


SAMPLES = {"easy": EASY, "hard": HARD}
"""The committed samples by the short names the audits and the ledger take."""


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


DRAWN_INPUTS = (
    "sample_sheet.tsv",
    "snp/barcodes.txt",
    "snp/unique_snp_ids.npy",
    "snp/cell_snp_Aallele.npz",
    "snp/cell_snp_Ballele.npz",
)
"""What a sample `port.sim.draw` wrote holds beside its slices (#445)."""

SLICE_INPUTS = ("filtered_feature_bc_matrix.h5ad", "spatial/tissue_positions_list.csv")


def _missing_inputs(path: Path) -> list[str]:
    """Inputs `cnaster` reads that `path` lacks, in either layout.

    CalicoST's: every input in `path`. `port.sim.draw`'s: the SNPs under
    `snp/` and one directory per slice, named by its `sample_id`.
    """
    if not (path / "snp").is_dir():
        return [f for f in (*INPUTS, *TRUTH) if not located(path / f).exists()]

    missing = [f for f in (*DRAWN_INPUTS, *TRUTH) if not (path / f).exists()]
    if (path / "sample_sheet.tsv").exists():
        sheet = pd.read_csv(path / "sample_sheet.tsv", sep="\t", dtype=str)
        missing += [
            f"{sid}/{f}"
            for sid in sheet["sample_id"]
            for f in SLICE_INPUTS
            if not (path / sid / f).exists()
        ]
    return missing


def load_simulated(name: str = EASY, root: Path = SIM_ROOT) -> SimulatedSample:
    """Read one sample's truth, and check it carries the inputs `cnaster` reads.

    Either layout: CalicoST's, whose truth is `barcode label x y`, or
    `port.sim.draw`'s (#445), whose truth adds `sample_id` and whose barcodes
    are `{barcode}_{sample_id}` over every slice.
    """
    path = root / name
    missing = _missing_inputs(path)

    if missing:
        msg = f"{path} is missing {missing}"
        raise FileNotFoundError(msg)

    table = truth_labels(path).reset_index()
    table.columns = ["barcode", "clone", "x", "y", *table.columns[4:]]
    clones = _clone_order(table["clone"])
    index = {clone: label for label, clone in enumerate(clones)}
    profile = pd.read_csv(located(path / "truth_acn_profile.tsv"), sep="\t")

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


ZENODO_CONFIG = REPOSITORY / "tests" / "data" / "zenodo_sim_config.yaml"


def sim_config(root: Path, resources: Path) -> dict[str, Any]:
    """`zenodo_sim_config.yaml`, reading the sample sheet under `root` and writing there."""
    document: dict[str, Any] = yaml.safe_load(ZENODO_CONFIG.read_text())
    document["paths"] = run_paths(root)
    document["references"].update(
        reference_files(
            resources,
            genetic_map="genetic_map_GRCh38_merged.tab.gz",
            gene_table="hgTables_hg38_gencode.txt",
            filter_genes="ig_gene_list.txt",
            filter_regions="HLA_regions.bed",
        ),
        annotation_file=str(root / "unused.gtf.gz"),
    )
    return document


def stage(sample: SimulatedSample, into: Path) -> Path:
    """`into/<name>/` holding `INPUTS` by name: a compressed one decompressed.

    `cnaster` opens its inputs by these names, so a `.gz` is written out
    plain; one committed in its own compressed format is linked.
    """
    # NB `Path(sample.name).name`: a sample loaded by absolute path carries
    #    that path as its name, and `into / "/abs"` is `/abs` itself, which
    #    unlinked every committed input and linked it to itself (#492).
    target = into / Path(sample.name).name

    if target.resolve() == sample.path.resolve():
        msg = f"staging {sample.path} into itself"
        raise ValueError(msg)

    for name in INPUTS:
        source = located(sample.path / name)
        if source.name.endswith(".gz"):
            decompress(source, target / name)
        else:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).unlink(missing_ok=True)
            (target / name).symlink_to(source.resolve())
    return target


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
    staged = stage(sample, root / "inputs")
    pd.DataFrame(
        {
            "bam": ["unused.bam"],
            "sample_id": [sample.name],
            "spaceranger_dir": [str(staged)],
            "snp_dir": [str(staged)],
        }
    ).to_csv(root / "sample_sheet.tsv", sep="\t", index=False)

    document = sim_config(root, resources)

    for key, value in (overrides or {}).items():
        section, _, name = key.partition(".")
        document[section][name] = value

    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(document))
    return config


def crop(
    sample: SimulatedSample, root: Path, window: tuple[float, float, float, float]
) -> Path:
    """Write the spots of `sample` inside `window`; return the new directory.

    `window` is `(x0, x1, y0, y1)`, half-open, in the truth file's
    coordinates, so the subset is one contiguous block of the array and the
    spatial neighbourhoods inside it are the original ones. Every per-spot
    input is subset in its own row order; the SNP list, the genes and the
    copy-number truth are unchanged. Refuses a window that loses a planted
    clone, the normal one included.
    """
    import anndata as ad
    import scipy.sparse as sp

    x0, x1, y0, y1 = window
    x, y = sample.coords[:, 0], sample.coords[:, 1]
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    kept = set(sample.barcodes[inside].astype(str))
    lost = sorted(set(range(sample.n_clones)) - set(sample.labels[inside].tolist()))

    if lost:
        msg = f"window {window} drops clones {[sample.clones[c] for c in lost]}"
        raise ValueError(msg)

    out = root / (sample.name + "_crop_" + "_".join(f"{v:g}" for v in window))
    if (out / ".complete").exists():
        return out
    (out / "spatial").mkdir(parents=True, exist_ok=True)
    (out / "unique_snp_ids.npy").write_bytes(
        read_bytes(sample.path / "unique_snp_ids.npy")
    )
    (out / "truth_acn_profile.tsv").write_bytes(
        read_bytes(sample.path / "truth_acn_profile.tsv")
    )

    barcodes = read_bytes(sample.path / "barcodes.txt").decode().split()
    rows = np.array([b in kept for b in barcodes])
    (out / "barcodes.txt").write_text(
        "\n".join(b for b, k in zip(barcodes, rows, strict=True) if k) + "\n"
    )

    for name in ("cell_snp_Aallele.npz", "cell_snp_Ballele.npz"):
        matrix = sp.load_npz(sample.path / name).tocsr()
        sp.save_npz(out / name, matrix[rows])

    assay = ad.read_h5ad(sample.path / "filtered_feature_bc_matrix.h5ad")
    assay[assay.obs_names.astype(str).isin(kept)].copy().write_h5ad(
        out / "filtered_feature_bc_matrix.h5ad"
    )

    positions = sample.path / "spatial" / "tissue_positions_list.csv"
    lines = read_bytes(positions).decode().splitlines()
    (out / "spatial" / "tissue_positions_list.csv").write_text(
        "\n".join(line for line in lines if line.split(",")[0] in kept) + "\n"
    )

    truth = read_bytes(sample.path / "truth_clone_labels.tsv").decode().splitlines()
    (out / "truth_clone_labels.tsv").write_text(
        "\n".join(
            [truth[0], *(line for line in truth[1:] if line.split("\t")[0] in kept)]
        )
        + "\n"
    )
    (out / ".complete").touch()
    return out


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


def purify(
    sample: SimulatedSample,
    root: Path,
    seed: int = PURE_SEED,
    normal: tuple[float, ...] = (),
) -> Path:
    """Write `sample` with every tumour spot pure; return the new directory.

    With `normal`, tumour clone `c`'s spots are instead `normal[c - 1]`
    normal, a planted fraction per clone that a fit can be asked to recover:
    depth `(1 - f) (A + B) / 2 + f` and share `((1 - f) A + f) / ((1 - f)
    (A + B) + 2 f)` in place of the pure `(A + B) / 2` and `A / (A + B)`.

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
    suffix = "_".join(f"{f:g}" for f in normal)
    out = root / (f"{sample.name}_normal_{suffix}" if normal else f"{sample.name}_pure")
    if (out / ".complete").exists():
        return out
    fraction = np.array([0.0, *normal])
    (out / "spatial").mkdir(parents=True, exist_ok=True)

    for name in (*INPUTS, *TRUTH):
        if not name.startswith("cell_snp") and not name.endswith(".h5ad"):
            (out / name).write_bytes(read_bytes(sample.path / name))

    by_barcode = dict(zip(sample.barcodes.astype(str), sample.labels, strict=True))

    assay = ad.read_h5ad(sample.path / "filtered_feature_bc_matrix.h5ad")
    labels = np.array([by_barcode[b] for b in assay.obs_names.astype(str)])
    counts = sp.csr_matrix(assay.X)
    genes = np.asarray(assay.var_names).astype(str)
    total = _gene_copies(sample, genes, resources)
    baseline = np.asarray(counts[labels == 0].sum(axis=0)).ravel()
    baseline = baseline.astype(np.float64)
    depth = np.asarray(counts.sum(axis=1)).ravel()
    rows = []

    for spot in range(counts.shape[0]):
        clone = int(labels[spot])

        if clone == 0:
            rows.append(counts[spot])
            continue

        f = fraction[clone] if clone < fraction.size else 0.0
        weights = baseline * ((1.0 - f) * total[:, clone] / 2.0 + f)
        drawn = rng.multinomial(int(depth[spot]), weights / weights.sum())
        rows.append(sp.csr_matrix(drawn[None, :]))

    assay.X = sp.vstack(rows).tocsr().astype(counts.dtype)
    assay.write_h5ad(out / "filtered_feature_bc_matrix.h5ad")

    snps = load_ids(sample.path / "unique_snp_ids.npy")
    chromosome = np.array([s.split("_")[0].removeprefix("chr") for s in snps])
    position = np.array([int(s.split("_")[1]) for s in snps])
    copies = sample.copies_at(chromosome, position)
    barcodes = read_bytes(sample.path / "barcodes.txt").decode().split()
    spot_labels = np.array([by_barcode[b] for b in barcodes])
    first = sp.load_npz(sample.path / "cell_snp_Aallele.npz").tocsr()
    second = sp.load_npz(sample.path / "cell_snp_Ballele.npz").tocsr()
    trials = (first + second).tocoo()
    clone = spot_labels[trials.row]
    pair = copies[trials.col, clone]
    tumour = (clone > 0) & (pair[:, 0] >= 0)
    admixed = np.where(
        clone < fraction.size, fraction[np.minimum(clone, fraction.size - 1)], 0.0
    )
    alleles = (1.0 - admixed) * pair[:, 0] + admixed
    copies_total = (1.0 - admixed) * pair.sum(axis=1) + 2.0 * admixed
    share = np.clip(
        np.where(copies_total > 0, alleles / np.maximum(copies_total, 1e-12), 0.5),
        0.0,
        1.0,
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
    (out / ".complete").touch()

    return out


R0 = SIM_ROOT / "generated" / "dev_tree" / "r0"
R0_MANIFEST = "sim/manifests/baseline/dev_tree.toml"
R0_HASH = "3381575a"
"""`dev_tree` r0 at CalicoST's 60 x 50 array per slice (#470): 6,000 spots.
The exponential-length generation every cached stage and r0 figure was measured
on, frozen when the live `dev_tree.toml` moved to lognormal lengths (#619); a
live draw writes the same directory and is refused here by its hash."""


def realization_hash(path: Path) -> str:
    """The first 8 hex of SHA-256 over a realization's files, names and decoded bytes.

    A file is keyed by its name with `.gz` stripped and hashed over its
    decompressed bytes, so how a file is stored cannot move the hash (#595):
    a sample with no `.gz` hashes as its stored bytes. A name present both
    plain and `.gz` counts once where the two decode equal, and is refused
    where they differ.

    Left out: the three files that record absolute paths, and what a run or
    a plot writes into the directory.
    """
    skipped = {"config.yaml", "manifest.json", "sample_sheet.tsv"}
    digest = hashlib.sha256()

    for directory, names, files in os.walk(path):
        names[:] = sorted(n for n in names if n not in {"output", "qa"})
        decoded: dict[str, bytes] = {}

        for name in files:
            stored = Path(directory, name)
            key = stored.relative_to(path).as_posix().removesuffix(".gz")
            data = stored.read_bytes()
            data = gzip.decompress(data) if name.endswith(".gz") else data

            if key in decoded and decoded[key] != data:
                msg = f"{stored}: plain and .gz copies of {key} decode differently"
                raise ValueError(msg)
            decoded[key] = data

        for key in sorted(decoded):
            if key in skipped:
                continue

            digest.update(key.encode())
            digest.update(decoded[key])

    return digest.hexdigest()[:8]


def r0() -> Path:
    """`dev_tree`'s realization 0, drawn if absent, refused if not `R0_HASH`."""
    if not (R0 / "truth_clone_labels.tsv").is_file():
        subprocess.run(
            [sys.executable, "-m", "port.sim.draw", R0_MANIFEST],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        )

    found = realization_hash(R0)

    if found != R0_HASH:
        msg = (
            f"{R0} hashes to {found}, not {R0_HASH}: the manifest or the "
            "simulator changed, so every cached stage of r0 is stale"
        )
        raise RuntimeError(msg)

    return R0
