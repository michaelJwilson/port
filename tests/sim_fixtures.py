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
