"""Write a pre-image to the files `cnaster.io.load_input_data` reads (#68).

`tests/unsegment.py` takes a binned fixture back to genes and blocks. This
takes that pre-image the rest of the way: to a sample sheet, a barcode list, a
SNP id array, two sparse allele matrices, an `AnnData` and a spatial table --
the files `run_cnaster` is pointed at.

The round trip is then over the real entry point rather than over one function
of it: the counts go out as files and come back as arrays, and what comes back
is compared to what went out.

**What each file has to be, learned by driving the loader rather than by
reading it.** Each of these was a `RuntimeError`, an `AttributeError` or an
assertion until it was right, and they are recorded because none is documented:

*   the sample sheet is **whitespace**-separated (`sep=r"\\s+"`), not comma,
    and needs `bam`, `sample_id`, `spaceranger_dir`, `snp_dir`;
*   spatial coordinates come from `spatial/tissue_positions.csv`, whose header
    is *replaced* by `("barcode", "in_tissue", "x", "y", "pixel_row",
    "pixel_col")` -- so the file's own column names are ignored and only the
    order matters;
*   `adata.X` must be **sparse**: `get_spaceranger_counts` calls `.toarray()`
    on it without checking, so a dense matrix raises `AttributeError`;
*   the loader reads `visium.filtered_feature_name` and
    `quality.local_outlier_filter` and `quality.normalize_gene_outliers` from
    the global config, none of which `run_core_inference` needs.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tests.fixtures import CoreInferenceTruth
from tests.unsegment import Unsegmented

GENE_SPACING = 200_000
"""Base pairs between one planted bin's gene interval and the next.

Wide enough that `assign_initial_blocks` -- which merges **overlapping** gene
intervals -- leaves each bin its own block, and well inside
`create_bin_ranges`' 5 Mb cap. It is what makes `cnaster`'s own binning land
on the planted partition, which is the thing #92 asks to be asserted rather
than assumed.
"""

GENE_LENGTH = 20_000
"""Each interval's extent. Any value below the spacing works; this is a gene."""

MARKER_SPACING = 500_000
"""Base pairs between genetic-map markers.

Real maps are denser than this; what matters for the interpolation
`assign_centiMorgans` does is that a marker interval spans several gene
intervals, so a position lands strictly between two rows rather than on one.
"""

CENTIMORGANS_PER_MEGABASE = (0.3, 2.5)
"""The rate a marker interval is drawn from, in cM/Mb.

The genome averages about one, and the spread is what makes the map a map
rather than a straight line: a constant rate would make `assign_centiMorgans`
an affine function of position and `compute_numbat_phase_switch_prob` a
function of distance alone, which is the degenerate case the interpolation is
there to handle.
"""

SAMPLE_ID = "S1"
"""The one slice's id; `sample_label` writes `S1`..`Sk` for several (#328)."""

FILTERED_FEATURE_NAME = "filtered_feature_bc_matrix"
"""What `visium.filtered_feature_name` has to name for the `.h5ad` to be found."""


@dataclass(frozen=True)
class WrittenInputs:
    """Where a pre-image was written, and what it should load back as."""

    root: Path
    sample_sheet: Path
    hgtable: Path
    barcodes: list[str]
    snp_ids: np.ndarray
    allele_a: np.ndarray
    allele_b: np.ndarray
    gene_counts: np.ndarray
    genetic_map: Path

    def config(self) -> dict[str, Any]:
        """The global config the loader reads, and nothing beyond it."""
        return {
            "paths": {
                "sample_sheet": str(self.sample_sheet),
                "output_dir": str(self.root / "output"),
            },
            # `run.legacy` selects the tab-separated gene table over a GTF;
            # `run.cache` off, or a second call reads the first one's pickle
            # and the round trip measures the cache.
            "run": {"legacy": True, "cache": False},
            "visium": {"filtered_feature_name": FILTERED_FEATURE_NAME},
            "quality": {
                "local_outlier_filter": False,
                "normalize_gene_outliers": False,
            },
        }


def write_genetic_map(truth: CoreInferenceTruth, path: Path) -> Path:
    """A recombination map over the planted chromosomes, at `path`.

    `get_reference_recomb_rates` reads a tab-separated table with `chrom`,
    `pos` and `pos_cm`, keeps `chr1`..`chr22`, and `assign_centiMorgans`
    interpolates a position's centiMorgans linearly between the rows either
    side of it. So the map has to span every gene interval the fixture writes,
    or a block at the far end of a chromosome interpolates off the end of the
    table.

    Mocked rather than shipped, and semi-realistic in the one way that bites:
    the rate varies per interval, drawn in `CENTIMORGANS_PER_MEGABASE` from the
    fixture's own stream. A constant rate would make the map affine and the
    interpolation exact by construction, which is the case that hides an error
    in it.
    """
    rng = np.random.default_rng([truth.seed, MARKER_SPACING])
    rows: list[dict[str, object]] = []

    for chromosome, extent in enumerate(truth.lengths, start=1):
        # One marker past the last gene interval, so every block has a row
        # above it as well as below it.
        span = int(extent) * GENE_SPACING + GENE_LENGTH
        positions = np.arange(0, span + MARKER_SPACING, MARKER_SPACING)

        rates = rng.uniform(*CENTIMORGANS_PER_MEGABASE, positions.size)
        steps = rates[1:] * np.diff(positions) / 1e6
        centimorgans = np.concatenate(([0.0], np.cumsum(steps)))

        rows += [
            {"chrom": f"chr{chromosome}", "pos": int(pos), "pos_cm": float(cm)}
            for pos, cm in zip(positions, centimorgans, strict=True)
        ]

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def write_tmp_inputs(
    truth: CoreInferenceTruth,
    pre_image: Unsegmented,
    root: Path,
    *,
    sample_label: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> WrittenInputs:
    """Serialize a pre-image into `load_input_data`'s inputs under `root`.

    The allele matrices are `(n_spots, n_blocks)` with the A count as
    `total - B`, so the pair carries the same information the blocks did and a
    loader that swapped them would return the complement.

    `sample_label`, the sample of each spot, writes one `spaceranger_S{k}` per
    sample and one sample-sheet row each, over one `snp_dir` (#328). The
    spots must be grouped by sample, as `get_sample_list` requires, and the
    barcodes carry the sample, `BC00000-1_S2`, so they are unique across
    samples. `positions`, `(n_spots, 2)`, replaces the lattice's own.
    """
    n_spots = truth.n_spots
    single = sample_label is None
    labels = np.zeros(n_spots, dtype=np.int64) if sample_label is None else sample_label

    if np.any(np.diff(labels) < 0):
        msg = "spots must be grouped by sample, in sample order"
        raise ValueError(msg)

    snp_dir = root / "snp"
    snp_dir.mkdir(parents=True)

    barcodes = (
        [f"BC{spot:05d}-1" for spot in range(n_spots)]
        if single
        else [f"BC{spot:05d}-1_S{k + 1}" for spot, k in enumerate(labels)]
    )
    (snp_dir / "barcodes.txt").write_text("\n".join(barcodes) + "\n")

    # Genomic coordinates: one interval per planted bin, chromosomes taken
    # from the planted segmentation so `lengths` is recoverable.
    chromosome_of, index_within = [], []
    for chromosome, extent in enumerate(truth.lengths, start=1):
        chromosome_of += [chromosome] * int(extent)
        index_within += list(range(int(extent)))

    table = pre_image.df_gene_snp
    genes = table[table.gene.notna() & table.bin_id.notna()]
    pd.DataFrame(
        [
            {
                "name": row.gene,
                "name2": row.gene,
                # `get_reference_genes` keeps `chr1`..`chr22` and reads the
                # integer back off the string, so the names are real ones.
                "chrom": f"chr{chromosome_of[int(row.bin_id)]}",
                "cdsStart": index_within[int(row.bin_id)] * GENE_SPACING,
                "cdsEnd": index_within[int(row.bin_id)] * GENE_SPACING + GENE_LENGTH,
            }
            for row in genes.itertuples()
        ]
    ).set_index("name").to_csv(root / "hgtable.tsv", sep="\t")

    # `form_gene_snp_table` parses the contig as `int(x.split("_")[0])`, so the
    # id carries a bare integer and not `chr1`.
    snp_rows = table[table.snp_id.notna()].sort_values("block_id")
    snp_ids = np.array(
        [
            f"{chromosome_of[int(row.bin_id)]}_"
            f"{index_within[int(row.bin_id)] * GENE_SPACING + 1000 + int(row.block_id) % 97}"
            "_A_T"
            for row in snp_rows.itertuples()
        ],
        dtype=object,
    )
    np.save(snp_dir / "unique_snp_ids.npy", snp_ids, allow_pickle=True)

    # `summarize_counts_for_blocks` reads **`cell_snp_Aallele`** into
    # `single_X[:, 1, :]` (`omics.py:468`), which everything downstream scores
    # as the B allele. So the file named A carries the haplotype the model
    # calls B. Written the way the code reads it, with the inversion stated.
    allele_a = pre_image.block_single_X[:, 1, :].T
    allele_b = pre_image.block_single_total_bb_RD.T - allele_a
    sp.save_npz(snp_dir / "cell_snp_Aallele.npz", sp.csr_matrix(allele_a))
    sp.save_npz(snp_dir / "cell_snp_Ballele.npz", sp.csr_matrix(allele_b))

    adata = pre_image.adata.copy()
    adata.obs_names = barcodes
    # Sparse because `get_spaceranger_counts` calls `.toarray()` unguarded.
    adata.X = sp.csr_matrix(adata.X)

    if positions is None:
        rows, columns = np.unravel_index(np.arange(n_spots), truth.lattice)
    else:
        rows, columns = positions[:, 0], positions[:, 1]

    sheet = []

    for k in np.unique(labels):
        spots = np.flatnonzero(labels == k)
        sample_id = SAMPLE_ID if single else f"S{k + 1}"
        spaceranger_dir = root / (
            "spaceranger" if single else f"spaceranger_{sample_id}"
        )
        (spaceranger_dir / "spatial").mkdir(parents=True)

        adata[spots].copy().write_h5ad(
            spaceranger_dir / f"{FILTERED_FEATURE_NAME}.h5ad"
        )
        pd.DataFrame(
            {
                "barcode": [barcodes[spot] for spot in spots],
                "in_tissue": 1,
                "x": rows[spots],
                "y": columns[spots],
                "pixel_row": rows[spots] * 100,
                "pixel_col": columns[spots] * 100,
            }
        ).to_csv(spaceranger_dir / "spatial" / "tissue_positions.csv", index=False)

        sheet.append(
            {
                "bam": "none",
                "sample_id": sample_id,
                "spaceranger_dir": str(spaceranger_dir),
                "snp_dir": str(snp_dir),
            }
        )

    sample_sheet = root / "sample_sheet.tsv"
    pd.DataFrame(sheet).to_csv(sample_sheet, index=False, sep="\t")

    return WrittenInputs(
        root=root,
        sample_sheet=sample_sheet,
        genetic_map=write_genetic_map(truth, root / "genetic_map.tab"),
        hgtable=root / "hgtable.tsv",
        barcodes=barcodes,
        snp_ids=snp_ids,
        allele_a=allele_a,
        allele_b=allele_b,
        gene_counts=np.asarray(pre_image.adata.layers["count"]),
    )


@contextmanager
def written_config(written: WrittenInputs) -> Iterator[None]:
    """Install the global config `cnaster` reads, and put back what was there.

    A context manager rather than a call, because the prep chain is several
    functions long and each of them reads the global -- `form_gene_snp_table`
    alone wants `paths.output_dir` and `run.cache`. Restoring after the first
    one would leave the rest without a config.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    try:
        set_global_config(YAMLConfig(written.config()))
        yield
    finally:
        set_global_config(previous)


def load_written(written: WrittenInputs) -> Any:
    """Run `cnaster.io.load_input_data` against what was written."""
    from cnaster.io import load_input_data

    with written_config(written):
        from cnaster.config import get_global_config

        return load_input_data(get_global_config())
