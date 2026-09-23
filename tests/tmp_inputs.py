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
from typing import Any, Literal

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
"""One slice. Multi-slice alignment is a separate concern and a separate fixture."""

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
    truth: CoreInferenceTruth, pre_image: Unsegmented, root: Path
) -> WrittenInputs:
    """Serialize a pre-image into `load_input_data`'s inputs under `root`.

    The allele matrices are `(n_spots, n_blocks)` with the A count as
    `total - B`, so the pair carries the same information the blocks did and a
    loader that swapped them would return the complement.
    """

    snp_dir, spaceranger_dir = root / "snp", root / "spaceranger"
    (spaceranger_dir / "spatial").mkdir(parents=True)
    snp_dir.mkdir(parents=True)

    n_spots = truth.n_spots
    barcodes = [f"BC{spot:05d}-1" for spot in range(n_spots)]
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
    adata.write_h5ad(spaceranger_dir / f"{FILTERED_FEATURE_NAME}.h5ad")

    rows, columns = np.unravel_index(np.arange(n_spots), truth.lattice)
    pd.DataFrame(
        {
            "barcode": barcodes,
            "in_tissue": 1,
            "x": rows,
            "y": columns,
            "pixel_row": rows * 100,
            "pixel_col": columns * 100,
        }
    ).to_csv(spaceranger_dir / "spatial" / "tissue_positions.csv", index=False)

    sample_sheet = root / "sample_sheet.tsv"
    pd.DataFrame(
        [
            {
                "bam": "none",
                "sample_id": SAMPLE_ID,
                "spaceranger_dir": str(spaceranger_dir),
                "snp_dir": str(snp_dir),
            }
        ]
    ).to_csv(sample_sheet, index=False, sep="\t")

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
def written_config(
    source: "WrittenInputs | Path | str | dict[str, Any]",
) -> Iterator[Any]:
    """Install the global config `cnaster` reads, yield it, and put back what was there.

    `source` is written inputs, whose loader configuration is installed; the
    path of a configuration file, read as `YAMLConfig.from_file` reads it; or
    a configuration as a dictionary.

    A context manager rather than a call, because the prep chain is several
    functions long and each of them reads the global -- `form_gene_snp_table`
    alone wants `paths.output_dir` and `run.cache`. Restoring after the first
    one would leave the rest without a config.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    try:
        if isinstance(source, WrittenInputs):
            set_global_config(YAMLConfig(source.config()))
        elif isinstance(source, Path | str):
            set_global_config(YAMLConfig.from_file(source))
        else:
            set_global_config(YAMLConfig(source))
        yield get_global_config()
    finally:
        set_global_config(previous)


@dataclass(frozen=True)
class Binned:
    """What the prep chain `run_cnaster` runs between loading and binning produced.

    `table` is the gene-SNP table as the last stage run left it; `bins` is
    `None` when the chain stopped before binning.
    """

    loaded: Any
    table: Any
    blocks: Any
    bins: Any


def read_to_bins(
    written: WrittenInputs,
    *,
    loaded: Any = None,
    initial_min_umi: int = 1,
    secondary_min_umi: int = 1,
    through: Literal["blocks", "ranges", "bins"] = "bins",
) -> Binned:
    """`load_input_data` through the stage `through` names, under the installed config.

    `run_cnaster`'s order: `form_gene_snp_table`, `assign_initial_blocks` and
    `summarize_counts_for_blocks` (`"blocks"`); `create_bin_ranges`
    (`"ranges"`); `summarize_counts_for_bins` with every block phased, no
    phase-switch shift and no genetic map (`"bins"`). Each stage reads the
    global config, so the caller installs one and holds it. `loaded` is a
    `load_input_data` return already in hand; without one the chain loads.
    """
    from cnaster.config import get_global_config
    from cnaster.io import load_input_data
    from cnaster.omics import (
        assign_initial_blocks,
        create_bin_ranges,
        form_gene_snp_table,
        summarize_counts_for_bins,
        summarize_counts_for_blocks,
    )

    if loaded is None:
        loaded = load_input_data(get_global_config())
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(written.hgtable), loaded.adata
    )
    table = assign_initial_blocks(
        table,
        loaded.adata,
        *alleles,
        loaded.unique_snp_ids,
        initial_min_umi=initial_min_umi,
    )
    blocks = summarize_counts_for_blocks(
        table, loaded.adata, *alleles, loaded.unique_snp_ids
    )
    if through == "blocks":
        return Binned(loaded=loaded, table=table, blocks=blocks, bins=None)

    table = create_bin_ranges(
        table,
        loaded.adata,
        *alleles,
        loaded.unique_snp_ids,
        blocks.X,
        blocks.total_bb_RD,
        blocks.lengths,
        secondary_min_umi=secondary_min_umi,
        secondary_min_snp_umi=secondary_min_umi,
        secondary_min_normal_umi=0,
    )
    if through == "ranges":
        return Binned(loaded=loaded, table=table, blocks=blocks, bins=None)

    bins = summarize_counts_for_bins(
        table,
        loaded.adata,
        blocks.X,
        blocks.total_bb_RD,
        np.ones(int(table.block_id.dropna().nunique()), dtype=bool),
        nu=1.0,
        logphase_shift=0.0,
        geneticmap_file=None,
    )
    return Binned(loaded=loaded, table=table, blocks=blocks, bins=bins)


def load_written(written: WrittenInputs) -> Any:
    """Run `cnaster.io.load_input_data` against what was written."""
    from cnaster.io import load_input_data

    with written_config(written) as config:
        return load_input_data(config)
