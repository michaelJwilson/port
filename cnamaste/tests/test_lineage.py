"""The run's two hierarchies, as `/lineage` records them and as the stages' own outputs state them.

**Segments:** genes -> blocks -> phase segments -> bins -> kept bins (-1
where the normal-BAF filter removed a bin) -> re-binned -> per-clone state
runs, and the two output tables, `cnv_seglevel.tsv` (one row per re-binned
bin) and `cnv_genelevel.tsv` (one row per called gene). Each level is a
`Segmentation.of` the gene rows, which refuses a label that is not
contiguous, spans two contigs or is out of genomic order; each is related to
its parent by `refines`, `select` or `coarsen`; and each level's derived
`lengths` is the stage's own.

**Clones:** the initial grid -> the BAF fit's ICM labels -> its Potts
merges -> its empty-clone re-indexing -> `merge_by_minspots` -> the RDR
start (a refinement of the merged BAF clones) -> the RDR fit's ICM, merges
and re-indexing -> `merge_by_minspots` -> `reindex_clones`. Composing the
recorded parent maps reproduces every later level, and `clone_labels.tsv`.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import pytest

from audit.segments import DROPPED, Segmentation

CHAIN = [
    ("blocks", "genes"),
    ("phase_segments", "blocks"),
    ("bins", "blocks"),
    ("phase_segments", "bins"),
    ("bins", "kept_bins"),
    ("rebinned", "kept_bins"),
]
"""(coarse, fine): every fine segment lies inside one coarse segment. Bins are unions of blocks inside one phase
segment (the first binning breaks at `refined_lengths`); the re-binned bins break only at contigs, so they need
not respect the phase segments."""


@pytest.mark.parametrize(("coarse", "fine"), CHAIN)
def test_a_level_coarsens_its_parent(lineage: Any, coarse: str, fine: str) -> None:
    """Every `fine` segment lies inside one `coarse` segment, over the genes both keep."""
    assert lineage.levels[fine].refines(lineage.levels[coarse])


def test_kept_bins_are_a_selection_of_the_bins(lineage: Any) -> None:
    """The normal-BAF filter drops whole bins and re-ranks the rest in order: `select`, exactly."""
    bins, kept = lineage.levels["bins"], lineage.levels["kept_bins"]
    keep = np.zeros(bins.n_segments, dtype=bool)
    keep[np.unique(bins.label[kept.label != DROPPED])] = True
    assert np.array_equal(bins.select(keep, name="selected").label, kept.label)


def test_dropped_genes_only_grow_along_the_chain(lineage: Any) -> None:
    """-1 is monotone: a gene a level drops, every later level drops."""
    order = ["genes", "blocks", "phase_segments", "bins", "kept_bins", "rebinned", "seglevel", "genelevel"]
    for before, after in zip(order, order[1:], strict=False):
        dropped_before = lineage.levels[before].label == DROPPED
        assert np.all(lineage.levels[after].label[dropped_before] == DROPPED), f"{after} keeps a gene {before} dropped"


def test_rebinned_drops_exactly_what_the_filter_dropped(lineage: Any) -> None:
    """The second binning merges kept bins and drops nothing more."""
    assert np.array_equal(lineage.levels["rebinned"].label == DROPPED, lineage.levels["kept_bins"].label == DROPPED)


def test_state_runs_coarsen_the_rebinned_bins(lineage: Any) -> None:
    """Each clone's runs of one decoded state are unions of re-binned bins, one contig each."""
    runs = [n for n in lineage.names if n.startswith("state_runs_clone")]
    assert runs
    for name in runs:
        assert lineage.levels["rebinned"].refines(lineage.levels[name]), name


def test_the_output_tables_are_the_rebinned_bins_and_the_called_genes(lineage: Any) -> None:
    """`cnv_seglevel.tsv` rows are the re-binned bins; `cnv_genelevel.tsv` rows the genes a re-binned bin holds."""
    assert np.array_equal(lineage.levels["seglevel"].label, lineage.levels["rebinned"].label)
    genelevel = lineage.levels["genelevel"]
    called = lineage.levels["rebinned"].label != DROPPED
    assert np.array_equal(genelevel.label != DROPPED, called)
    assert genelevel.n_segments == called.sum()


LENGTHS = {
    "blocks": "02_blocks/summarize_counts_for_blocks/out/lengths",
    "bins": "04_bins/summarize_counts_for_bins/out/lengths",
    "kept_bins": "06_normal/normal_baf_bin_filter/out/1/lengths",
    "rebinned": "07_rebin/summarize_counts_for_bins/out/lengths",
}


@pytest.mark.parametrize("level", list(LENGTHS))
def test_derived_lengths_are_the_stages(sim: Any, lineage: Any, level: str) -> None:
    """`lengths`, segments per contig, derived from the labels, is the stage's; a contig the stage counts as 0 has no entry."""
    stated = np.asarray(sim.stored(LENGTHS[level]))
    assert np.array_equal(lineage.levels[level].lengths, stated[stated > 0])


def test_the_seglevel_coordinates_are_the_rebinned_extents(sim: Any, lineage: Any) -> None:
    """Each `cnv_seglevel.tsv` row's CHR, START, END is its bin's: the first gene's or SNP's start, the last row's end."""
    seglevel = pd.read_csv(io.BytesIO(sim.file("cnv_seglevel.tsv")), sep="\t")
    rebinned = lineage.levels["rebinned"]
    assert np.array_equal(seglevel["CHR"].to_numpy(), rebinned.contig)
    # NB a bin's first row may be a SNP before its first gene: START at or before the first gene's.
    assert np.all(seglevel["START"].to_numpy() <= rebinned.start)
    assert np.all(seglevel["END"].to_numpy() >= rebinned.start)


def planted_breakpoints(truth: Any) -> pd.DataFrame:
    """Each position inside a contig where some planted clone's (A, B) changes."""
    profile = truth.profile
    copies = profile.filter(like="_copy").to_numpy()
    same_contig = profile["chr"].to_numpy()[1:] == profile["chr"].to_numpy()[:-1]
    changed = np.any(copies[1:] != copies[:-1], axis=1) & same_contig
    return pd.DataFrame({"chr": profile["chr"].to_numpy()[1:][changed], "position": profile["start"].to_numpy()[1:][changed]})


BOUNDARY_LEVELS = {"blocks": 1.0e6, "bins": 1.0e6, "rebinned": 1.0e6}
"""Level -> the largest median distance, in bp, from a planted breakpoint to the level's nearest segment start.

The median, not the maximum: on easy one breakpoint (chr 6) lies 16.8 Mb from any gene, so no
gene-based segment can start nearer, at any level."""


@pytest.mark.parametrize("level", list(BOUNDARY_LEVELS))
def test_planted_breakpoints_fall_near_a_boundary(lineage: Any, truth: Any, level: str) -> None:
    """Planted breakpoints lie, at the median, within the stated distance of a segment start on their contig."""
    segments: Segmentation = lineage.levels[level]
    starts, contigs = segments.start, segments.contig
    distances = []
    for chrom, position in planted_breakpoints(truth).itertuples(index=False):
        on = starts[contigs == int(str(chrom).removeprefix("chr"))]
        distances.append(np.min(np.abs(on - position)))
    print(f"{level}: breakpoint distances (Mb) {np.round(np.array(distances) / 1e6, 2)}")
    assert distances and float(np.median(distances)) <= BOUNDARY_LEVELS[level]


# --- clones -------------------------------------------------------------------

CLONE_CHAIN = [
    ("baf_icm", "baf_raw"),
    ("baf_raw", "baf_fit"),
    ("baf_fit", "baf_merged"),
    ("rdr_icm", "rdr_raw"),
    ("rdr_raw", "rdr_fit"),
    ("rdr_fit", "rdr_merged"),
    ("rdr_merged", "final"),
]
"""(level, next level): the next is the recorded parent map applied to the level, spot by spot."""


@pytest.mark.parametrize(("before", "after"), CLONE_CHAIN)
def test_a_parent_map_takes_one_level_to_the_next(clones: Any, before: str, after: str) -> None:
    """`parent/<after>[labels at <before>]` is the labels at `after`."""
    assert np.array_equal(clones.parents[after][clones[before]], clones[after])


def test_the_composed_chain_gives_the_written_labels(sim: Any, clones: Any) -> None:
    """From each fit's ICM labels to `clone_labels.tsv` through the composed maps, and the RDR start inside the merged BAF clones."""
    composed = clones["rdr_icm"]
    for _, after in CLONE_CHAIN[3:]:
        composed = clones.parents[after][composed]
    written = pd.read_csv(io.BytesIO(sim.file("clone_labels.tsv")), sep="\t", comment="#", index_col="barcode")
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    assert np.array_equal(composed, written.loc[barcodes, "clone_label"].to_numpy())
    assert np.array_equal(clones.parents["rdr_init"][clones["rdr_init"]], clones["baf_merged"])


def test_the_fits_end_where_the_chain_says(sim: Any, clones: Any) -> None:
    """The recorded maps agree with the stages: each fit's labels, the merges' and the reindex's."""
    pairs = {
        "baf_fit": "05_baf/run_core_inference/out/assignment/new_assignment",
        "rdr_fit": "08_rdr/run_core_inference/out/assignment/new_assignment",
        "rdr_init": "08_rdr/initialize_rdr_clone_refininement/out/0",
    }
    for level, path in pairs.items():
        assert np.array_equal(clones[level], np.asarray(sim.stored(path))), level
    for run in ("05_baf", "08_rdr"):
        internal = sim.internal(run)
        level = "baf" if run == "05_baf" else "rdr"
        assert np.array_equal(internal["raw"], clones[f"{level}_raw"])
        assert {int(k): v for k, v in internal["re_indexing"].items()} == {
            int(c): int(i) for c, i in enumerate(clones.parents[f"{level}_fit"]) if i >= 0
        }
