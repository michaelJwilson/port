"""The run's two hierarchies, as `/lineage` records them and as the stages' own outputs state them.

**Segments:** genes -> blocks -> phase segments -> bins -> kept bins (-1
where the normal-BAF filter removed a bin) -> re-binned -> per-clone state
runs, and the two output tables, `cnv_seglevel.tsv` (one row per re-binned
bin) and `cnv_genelevel.tsv` (one row per called gene). Each level is a
`Segmentation.of` the gene rows, which refuses a label that is not
contiguous, spans two contigs or is out of genomic order; each is related to
its parent by `refines`, `select` or `coarsen`; and each level's derived
`lengths` is the stage's own. `CHAIN` states the rule that built each level
from the one below, and checks it on counts recomputed from the staged inputs.

**Clones:** the initial grid -> the BAF fit's ICM labels -> its Potts
merges -> its empty-clone re-indexing -> `merge_by_minspots` -> the RDR
start (a refinement of the merged BAF clones) -> the RDR fit's ICM, merges
and re-indexing -> `merge_by_minspots` -> `reindex_clones`. Composing the
recorded parent maps reproduces every later level, and `clone_labels.tsv`.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from audit.criteria import Units, bins_of, minor_baf, normal_baf_outside
from audit.segments import DROPPED, Segmentation

def overlapping_genes_share_a_block(u: Units) -> None:
    genes = u.lineage.genes
    reach, group = genes.end.copy(), np.zeros(genes.n_genes, dtype=np.int64)
    for g in range(1, genes.n_genes):
        overlaps = genes.contig[g] == genes.contig[g - 1] and genes.start[g] < reach[g - 1]
        reach[g] = max(reach[g], reach[g - 1]) if overlaps else reach[g]
        group[g] = group[g - 1] + (not overlaps)
    print(f"blocks: {u.lineage.levels['blocks'].n_segments} from {int(group[-1]) + 1} merged gene ranges")
    assert Segmentation.of(genes, group, name="ranges").refines(u.lineage.levels["blocks"])


def phase_segments_break_at_contigs_and_baf_steps(u: Units) -> None:
    res, _, refined = u.replayed.value("03_phasing/initial_phase_given_partition/out")
    lengths = np.asarray(u.sim.stored("02_blocks/summarize_counts_for_blocks/out/lengths"))
    minor = minor_baf(res, u.setting("hmm.n_states"), int(lengths.sum()))
    gap, step = u.setting("phasing.min_new_segment_size"), u.setting("phasing.baf_change_threshold")
    expected, offset = [], 0
    for n in lengths:
        s = 0
        for i in range(n):
            if i > s + gap and np.any(np.abs(minor[:, offset + i] - minor[:, offset + i - 1]) >= step):
                expected.append(i - s)
                s = i
        expected.append(n - s)
        offset += n
    print(f"phase segments: {len(expected)}, {len(expected) - lengths.size} at a BAF step")
    assert np.array_equal(np.asarray(refined), expected)
    assert int(np.sum(refined)) == u.lineage.levels["blocks"].n_segments


def bins_stay_inside_their_runs(u: Units) -> None:
    _, _, refined = u.replayed.value("03_phasing/initial_phase_given_partition/out")
    assert bins_of(u, "bins", "blocks", np.asarray(refined), normal=False).run_edges_kept


def bins_leave_where_normal_baf_is_off_balance(u: Units) -> None:
    removed = normal_baf_outside(u)
    bins, kept = u.lineage.levels["bins"], u.lineage.levels["kept_bins"]
    print(f"bins removed: {int(removed.sum())} of {removed.size}")
    assert np.array_equal(kept.label, bins.select(~removed, name="kept").label)
    assert np.array_equal(np.unique(kept.label[kept.label != DROPPED]), np.arange(int((~removed).sum())))


RECORDED_CROSSINGS = 0
"""Re-binned bins whose genes span two phase segments, on easy. The second binning cuts runs only at contigs, so a
re-binned bin need not respect the phase segments; on easy none crosses one."""


def rebins_stay_inside_contigs(u: Units) -> None:
    lengths = np.asarray(u.sim.stored("06_normal/normal_baf_bin_filter/out/1/lengths"))
    assert bins_of(u, "rebinned", "kept_bins", lengths, normal=True).run_edges_kept
    rebinned, phase = u.lineage.levels["rebinned"], u.lineage.levels["phase_segments"]
    called = rebinned.label != DROPPED
    crossing = int(np.sum(pd.Series(phase.label[called]).groupby(rebinned.label[called]).nunique().to_numpy() > 1))
    print(f"re-binned bins across a phase-segment boundary: {crossing} of {rebinned.n_segments}")
    assert crossing == RECORDED_CROSSINGS


class Row(NamedTuple):
    coarse: str
    fine: str
    rule: Callable[[Units], None] | None
    doc: str


CHAIN = [
    Row("blocks", "genes", overlapping_genes_share_a_block,
        "`omics.assign_initial_blocks` (run_cnamaste.py:172-179): gene ranges that overlap on one contig merge "
        "(omics.py:557-575); consecutive ranges then extend, never across a contig (omics.py:634-700, Ticket#466). "
        "The SNP-UMI floor is `test_stages.py`'s CRITERIA."),
    Row("phase_segments", "blocks", phase_segments_break_at_contigs_and_baf_steps,
        "`phasing.initial_phase_given_partition` (phasing.py:140-155, 290-332): `refined_lengths` breaks at every contig end "
        "and at block i when i > s + phasing.min_new_segment_size and some clone's minor BAF steps by phasing.baf_change_threshold."),
    Row("bins", "blocks", bins_stay_inside_their_runs,
        "The first `omics.create_bin_ranges` (run_cnamaste.py:425): runs cut at cumsum(refined_lengths) and either side of a "
        "block over quality.max_binlength (omics.py:867-874); `greedy_binning_nobreak` bins inside each run (omics.py:16-105)."),
    Row("phase_segments", "bins", None,
        "The same binning: a run never crosses a phase segment, so neither does a bin."),
    Row("bins", "kept_bins", bins_leave_where_normal_baf_is_off_balance,
        "`normal_spot.normal_baf_bin_filter` (run_cnamaste.py:939): a bin leaves when its pooled normal-spot B count is outside "
        "the beta-binomial interval at quality.normal_allele_specific_confidence (normal_spot.py:954-1000); survivors re-rank."),
    Row("rebinned", "kept_bins", rebins_stay_inside_contigs,
        "The second `create_bin_ranges` (run_cnamaste.py:983, key=bin_id, normal candidates): runs cut only at contigs "
        "(`lengths`), so a re-binned bin need not respect the phase segments."),
]
"""(coarse, fine, the rule that built `coarse` from `fine`, its source): every fine segment lies inside one coarse segment,
and the construction holds of the units recomputed from the staged inputs (`audit.criteria`)."""


@pytest.mark.parametrize("row", CHAIN, ids=[f"{r.coarse}-{r.fine}" for r in CHAIN])
def test_a_level_follows_its_construction_rule(lineage: Any, request: pytest.FixtureRequest, row: Row) -> None:
    """`fine` refines `coarse`, and `row.rule` holds on easy."""
    assert lineage.levels[row.fine].refines(lineage.levels[row.coarse])
    if row.rule is not None:
        row.rule(request.getfixturevalue("units"))


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
