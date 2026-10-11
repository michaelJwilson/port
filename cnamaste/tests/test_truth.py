"""The staged run against its planted truth: phasing, slices, integer copies per class (Ticket#836).

Each row reads the staged file alone (`/truth`, the stored stage outputs and
the written tables), so it runs without the inputs; easy's file predates
`/truth` and its truth is read from the committed sample. A pinned value is
a table keyed by fixture hash, set from that fixture's capture, and a row's
bound is the recorded value less (or, for an error, plus) `MARGIN`, as in
`test_stages.py`. Every row is `end2end`: the planted truth that generated
the inputs judges the run's outputs. Every row runs on all four fixtures,
easy (`SUPPORTED_SIMS`) and `TRUTH_SIMS`, whatever `--sims` selects: the
four take 3 s.

- **Phasing:** the run's phase is `initial_phase_given_partition`'s one
  indicator per SNP block, which every later stage sums through. A SNP's
  orientation is whether its phased count is the planted `B` allele: the
  indicator, exchanged where the inputs wrote it switched. Phase is defined
  up to one exchange per planted segment, so a SNP is phased correctly
  where its orientation is its segment's majority, and only segments some
  tumour clone carries allele-specifically (`A != B`) are scored: elsewhere
  no data decides it. The planted switches of a drawn fixture are its
  realization's `truth_phase.npy` at `load_input_data`'s SNP ids
  (`data/truth_phase_<hash>.npy`). CalicoST easy and hard write their
  counts in the planted orientation, with no switch: in their committed
  inputs, every LOH SNP of a tumour clone with 5 or more pseudobulk reads
  has its B fraction on the planted side of 1/2.
- **Slices:** on the two-slice fixtures, each spot keeps its planted slice
  through the run, and the clone labels are scored per slice.
- **Integer copies:** clone-bins (`cnv_seglevel.tsv`'s bins at each matched
  clone, as the ledger scores them) split by the planted pair's class, and
  the fraction whose fitted pair is the planted one, either haplotype order.
  A class recovered at 0.0 is a strict xfail citing its ticket.
- **Annotation and H&E:** not scored: no staged sample carries them
  (Ticket#311, Ticket#113).
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from audit.capture import Capture
from audit.scoring import Truth, at_truth, matched, overlap, phase_free, planted, planted_classes, swapped
from audit.sims import SUPPORTED_SIMS, TRUTH_SIMS

pytestmark = pytest.mark.end2end

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent

MARGIN = 0.02

@pytest.fixture(scope="module", params=[*SUPPORTED_SIMS, *TRUTH_SIMS])
def staged_hash(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def sim(staged_hash: str) -> Iterator[Capture]:
    """The staged file, open for the module."""
    staged = SUPPORTED_SIMS[staged_hash][1] if staged_hash in SUPPORTED_SIMS else TRUTH_SIMS[staged_hash]
    capture = Capture(PROJECT / staged)
    assert capture.config["fixture_hash"] == staged_hash
    yield capture
    capture.close()


@pytest.fixture(scope="module")
def truth(sim: Capture, staged_hash: str) -> Truth:
    """The planted truth: the staged file's `/truth`, else the committed sample's files."""
    stored = sim.truth()
    return planted(stored) if stored else planted(REPOSITORY / SUPPORTED_SIMS[staged_hash][0])


TWO_SLICES = ("23acef50", "e4bbe03f")
"""The drawn fixtures, two slices each, with phase switch errors planted (`[phasing] switch_errors = true`)."""


# --- phasing -------------------------------------------------------------------


PLANTED_PHASE = {"e4bbe03f": "truth_phase_e4bbe03f.npy", "23acef50": "truth_phase_e4bbe03f.npy"}
"""Fixture hash -> its planted phase in `tests/data/`. `dev_tree_easy_square` changes only `dev_tree_easy`'s array:
realization 0 draws its phase from the same stream over the same SNPs, and the two are equal."""


def planted_switched(sim: Any, staged_hash: str) -> np.ndarray:
    """Per `load_input_data` SNP: True where the inputs wrote `A` and `B` exchanged."""
    n = np.asarray(sim.stored("00_inputs/load_input_data/out/6")).size
    if staged_hash not in PLANTED_PHASE:
        return np.zeros(n, dtype=bool)
    switched = np.load(PROJECT / "tests" / "data" / PLANTED_PHASE[staged_hash])
    assert switched.shape == (n,) and switched.dtype == bool
    return np.asarray(switched)


class Phasing:
    """The scored SNPs (in allele-specific planted segments), in genome order, with their segment and block."""

    def __init__(self, sim: Any, truth: Truth, staged_hash: str) -> None:
        table = sim.stored("02_blocks/assign_initial_blocks/out")
        snps = table.loc[table["snp_id"].notna()]
        ids = np.asarray(sim.stored("00_inputs/load_input_data/out/6")).astype(str)
        index = pd.Index(ids).get_indexer(snps["snp_id"].astype(str))
        assert np.all(index >= 0), "a block's SNP is not one load_input_data kept"
        indicator = np.asarray(sim.stored("03_phasing/initial_phase_given_partition/out/1")).astype(bool)
        block = snps["block_id"].to_numpy(dtype=np.int64)
        switched = planted_switched(sim, staged_hash)[index]
        # NB the phased B count is B where the indicator is 1, A where 0 (test_stages' bookkeeping)
        orientation = indicator[block] != switched
        chrom = snps["CHR"].astype(str).to_numpy()
        segment = self._segments(truth, chrom, snps["START"].to_numpy())
        tumour = truth.copies_at(chrom, snps["START"].to_numpy())[:, 1:, :]
        allelic = (segment >= 0) & np.any(tumour[..., 0] != tumour[..., 1], axis=1)
        self.segment, self.block = segment[allelic], block[allelic]
        self.orientation, self.switched = orientation[allelic], switched[allelic]
        votes = pd.Series(self.orientation).groupby(self.segment).mean()
        self.correct = self.orientation == (votes.reindex(self.segment).to_numpy() >= 0.5)

    @staticmethod
    def _segments(truth: Truth, chrom: np.ndarray, position: np.ndarray) -> np.ndarray:
        profile = truth.profile
        names = profile["chr"].astype(str).str.removeprefix("chr").to_numpy()
        found = np.full(position.size, -1)
        for row in range(len(profile)):
            inside = (chrom == names[row]) & (position >= profile["start"].iloc[row]) & (position < profile["end"].iloc[row])
            found[inside] = row
        return found

    def switches(self, values: np.ndarray) -> int:
        """Changes of `values` between consecutive scored SNPs of one segment."""
        same = self.segment[1:] == self.segment[:-1]
        return int(np.sum((values[1:] != values[:-1]) & same))


def snps_phased(p: Phasing) -> float:
    return float(np.mean(p.correct))


def blocks_phased(p: Phasing) -> float:
    return float(pd.Series(p.correct).groupby(p.block).all().mean())


def switches_left(p: Phasing) -> float:
    return p.switches(p.orientation) / p.switches(p.switched)


PHASING: dict[str, tuple[Callable[[Phasing], float], str]] = {
    "SNPs phased correctly": (snps_phased, ">="),
    "blocks phased correctly (every SNP)": (blocks_phased, ">="),
    "switch errors left per planted switch": (switches_left, "<="),
}
"""Name -> (score, direction). A SNP phased against its segment's majority fails the first two; a phase switch the
run leaves, or adds, between consecutive SNPs of a segment raises the third."""

RECORDED_PHASING: dict[str, dict[str, float]] = {
    "2d4ce9a9": {"SNPs phased correctly": 0.9987, "blocks phased correctly (every SNP)": 0.9966},
    "8797710b": {"SNPs phased correctly": 0.9973, "blocks phased correctly (every SNP)": 0.9931},
    "23acef50": {"SNPs phased correctly": 0.9666, "blocks phased correctly (every SNP)": 0.9364,
                 "switch errors left per planted switch": 0.4286},
    "e4bbe03f": {"SNPs phased correctly": 0.9583, "blocks phased correctly (every SNP)": 0.9253,
                 "switch errors left per planted switch": 0.4314},
}
"""Fixture hash -> each phasing score as its capture records it, 4 decimals. Leaving the inputs' phase as written
scores 0.63-0.64 SNPs, 0.62 blocks and 1.0 switches left on the drawn fixtures; a random phase 0.54, 0.32-0.34 and
4.3."""


@pytest.fixture(scope="module")
def phasing(sim: Any, truth: Truth, staged_hash: str) -> Phasing:
    return Phasing(sim, truth, staged_hash)


def bounded(recorded: float, direction: str) -> float:
    return recorded - MARGIN if direction == ">=" else recorded + MARGIN


@pytest.mark.parametrize("name", list(PHASING))
def test_phasing(phasing: Phasing, staged_hash: str, name: str) -> None:
    """A phasing score against the planted phase, at its floor (or ceiling) for this fixture."""
    score, direction = PHASING[name]
    if name.startswith("switch") and staged_hash not in TWO_SLICES:
        pytest.skip(f"{staged_hash}: no phase switch planted (CalicoST writes the planted orientation)")
    value = score(phasing)
    recorded = RECORDED_PHASING[staged_hash][name]
    limit = bounded(recorded, direction)
    print(f"{staged_hash} {name}: {value:.4f} (recorded {recorded:.4f}, bound {limit:.4f})")
    assert value >= limit if direction == ">=" else value <= limit


# --- slices --------------------------------------------------------------------


def planted_slices(sim: Any) -> pd.Series:
    """Barcode -> its planted slice (`truth_clone_labels.tsv`'s `sample_id`)."""
    table = pd.read_csv(io.BytesIO(sim.truth()["truth_clone_labels.tsv"]), sep="\t", index_col=0)
    return table["sample_id"].astype(str)


def two_slices(staged_hash: str) -> None:
    if staged_hash not in TWO_SLICES:
        pytest.skip(f"{staged_hash}: one slice")


def test_each_spot_keeps_its_slice(sim: Any, staged_hash: str) -> None:
    """Every planted spot is loaded once, under its own slice, and every output names that slice (Ticket#446).

    Fails where a slice loads no spots (Ticket#446's None sample ids), a spot moves to the other slice in
    `get_sample_list`'s index, or `clone_labels.tsv` writes another slice's id."""
    two_slices(staged_hash)
    slices = planted_slices(sim)
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    assert sorted(barcodes) == sorted(slices.index), "the run did not load each planted spot once"
    names, index = sim.stored("00_inputs/get_sample_list/out")
    assert sorted(names) == sorted(slices.unique())
    assert np.array_equal(np.asarray(names)[np.asarray(index)], slices.loc[barcodes].to_numpy())
    written = pd.read_csv(io.BytesIO(sim.file("clone_labels.tsv")), sep="\t", index_col=0)["sample_id"].astype(str)
    assert written.loc[slices.index].equals(slices)


RECORDED_SLICE_ARI: dict[str, tuple[float, float]] = {"23acef50": (0.7050, 0.5833), "e4bbe03f": (0.5837, 0.4694)}
"""Fixture hash -> the final clone ARI against the planted clones on each slice, in `get_sample_list`'s order."""


@pytest.mark.parametrize("slice_index", [0, 1])
def test_clones_per_slice(sim: Any, truth: Truth, staged_hash: str, slice_index: int) -> None:
    """The final clones against the planted ones on one slice (ARI): fails where one slice's clones are mixed while
    the other's carry the pooled score."""
    two_slices(staged_hash)
    names, _ = sim.stored("00_inputs/get_sample_list/out")
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    final = np.asarray(sim.result("08_rdr/reindex_clones/out/0")["new_assignment"])
    fitted = at_truth(barcodes, final, truth)
    on = (planted_slices(sim).loc[truth.barcodes].to_numpy() == names[slice_index]) & (fitted >= 0)
    value = float(adjusted_rand_score(truth.labels[on], fitted[on]))
    recorded = RECORDED_SLICE_ARI[staged_hash][slice_index]
    print(f"{staged_hash} slice {names[slice_index]}: ARI {value:.4f} (recorded {recorded:.4f})")
    assert value >= recorded - MARGIN


# --- integer copies per class ----------------------------------------------------


CLASSES = {"LOH": "loh", "balanced": "balanced_gain", "unbalanced": "unbalanced_gain"}
"""This file's class -> `audit.scoring.planted_classes`' key: LOH a pair with a zero, balanced `A == B > 1`,
unbalanced `A != B`, both nonzero."""


def exact_per_class(sim: Any, truth: Truth) -> dict[str, tuple[float, int]]:
    """Per class: the fraction of clone-bins whose fitted pair is the planted one in either order, and their count."""
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    final = np.asarray(sim.result("08_rdr/reindex_clones/out/0")["new_assignment"])
    fitted = at_truth(barcodes, final, truth)
    seglevel = pd.read_csv(io.BytesIO(sim.file("cnv_seglevel.tsv")), sep="\t")
    scored = fitted >= 0
    clone_of = matched(overlap(truth.labels[scored], fitted[scored], truth.n_clones, int(fitted.max()) + 1))
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    copies = truth.copies_at(seglevel["CHR"].astype(str).to_numpy(), middle)
    covered = copies[:, 0, 0] >= 0
    t, ab = [], []
    for clone, fit in clone_of.items():
        t.append(copies[covered, clone, 0] * 1_000 + copies[covered, clone, 1])
        ab.append(seglevel[f"clone{fit} A"].to_numpy()[covered] * 1_000 + seglevel[f"clone{fit} B"].to_numpy()[covered])
    planted, fit = np.concatenate(t), np.concatenate(ab)
    exact = (planted == fit) | (planted == swapped(fit))
    assert np.array_equal(exact, phase_free(planted) == phase_free(fit))
    classes = planted_classes(planted)
    return {name: (float(np.mean(exact[classes[key]])) if classes[key].any() else float("nan"), int(classes[key].sum()))
            for name, key in CLASSES.items()}


RECORDED_COPIES: dict[str, dict[str, float]] = {
    "2d4ce9a9": {"LOH": 0.5333, "balanced": 0.0, "unbalanced": 0.3299},
    "8797710b": {"LOH": 0.3458, "balanced": 0.0, "unbalanced": 0.3673},
    "23acef50": {"LOH": 0.4821, "balanced": 0.0, "unbalanced": 0.9955},
    "e4bbe03f": {"LOH": 0.0, "balanced": 1.0, "unbalanced": 0.7802},
}
"""Fixture hash -> each class's exact-recovery fraction as its capture records it, 4 decimals, over 6-232
clone-bins."""

MOST = 0.5
"""A failing class's row asserts that most of its clone-bins are exact, so a fix fails the strict xfail."""

FAILING: dict[tuple[str, str], str] = {
    **{(h, "balanced"): ("Ticket#903: the RDR fit holds the planted (2, 2) at neutral depth (fitted mu / the neutral bins' "
                         f"{r}), so it decodes (1, 1)") for h, r in (("2d4ce9a9", "1.015"), ("8797710b", "1.022"),
                                                                     ("23acef50", "1.026"))},
    ("e4bbe03f", "LOH"): ("Ticket#904: the fitted clones matched to the two LOH clones are mixtures (planted share 0.58, 0.43), "
                          "their minor BAF 0.35-0.36, so the LOH bins decode (1, 2)"),
}
"""(fixture hash, class) -> why the class is recovered at 0.0: the ticket and the mechanism measured on the capture."""


@pytest.mark.parametrize("name", list(CLASSES))
def test_integer_copies_per_class(sim: Any, truth: Truth, staged_hash: str, name: str, request: pytest.FixtureRequest) -> None:
    """One class's exact (A, B) recovery at its floor: fails where the run loses a class the pooled copy scores hide,
    e.g. LOH decoded heterozygous. A class recovered at 0.0 asserts `MOST` under a strict xfail citing its ticket."""
    value, count = exact_per_class(sim, truth)[name]
    recorded = RECORDED_COPIES[staged_hash][name]
    assert count > 0, f"{staged_hash}: no {name} clone-bin planted"
    print(f"{staged_hash} {name}: {value:.4f} over {count} clone-bins (recorded {recorded:.4f})")
    reason = FAILING.get((staged_hash, name))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
        assert value >= MOST
    assert value >= recorded - MARGIN
