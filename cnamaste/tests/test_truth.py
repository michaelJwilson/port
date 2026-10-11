"""The staged run against its planted truth: phasing and integer copies per class (Ticket#836).

Each row reads the staged file (the stored stage outputs and the written
tables) and the committed sample's truth files. A pinned value is a table
keyed by fixture hash, set from that fixture's capture, and a row's bound
is the recorded value less `MARGIN`, as in `test_stages.py`. Every row is
`end2end`: the planted truth that generated the inputs judges the run's
outputs. Easy (`2d4ce9a9`) is the one staged fixture; the drawn fixtures'
planted phase switches, slices and switch-error rows arrive with them.

- **Phasing:** the run's phase is `initial_phase_given_partition`'s one
  indicator per SNP block, which every later stage sums through. A SNP's
  orientation is whether its phased count is the planted `B` allele: the
  indicator. Phase is defined up to one exchange per planted segment, so a
  SNP is phased correctly where its orientation is its segment's majority,
  and only segments some tumour clone carries allele-specifically
  (`A != B`) are scored: elsewhere no data decides it. CalicoST easy writes
  its counts in the planted orientation, with no switch: in its committed
  inputs, none of 1,126 LOH SNP pseudobulks of a tumour clone with 5 or
  more reads has its B fraction on the wrong side of 1/2.
- **Integer copies:** clone-bins (`cnv_seglevel.tsv`'s bins at each matched
  clone, as the ledger scores them) split by the planted pair's class, and
  the fraction whose fitted pair is the planted one, either haplotype order.
  A class recovered at 0.0 is a strict xfail citing its ticket.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from audit.scoring import Truth, at_truth, matched, overlap, phase_free, planted_classes, swapped

pytestmark = pytest.mark.end2end

PROJECT = Path(__file__).resolve().parents[1]

MARGIN = 0.02

# --- phasing -------------------------------------------------------------------


class Phasing:
    """The scored SNPs (in allele-specific planted segments), in genome order, with their segment and block."""

    def __init__(self, sim: Any, truth: Truth) -> None:
        table = sim.stored("02_blocks/assign_initial_blocks/out")
        snps = table.loc[table["snp_id"].notna()]
        ids = np.asarray(sim.stored("00_inputs/load_input_data/out/6")).astype(str)
        index = pd.Index(ids).get_indexer(snps["snp_id"].astype(str))
        assert np.all(index >= 0), "a block's SNP is not one load_input_data kept"
        indicator = np.asarray(sim.stored("03_phasing/initial_phase_given_partition/out/1")).astype(bool)
        block = snps["block_id"].to_numpy(dtype=np.int64)
        # NB the phased B count is B where the indicator is 1, A where 0 (test_stages' bookkeeping)
        orientation = indicator[block]
        chrom = snps["CHR"].astype(str).to_numpy()
        segment = self._segments(truth, chrom, snps["START"].to_numpy())
        tumour = truth.copies_at(chrom, snps["START"].to_numpy())[:, 1:, :]
        allelic = (segment >= 0) & np.any(tumour[..., 0] != tumour[..., 1], axis=1)
        self.segment, self.block = segment[allelic], block[allelic]
        self.orientation = orientation[allelic]
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


def snps_phased(p: Phasing) -> float:
    return float(np.mean(p.correct))


def blocks_phased(p: Phasing) -> float:
    return float(pd.Series(p.correct).groupby(p.block).all().mean())


PHASING: dict[str, tuple[Callable[[Phasing], float], str]] = {
    "SNPs phased correctly": (snps_phased, ">="),
    "blocks phased correctly (every SNP)": (blocks_phased, ">="),
}
"""Name -> (score, direction). A SNP phased against its segment's majority fails both."""

RECORDED_PHASING: dict[str, dict[str, float]] = {
    "2d4ce9a9": {"SNPs phased correctly": 0.9987, "blocks phased correctly (every SNP)": 0.9966},
}
"""Fixture hash -> each phasing score as its capture records it, 4 decimals."""


@pytest.fixture(scope="module")
def phasing(sim: Any, truth: Truth) -> Phasing:
    return Phasing(sim, truth)


def bounded(recorded: float, direction: str) -> float:
    return recorded - MARGIN if direction == ">=" else recorded + MARGIN


@pytest.mark.parametrize("name", list(PHASING))
def test_phasing(phasing: Phasing, sim_hash: str, name: str) -> None:
    """A phasing score against the planted phase, at its floor (or ceiling) for this fixture."""
    score, direction = PHASING[name]
    value = score(phasing)
    recorded = RECORDED_PHASING[sim_hash][name]
    limit = bounded(recorded, direction)
    print(f"{sim_hash} {name}: {value:.4f} (recorded {recorded:.4f}, bound {limit:.4f})")
    assert value >= limit if direction == ">=" else value <= limit


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
}
"""Fixture hash -> each class's exact-recovery fraction as its capture records it, 4 decimals."""

MOST = 0.5
"""A failing class's row asserts that most of its clone-bins are exact, so a fix fails the strict xfail."""

FAILING: dict[tuple[str, str], str] = {
    ("2d4ce9a9", "balanced"): ("Ticket#903: the RDR fit holds the planted (2, 2) at neutral depth (fitted mu / the neutral "
                               "bins' 1.015), so it decodes (1, 1)"),
}
"""(fixture hash, class) -> why the class is recovered at 0.0: the ticket and the mechanism measured on the capture."""


@pytest.mark.parametrize("name", list(CLASSES))
def test_integer_copies_per_class(sim: Any, truth: Truth, sim_hash: str, name: str, request: pytest.FixtureRequest) -> None:
    """One class's exact (A, B) recovery at its floor: fails where the run loses a class the pooled copy scores hide,
    e.g. LOH decoded heterozygous. A class recovered at 0.0 asserts `MOST` under a strict xfail citing its ticket."""
    value, count = exact_per_class(sim, truth)[name]
    recorded = RECORDED_COPIES[sim_hash][name]
    assert count > 0, f"{sim_hash}: no {name} clone-bin planted"
    print(f"{sim_hash} {name}: {value:.4f} over {count} clone-bins (recorded {recorded:.4f})")
    reason = FAILING.get((sim_hash, name))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
        assert value >= MOST
    assert value >= recorded - MARGIN
