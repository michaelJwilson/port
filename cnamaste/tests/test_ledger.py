"""`docs/ledger/ledger.tsv` against the staged files: each fixture's latest row is what its staged file scores."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from cnamaste import ledger
from cnamaste.audit import PROJECT
from cnamaste.audit.capture import Capture
from cnamaste.audit.sims import NAMES, staged


@pytest.mark.parametrize("fixture_hash", list(NAMES))
def test_the_latest_row_is_what_the_staged_file_scores(fixture_hash: str) -> None:
    """Every column but `date`, as printed, on all four staged files: a re-capture without its row, or a scorer change,
    fails here."""
    row = ledger.latest(ledger.read()).get(fixture_hash)
    assert row is not None, f"ledger.tsv has no row for {fixture_hash}: run `python -m cnamaste.ledger` on its staged file"
    found = ledger.scored(PROJECT / staged(fixture_hash))
    assert {k: row[k] for k in found} == found


OUT_OF_DATE = {h: f"Ticket#438: {NAMES[h]} was captured at efb2e9e, before contigs were ordered 1, 2, ..., 22; re-capture"
               for h in ("8797710b", "23acef50", "e4bbe03f")}
"""Fixture hash -> why its staged file predates the code: a re-capture removes its entry. Only shrinks."""


@pytest.mark.parametrize("fixture_hash", [pytest.param(h, marks=pytest.mark.xfail(strict=True, reason=OUT_OF_DATE[h])) if h in OUT_OF_DATE else h
                                          for h in NAMES])
def test_the_staged_file_is_current(fixture_hash: str) -> None:
    """The staged phase-switch kernel has no contig with every bin at the min_prob floor: a capture from before
    Ticket#438's fix puts all of chr2-9 there."""
    sim = Capture(PROJECT / staged(fixture_hash))
    log = np.asarray(sim.stored("04_bins/get_sitewise_transmat/out"))
    contig = sim.stored("04_bins/create_bin_ranges/out").groupby("bin_id")["CHR"].first().to_numpy()
    phasing = yaml.safe_load(sim.config["yaml"])["phasing"]
    floored = np.isclose(log, np.log(phasing["min_prob"]) - phasing["logphase_shift"])
    assert not [c for c in range(1, 23) if floored[contig == c].size and floored[contig == c].all()]
