"""`docs/ledger/ledger.tsv` against the staged files: each fixture's latest row is what its staged file scores."""

from __future__ import annotations

import pytest
from cnamaste import ledger
from cnamaste.audit import PROJECT
from cnamaste.audit.sims import NAMES, staged


@pytest.mark.parametrize("fixture_hash", list(NAMES))
def test_the_latest_row_is_what_the_staged_file_scores(fixture_hash: str) -> None:
    """Every column but `date`, as printed, on all four staged files: a re-capture without its row, or a scorer change,
    fails here."""
    row = ledger.latest(ledger.read()).get(fixture_hash)
    assert row is not None, f"ledger.tsv has no row for {fixture_hash}: run `python -m cnamaste.ledger` on its staged file"
    found = ledger.scored(PROJECT / staged(fixture_hash))
    assert {k: row[k] for k in found} == found
