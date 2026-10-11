"""`docs/ledger/ledger.tsv` against the staged files: each fixture's latest row is what its staged file scores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from audit.sims import NAMES, staged

PROJECT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("cnamaste_ledger", PROJECT / "docs" / "ledger" / "ledger.py")
assert _SPEC is not None and _SPEC.loader is not None
ledger = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ledger)


@pytest.mark.parametrize("fixture_hash", list(NAMES))
def test_the_latest_row_is_what_the_staged_file_scores(fixture_hash: str) -> None:
    """Every column but `date`, as printed, on all four staged files: a re-capture without its row, or a scorer change,
    fails here."""
    row = ledger.latest(ledger.read()).get(fixture_hash)
    assert row is not None, f"ledger.tsv has no row for {fixture_hash}: run docs/ledger/ledger.py on its staged file"
    found = ledger.scored(PROJECT / staged(fixture_hash))
    assert {k: row[k] for k in found} == found
