"""`docs/ledger/ledger.tsv` against the staged files: each fixture's latest row is what its staged file scores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("cnamaste_ledger", PROJECT / "docs" / "ledger" / "ledger.py")
assert _SPEC is not None and _SPEC.loader is not None
ledger = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ledger)

from audit.sims import SUPPORTED_SIMS  # noqa: E402


def test_the_latest_row_is_what_the_staged_file_scores(sim_hash: str) -> None:
    """Every column but `date`, as printed: a re-capture without its row, or a scorer change, fails here."""
    row = ledger.latest(ledger.read()).get(sim_hash)
    assert row is not None, f"ledger.tsv has no row for {sim_hash}: run docs/ledger/ledger.py on its staged file"
    found = ledger.scored(PROJECT / SUPPORTED_SIMS[sim_hash].staged)
    assert {k: row[k] for k in found} == found
