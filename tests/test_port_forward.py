"""`docs/port-forward.md` is what `scripts.port_forward` writes (T- #617 WP11)."""

from __future__ import annotations

import pytest

from scripts.port_forward import OUT, STAGES, render, rows
from tests.source_graph import tables


@pytest.mark.infra
def test_the_committed_table_is_the_generated_one() -> None:
    """A row added, removed or re-tested regenerates the table in the same change."""
    assert OUT.read_text() == render(), "run python -m scripts.port_forward"


@pytest.mark.infra
def test_every_row_is_one_cnaster_function_in_a_stage() -> None:
    expected = sum(len(swaps) for swaps in tables().values())
    found = rows()

    assert len(found) == expected
    assert {row[1] for row in found} <= {name for name, _ in STAGES}
