"""Every swap row has a counting referee, or is declared without one (#517 E5).

`CLAUDE.md`: only `end2end` and `oracle` count, and they are aimed at the
whole of `run_cnaster` component by component. A row is refereed if a test
carrying one of the two reaches it by name -- in its body, or in a function of
its own module it calls (`tests.source_graph.counting_mentions`). A whole-run
test reaches every row and names none, so it does not count here: it cannot
say which stage was right.

Report first, then gating: `UNCOUNTED` is the rows with no referee today, so
this fails only when the set changes. A new row arrives with a referee or
with an entry here; a referee written lands with its entry removed.
"""

from __future__ import annotations

import pytest

from tests.source_graph import counting_mentions, tables

UNCOUNTED = frozenset(
    {
        "FIGURE_SWAPS:plot_ascn_legend",
        "FIGURE_SWAPS:plot_clones_genomic",
        "FIGURE_SWAPS:plot_clones_spatial",
        "FIGURE_SWAPS:plot_copy_number_profile",
        "FIGURE_SWAPS:write_fig",
        "PLOT_OFF_SWAPS:write_fig",
        "SWAPS:assign_initial_blocks",
        "SWAPS:create_bin_ranges",
        "SWAPS:get_aggregated_barcodes",
        "SWAPS:get_reference_genes",
        "SWAPS:initialize_rectangular_clones",
        "SWAPS:summarize_blocks",
    }
)
"""12 of 32 rows. The five figure rows have no truth to count against."""


def _uncounted() -> set[str]:
    mentions = counting_mentions()
    out = set()

    for table, swaps in tables().items():
        for swap in swaps:
            name = swap.name
            attribute = swap.replacement.rpartition(".")[2].rpartition(":")[2]

            if not (mentions.get(name) or mentions.get(attribute)):
                out.add(f"{table}:{name}")

    return out


@pytest.mark.infra
def test_the_rows_without_a_counting_referee_are_the_declared_ones() -> None:
    found = _uncounted()

    assert found == set(UNCOUNTED), (
        f"newly uncounted: {sorted(found - UNCOUNTED)}; "
        f"now refereed, remove: {sorted(UNCOUNTED - found)}"
    )
