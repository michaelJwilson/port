"""Every swap row has a counting referee (`end2end` or `oracle` naming it), or is declared
without one (#517 E5).

`UNCOUNTED` lists rows without one today; the test fails when that set changes.
"""

from __future__ import annotations

import pytest

from tests.source_graph import counting_mentions, tables

UNCOUNTED = frozenset(
    {
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
"""11 of 31 rows. The four figure rows have no truth to count against."""


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
