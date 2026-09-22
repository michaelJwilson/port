"""`cnaster.omics`' block construction and its summaries (#250).

`blocks` carries the four entry points `SWAPS` installs; `summaries` carries
`summarize_blocks`, which was a separate module for no reason a reader of
`cnaster/omics.py` could have guessed.

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.omics.blocks` and a reader can open `cnaster.omics` and find it.
"""

from __future__ import annotations

from port.patch.omics.blocks import (
    assign_initial_blocks,
    block_of_row,
    form_gene_snp_table,
    merged_gene_intervals,
    preceding_gene,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from port.patch.omics.summaries import (
    block_summary,
    summarize_blocks,
)

__all__ = [
    "assign_initial_blocks",
    "block_of_row",
    "block_summary",
    "form_gene_snp_table",
    "merged_gene_intervals",
    "preceding_gene",
    "summarize_blocks",
    "summarize_counts_for_bins",
    "summarize_counts_for_blocks",
]
