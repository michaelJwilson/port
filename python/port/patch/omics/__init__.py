"""Replaces `cnaster.omics`' block construction and summaries (#250); re-exports `blocks` and `summaries`."""

from __future__ import annotations

from port.patch.omics.blocks import (
    assign_initial_blocks,
    block_of_row,
    create_bin_ranges,
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
    "create_bin_ranges",
    "form_gene_snp_table",
    "merged_gene_intervals",
    "preceding_gene",
    "summarize_blocks",
    "summarize_counts_for_bins",
    "summarize_counts_for_blocks",
]
