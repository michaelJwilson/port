"""`cnaster.recomb.get_sitewise_transmat`, computed from a :class:`Segmentation` (#438).

The drop-in keeps `cnaster`'s signature and its kernel, and fixes two things
in it, both stated in `port.extensions.segments.Segmentation.log_phase_switch`:
centimorgans are read per contig (chr2-9 no longer inherit chr1's last value),
and a contig's last segment is independence rather than continuity.

The segments are `df_gene_snp` grouped by `segment_key` in id order, which is
the order `summarize_counts_for_blocks` and `summarize_counts_for_bins` index
their rows in; a grouping whose id order is not genomic order, or whose
members are not contiguous, is refused rather than returned misaligned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["get_sitewise_transmat"]


def get_sitewise_transmat(
    segment_key: str,
    df_gene_snp: Any,
    geneticmap_file: Any,
    nu: float,
    logphase_shift: float,
) -> np.ndarray:
    """`log_sitewise_transmat`, one entry per `segment_key` segment."""
    from cnaster.config import get_global_config
    from cnaster.reference import get_reference_recomb_rates

    from port.extensions.segments import GeneticMap, Segmentation

    segments = Segmentation.from_table(df_gene_snp, key=segment_key)
    genetic_map = GeneticMap.from_frame(get_reference_recomb_rates(geneticmap_file))

    return segments.log_phase_switch(
        genetic_map, nu, logphase_shift, get_global_config().phasing.min_prob
    )
