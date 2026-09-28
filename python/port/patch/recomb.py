"""`cnaster.recomb.get_sitewise_transmat`, computed from a :class:`Segmentation` (#438).

The drop-in keeps `cnaster`'s signature and its kernel, and fixes two things
in it, both stated in `port.extensions.segments.Segmentation.log_phase_switch`:
centimorgans are read per contig (chr2-9 no longer inherit chr1's last value),
and a contig's last segment is independence rather than continuity.

The segments are `df_gene_snp`'s gene rows labelled by `segment_key`, in id
order, which is the order `summarize_counts_for_blocks` and
`summarize_counts_for_bins` index their rows in; a labelling whose id order
is not genomic order, whose genes are not contiguous, or with a segment that
holds no gene, is refused rather than returned misaligned. A segment ends at
its last gene's `END`, where `cnaster` reads the last row's, a SNP.
"""

from __future__ import annotations

import functools
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

    from port.extensions.segments import observe

    segments = observe(df_gene_snp, segment_key)
    genetic_map = _genetic_map(str(geneticmap_file))

    return segments.log_phase_switch(
        genetic_map, nu, logphase_shift, get_global_config().phasing.min_prob
    )


def _genetic_map(path: str) -> Any:
    """The map at `path`, read once per file version (#438 D4).

    `run_cnaster` asks for the kernel four times and `cnaster` re-reads the
    map each time. Keyed on the path and its modification time, so a map
    rewritten between runs in one process is read again.
    """
    from pathlib import Path

    return _read_map(path, Path(path).stat().st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _read_map(path: str, mtime: int) -> Any:  # noqa: ARG001 -- the cache key
    from cnaster.reference import get_reference_recomb_rates

    from port.extensions.segments import GeneticMap

    return GeneticMap.from_frame(get_reference_recomb_rates(path))
