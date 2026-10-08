"""Replaces `cnaster.recomb.get_sitewise_transmat`, computed from a `Segmentation` (#438).

Fixes: centimorgans read per contig, and a contig's last segment is
independence. A segment ends at its last gene's `END`, not a SNP's.
Misaligned or non-contiguous labellings are refused.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

__all__ = ["get_sitewise_transmat", "release"]


def get_sitewise_transmat(
    segment_key: str,
    df_gene_snp: Any,
    geneticmap_file: Any,
    nu: float,
    logphase_shift: float,
    *,
    composable: bool = False,
) -> np.ndarray:
    """`log_sitewise_transmat`, one entry per `segment_key` segment; `composable` is #449's law."""
    from cnaster.config import get_global_config

    from port.extensions.segments import observe

    segments = observe(df_gene_snp, segment_key)
    genetic_map = _genetic_map(str(geneticmap_file))

    return segments.log_phase_switch(
        genetic_map,
        nu,
        logphase_shift,
        get_global_config().phasing.min_prob,
        composable=composable,
    )


def _genetic_map(path: str) -> Any:
    """The map at `path`, cached by path and modification time (#438 D4)."""
    from pathlib import Path

    return _read_map(path, Path(path).stat().st_mtime_ns)


def release() -> None:
    """Drop the cached maps; `port.pipeline.patched` calls this on exit (#617)."""
    _read_map.cache_clear()


@functools.lru_cache(maxsize=4)
def _read_map(path: str, mtime: int) -> Any:  # noqa: ARG001 -- the cache key
    from cnaster.reference import get_reference_recomb_rates

    from port.extensions.segments import GeneticMap

    return GeneticMap.from_frame(get_reference_recomb_rates(path))
