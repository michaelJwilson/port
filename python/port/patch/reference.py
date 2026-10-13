"""Replaces `cnaster.reference.get_reference_genes`, read with `polars` (#185).

Returns bitwise what `cnaster` returns (values, index, column order, dtypes);
a simplification, not a speedup.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from cnaster.config import get_global_config, start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

AUTOSOMES = [f"chr{index}" for index in range(1, 23)]
"""The contigs `cnaster` keeps: autosomes only, as upstream."""


def get_reference_genes(hgtable_file: str) -> Any:
    """`cnaster.reference.get_reference_genes`; upstream's GTF branch is unreachable."""
    # NB read for its raise: upstream fails without an installed config too.
    get_global_config()

    logger.info_once(f"Assuming legacy gene annotation={hgtable_file}.")

    # NB `snp_id`/`is_interval` are literals; their `object`/`bool` dtypes are pinned.
    frame: Any = (
        pl.read_csv(hgtable_file, separator="\t")
        .filter(pl.col("chrom").is_in(AUTOSOMES))
        .select(
            pl.col("chrom").str.slice(3).cast(pl.Int64).alias("CHR"),
            pl.col("cdsStart").alias("START"),
            pl.col("cdsEnd").alias("END"),
            pl.lit(None).alias("snp_id"),
            pl.col("name2").alias("gene"),
            pl.lit(True).alias("is_interval"),
        )
        .to_pandas()
    )
    return frame
