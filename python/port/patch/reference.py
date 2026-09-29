"""`cnaster.reference.get_reference_genes`, read with `polars` (#185).

The function reads one tab-separated file and hands back six columns. The
return is **bitwise** what `cnaster` returns -- values, index, column order
and dtypes -- which `tests/test_reference_patch.py` pins. It is offered as a
simplification, not a speedup.

Two costs, not one:

*   `pd.read_csv` builds a frame whose every column is then taken out again
    with `.to_numpy()`. `polars` reads the same file with a multi-threaded
    parser and hands back the arrays directly, so the frame in the middle
    never exists.
*   `[int(x[3:]) for x in df_hgtable.Chromosome.to_numpy()]` is a Python loop
    over every transcript, slicing a string and parsing an integer. It is a
    `str.slice` and a `cast` here.

**`polars` is not a new dependency to install** -- `cnaster` already requires
it and uses it for the Visium HD spatial reads -- but it is a new one to
*declare*, and #185 carries the permission and the reasoning.

**`pyarrow` is a real install, and it buys memory rather than time.** The
whole transform is one `select` and one `to_pandas()`, so the numeric columns
cross by Arrow buffer instead of being pulled out as six `numpy` arrays and
rebuilt into a dict.

Measured: `docs/measurements.md`, `port.patch.reference`.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from cnaster.config import get_global_config, start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

AUTOSOMES = [f"chr{index}" for index in range(1, 23)]
"""The contigs `cnaster` keeps. Sex chromosomes and the mitochondrion are not
among them, which is upstream's choice and carried across unchanged."""


def get_reference_genes(hgtable_file: str) -> Any:
    """What `cnaster.reference.get_reference_genes` returns, read with `polars`.

    The GTF branch is delegated rather than reimplemented: `cnaster` guards it
    with `if True or config.run.legacy`, so it is unreachable, and a patch
    that rewrote unreachable code would be measuring nothing.
    """
    # NB upstream guards its GTF branch with `if True or config.run.legacy`,
    #    so only this one is reachable. `get_global_config()` is still read,
    #    because a caller reaching here without a config installed is a
    #    misconfiguration upstream would raise on too.
    get_global_config()

    logger.info_once(f"Assuming legacy gene annotation={hgtable_file}.")

    # NB `snp_id` and `is_interval` are literals upstream sets on the frame
    #    rather than columns of the file, and they carry `object` and `bool`
    #    through Arrow as they do through `numpy` -- which the bitwise test
    #    checks, dtypes included, because a `None` column is exactly where a
    #    conversion is free to choose a different one.
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
