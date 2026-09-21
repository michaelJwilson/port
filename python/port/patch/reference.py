"""`cnaster.reference.get_reference_genes`, read with `polars` (#185).

**13.3x and 55.2 MB to 23.3 MB at 250,000 transcripts, which is a human
reference's size.** The function reads one tab-separated file and hands back
six columns, and `pandas` is 367 ms of that on its own.

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
rebuilt into a dict. Measured at 250,000 transcripts, warm, best of five:

    cnaster                430.65 ms   49.54 MB
    column by column        42.84 ms   23.28 MB   10.1x, 2.1x less
    through Arrow           60.53 ms   15.49 MB    7.1x, 3.2x less

and at the dev instance's 1,213 transcripts, 3.94 / 1.90 / 4.03 ms.

So it is **1.41x slower than the route it replaces at a stress size and a
wash at a gate size**, for 33 per cent less peak and a function that is one
expression rather than seven. `CLAUDE.md` is what decides which of those
wins: a speedup claim needs 2x at a stress size and this is not offered as
one; a simplification needs evidence of equivalence, which is the bitwise
test. The cost is stated rather than buried -- `pyarrow` is 152 MB installed
and the largest wheel in the environment, against 7.8 MB of peak saved on a
stage that is 0.17 s of a whole run.

The return is **bitwise** what `cnaster` returns -- values, index, column
order and dtypes -- which `tests/test_reference_patch.py` pins.
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
