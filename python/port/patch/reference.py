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
*declare*, and #185 carries the permission and the reasoning. What it is not
is a route back to `pandas`: `DataFrame.to_pandas()` needs `pyarrow`, which
is not installed, so the frame is rebuilt column by column through `numpy`.
That is the conversion the 13.3x is measured with.

The return is **bitwise** what `cnaster` returns -- values, index, column
order and dtypes -- which `tests/test_reference_patch.py` pins.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
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

    table = pl.read_csv(hgtable_file, separator="\t").filter(
        pl.col("chrom").is_in(AUTOSOMES)
    )

    return pd.DataFrame(
        {
            "CHR": table["chrom"].str.slice(3).cast(pl.Int64).to_numpy(),
            "START": table["cdsStart"].to_numpy(),
            "END": table["cdsEnd"].to_numpy(),
            "snp_id": None,
            "gene": table["name2"].to_numpy(),
            "is_interval": True,
        }
    )
