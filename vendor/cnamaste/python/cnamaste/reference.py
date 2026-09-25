import logging
import re

import pandas as pd
import polars as pl
import pyranges as pr

from cnamaste.config import get_global_config, start_time
from cnamaste.logger import get_logger
from typing import Any

logger = get_logger(__name__, start_time=start_time)


cancer_gene_patterns = (
    r"^MT-",  # Mitochondrial genes
    r"^S100A6$",  # Exact match for S100A6
    r"^B2M$",  # Exact match for B2M
    r"^HLA-B$",  # Exact match for HLA-B
    r"^TPT1$",  # Exact match for TPT1
    r"^FTL$",  # Exact match for FTL
    r"^FTH1$",  # Exact match for FTH1
    r"^RPL",  # Ribosomal protein L genes
    r"^RPS",  # Ribosomal protein S genes
)

pl.Config.set_tbl_cols(-1)


def exp_cancer_gene(gene_name):
    return any(re.match(pattern, gene_name) for pattern in cancer_gene_patterns)


AUTOSOMES = [f"chr{index}" for index in range(1, 23)]
"""The contigs `cnamaste` keeps. Sex chromosomes and the mitochondrion are not
among them, which is upstream's choice and carried across unchanged."""


def get_reference_genes(hgtable_file: str) -> Any:
    """What `cnamaste.reference.get_reference_genes` returns, read with `polars`.

    The GTF branch is delegated rather than reimplemented: `cnamaste` guards it
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


def get_reference_recomb_rates(geneticmap_file):
    """
    Attributes
    ----------
    chr_pos_vector : list of pairs
        list of (chr, pos) pairs of SNPs
    """
    df = pd.read_csv(geneticmap_file, header=0, sep="\t")
    df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    # NB drop "chr" prefix.
    df["chrom"] = df.chrom.str.replace("chr", "")

    df = df.sort_values(by=["chrom", "pos"])

    logger.info_once(
        f"Read reference recombination rates from {geneticmap_file}:\n{df.head()}"
    )

    return df
