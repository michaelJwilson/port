import copy
import logging
import os
from collections import namedtuple
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp

# from cnamaste.utils import cacher
from sklearn.neighbors import LocalOutlierFactor

from cnamaste.config import get_global_config, start_time
from cnamaste.filter import get_filter_genes, get_filter_ranges
from cnamaste.he import get_he_image
from cnamaste.logger import get_logger
from cnamaste.reference import exp_cancer_gene
from typing import Any

logger = get_logger(__name__, start_time=start_time)

pl.Config.set_tbl_cols(-1)


def construct_df_clone_label(barcodes, coords, assignment, tumor_prop=None):
    """
    Construct a data frame with assigned clone label for all samples.

    Parameters
    ----------
    barcodes : list of str
        List of barcodes corresponding to the samples.
    coords : np.ndarray
        Array of coordinates for the samples.
    assignment : list or np.ndarray
        List or array of clone assignments for the samples.
    tumor_prop : np.ndarray, optional
        Array of tumor proportions for the samples. Default is None.

    Returns
    -------
    pd.DataFrame
        Data frame containing sample_id, x, y, clone_label, and optionally tumor_proportion.
    """
    df_clone_label = pd.DataFrame(
        {
            "sample_id": [barcode.split("_")[-1] for barcode in barcodes],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "clone_label": assignment,
        },
        index=barcodes,
    )

    if tumor_prop is not None:
        df_clone_label["tumor_proportion"] = tumor_prop

    # Sort by (sample_id, (x,y)).
    # df_clone_label = df_clone_label.groupby("sample_id", group_keys=False).apply(
    #     lambda g: g.sort_values(["x", "y"])
    # )

    df_clone_label = df_clone_label.sort_values(["sample_id", "x", "y"])

    return df_clone_label


def get_sample_sheet(sample_sheet_path):
    """
    Read the sample sheet from a CSV file; expects

    'bam',
    'sample_id',
    'spaceranger_dir',
    snp_dir',

    in order to retrieve called snps and space-ranger
    derived transcript counts.

    Parameters
    ----------
    sample_sheet_path : str
        Path to the sample sheet CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sample metadata.
    """
    df_meta = pd.read_csv(sample_sheet_path, sep=r"\s+", comment="#")

    required_columns = {"bam", "sample_id", "spaceranger_dir", "snp_dir"}

    # TODO
    assert required_columns.issubset(
        df_meta.columns
    ), f"sample_sheet has columns {df_meta.columns} which missese the required: {required_columns - set(df_meta.columns)}:\n{df_meta}"

    logger.info(
        f"Input sample_sheet_path={sample_sheet_path} contains {len(df_meta)} samples:\n{df_meta}"
    )

    return df_meta


def get_barcodes(barcode_file):
    # NB see https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/calicost.smk#L32
    found_file = None

    for ext in (".tsv.gz", ".tsv", ".txt", ".txt.gz"):
        candidate = barcode_file.replace(".txt", ext)

        if os.path.exists(candidate):
            found_file = candidate
            break

    if found_file is None:
        raise RuntimeError(
            f"Failed to retrieves barcodes.txt (or its known alternative extensions: ().tsv.gz, .tsv, .txt.gz) from {barcode_file})"
        )

    # NB spot bar-codes combined across slices (TBC).
    if found_file.endswith(".tsv.gz"):
        df_barcodes = pd.read_csv(
            found_file,
            sep="\t",
            header=None,
            names=["combined_barcode"],
            compression="gzip",
        )
    else:
        # NB default for .txt
        df_barcodes = pd.read_csv(found_file, header=None, names=["combined_barcode"])

    return df_barcodes


# TODO check (e.g. sample sheet with john):
#             AAACAAGTATCTCCCA-1_HT112C1-U1 == {spot}-1_{sample_id}-{slice}.
def get_aggregated_barcodes(barcode_file, known_sample_id=None):
    df_barcodes = get_barcodes(barcode_file)

    sample_id_defined = df_barcodes.combined_barcode.str.contains("_").all()

    # NB Visium 10x provided spot barcode for each slice.
    df_barcodes["barcode"] = [
        x.split("_")[0] for x in df_barcodes.combined_barcode.to_numpy()
    ]

    # NB {sample_id}-{slice}
    if sample_id_defined:
        df_barcodes["sample_id"] = [
            x.split("_")[-1] for x in df_barcodes.combined_barcode.to_numpy()
        ]
    else:
        logger.warning(
            f"Unable to resolve sample_ids from aggregated barcodes.  Assuming known sample_id={known_sample_id}."
        )
        df_barcodes["sample_id"] = known_sample_id or "UNKNOWN"

    # TODO HACK assumes {sample_id}-{slice} is not provided, i.e. single slice.
    df_barcodes["barcode"] = df_barcodes.combined_barcode
    df_barcodes["sample_id"] = known_sample_id

    # TODO sample ids currently slice, e.g. U1;
    logger.info(
        f"Input aggregated barcode file {barcode_file} with {df_barcodes.shape[0]:_} barcodes for all samples/bams, e.g.\n{df_barcodes.head()}\n"
    )

    return df_barcodes


def get_spatial_positions(spaceranger_dir, filter_in_tissue=True):
    # TODO x,y vs row,col?  sub-pixel position?
    names = ("barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col")

    if Path(
        f"{spaceranger_dir}/spatial/tissue_positions.csv",
    ).exists():
        df_this_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions.csv",
            sep=",",
            header=0,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions.csv")

    elif Path(f"{spaceranger_dir}/spatial/tissue_positions_list.csv").exists():
        df_this_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions_list.csv",
            sep=",",
            header=None,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions_list.csv")

    elif Path(f"{spaceranger_dir}/spatial/tissue_positions.parquet").exists():
        # NB 11,222,500 rows vs 4,992 rows for visium HD.
        #    see https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
        #
        #
        #    native columns :  barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres.
        #
        #    visium HD : x2 6.5mm capture areas, array_row = row coordinate of the spot in the array from 0 to 77.
        #                                        array_col = given orange crate [sic], i.e. hexagonal, for 6.55 this uses even numbers from 0 to 126 for even rows
        #                                                   and odd numbers from 1 to 127 for odd rows.
        #                                        pxl_row_in_fullres: The row pixel coordinate of the center of the spot in the full resolution image
        #                                        pxl_col_in_fullres: The column pixel coordinate of the center of the spot in the full resolution image. This is the x coordinate in pixel space.
        """
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/tissue_positions.parquet")
            .rename({"pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
            .select(["barcode", "in_tissue", "x", "y"])
            # .filter(pl.col("in_tissue") == True)
            .collect()
            .with_columns(pl.col("barcode").alias("square_002um"))
        )

        # NB native columns: square_002um, square_008um, square_016um, cell_id, in_nucleus, in_cell
        # TODO CHECK in_cell
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/barcode_mappings.parquet")
            .filter(pl.col("in_cell") == True)
            .collect()
            .join(
                df_this_pos.select(["square_002um", "x", "y", "in_tissue"]),
                on="square_002um",
                how="left",
            )
            .select(["square_002um", "cell_id", "x", "y", "in_tissue"])
            .group_by("cell_id")
            .agg(
                [
                    pl.col("x").mean().alias("x"),
                    pl.col("y").mean().alias("y"),
                    pl.col("square_002um").n_unique().alias("num_square_002um"),
                    pl.col("in_tissue").cast(pl.Boolean).any().alias("in_tissue"),
                ]
            )
            .with_columns(pl.col("cell_id").alias("barcode"))
        )
        """
        # TODO invert y.
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/tissue_positions.parquet")
            .rename({"pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
            .filter(pl.col("in_tissue") == True)
            .collect()
        )

        logger.info(
            f"Read {spaceranger_dir}/spatial/tissue_positions.parquet:\n{df_this_pos}"
        )

        df_this_pos = df_this_pos.to_pandas()
    else:
        logger.error(f"No spatial coordinate file @ {spaceranger_dir}.")
        raise RuntimeError()

    # TODO alignment defined for in_tissue == True only?
    if filter_in_tissue:
        logger.warning(
            f"Filtering spatial positions to in_tissue == True (retained {100. * np.mean(df_this_pos.in_tissue)}%)."
        )

        result = df_this_pos[df_this_pos.in_tissue == True]
    else:
        result = df_this_pos

    # NB x,y positions for each barcode in this sample.
    return result


def get_spaceranger_counts(spaceranger_dir):
    config = get_global_config()
    filtered_feature_name = config.visium.filtered_feature_name

    supported_types = ["filtered_feature_bc_matrix", "filtered_feature_cell_matrix"]

    if filtered_feature_name not in supported_types:
        logger.error(
            f"{filtered_feature_name} is not supported; expected one  of {supported_types}"
        )
        raise ValueError()

    # NB see https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html
    if Path(f"{spaceranger_dir}/{filtered_feature_name}.h5").exists():
        adatatmp = sc.read_10x_h5(
            f"{spaceranger_dir}/{filtered_feature_name}.h5",  # gex_only=True
        )
        logger.info(f"Reading {spaceranger_dir}/{filtered_feature_name}.h5")

    elif Path(f"{spaceranger_dir}/{filtered_feature_name}.h5ad").exists():
        adatatmp = sc.read_h5ad(
            f"{spaceranger_dir}/{filtered_feature_name}.h5ad",  # gex_only=True
        )
        logger.info(f"Reading {spaceranger_dir}/{filtered_feature_name}.h5ad")

    else:
        logging.error(
            f"{spaceranger_dir} directory does not have a {filtered_feature_name}.h5(ad)!"
        )
        raise RuntimeError()

    # TODO comment on adatatmp.x (nobs x nvars for space ranger, i.e. barcodes x gene transcripts).
    adatatmp.layers["count"] = adatatmp.X.toarray()

    is_nan = np.isnan(adatatmp.layers["count"])

    logger.info(f"Found {100.0 * np.mean(is_nan):.3f}% NaN counts in anndata.")

    # NB replace nan with 0 and cast to int.
    if np.any(is_nan):
        adatatmp.layers["count"][is_nan] = 0

    # TODO CHECK
    adatatmp.layers["count"] = adatatmp.layers["count"].astype(int)

    # e.g. duplicated:  TBCE  2, LINC01238  2.3; why?
    # duplicated_mask = adatatmp.var_names.duplicated(keep=False)
    # non_unique_vars = adatatmp.var_names[duplicated_mask]

    # duplicate_counts = non_unique_vars.value_counts()

    logger.info(
        f"Read transcript counts of shape {adatatmp.shape}, i.e. (barcodes, genes) from {spaceranger_dir}"
    )

    logger.info(
        f"Example names for {len(adatatmp.obs_names):_} barcodes:\n{adatatmp.obs_names[:5]}"
    )
    logger.info(
        f"Example names for {len(adatatmp.var_names):_} genes:\n{adatatmp.var_names[:5]}"
    )

    # NB var names made unique by appending an index string,
    #    see https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.var_names_make_unique.html
    adatatmp.var_names_make_unique()

    # NB data matrix X (ndarray/csr matrix, dask ...): observations/cells are named by their barcode and variables/genes by gene name.
    return adatatmp


# DEPRECATE
# TODO massively inefficient?
# NB mirrors https://github.com/raphael-group/CalicoST/blob/c1abcae3e3657e01e547ee4529e3b9d039221453/src/calicost/utils_IO.py#L127
def get_alignments(alignment_files, df_meta, df_agg_barcode, significance=1.0e-6):
    if alignment_files is None:
        return None

    row_ind, col_ind = [], []
    dat = []

    offset = 0

    for i, f in enumerate(alignment_files):
        pi = np.load(f)

        # normalize p such that max( row_sums(pi), cols_sum(pi) ) = 1;
        # TODO? max alignment weight = 1
        pi = pi / np.max(np.append(np.sum(pi, axis=0), np.sum(pi, axis=1)))

        # NB assumes alignments ordered by df_meta sample ids.
        sname1 = df_meta.sample_id.to_numpy()[i]
        sname2 = df_meta.sample_id.to_numpy()[i + 1]

        assert pi.shape[0] == np.sum(df_agg_barcode["sample_id"] == sname1)
        assert pi.shape[1] == np.sum(df_agg_barcode["sample_id"] == sname2)

        # NB for each spot s in sname1, select {t: spot t in sname2 and pi[s,t] >= np.max(pi[s,:])} as the corresponding spot in the other slice
        for row in range(pi.shape[0]):
            row_max = np.max(pi[row, :])

            # NB their exists an element in the alignment of a sample 1 spot with significant probability (> significance)
            cutoff = row_max if row_max > significance else 1.0 + significance

            list_cols = np.where(pi[row, :] >= cutoff - significance)[0]

            row_ind += [offset + row] * len(list_cols)

            # NB zero_point = offset + pi.shape[0] +1 per col entry.
            col_ind += list(offset + pi.shape[0] + list_cols)

            dat += list(pi[row, list_cols])

        offset += pi.shape[0]

    across_slice_adjacency_mat = sp.csr_matrix(
        (dat, (row_ind, col_ind)), shape=(adata.shape[0], adata.shape[0])
    )

    # TODO symmetric by definition.
    across_slice_adjacency_mat += across_slice_adjacency_mat.T

    return across_slice_adjacency_mat


def map_unique_snps_enum(unique_snp_ids):
    """
    Given unique_snp_ids (array) of {contig}_{pos}_{r}_{a} for all snps,
    where r = a = 'N' is a unknown marker.

    map each to a unique id of the form {contig}_{pos}_{enum}, where enum
    allows for erroneous repeats, but is 0 in almost all cases.
    """
    # NB log the number of unique snps and warn on any repeats
    bonafide_unique_snps, cnts = np.unique(unique_snp_ids, return_counts=True)
    logger.info(
        f"Detected {len(bonafide_unique_snps)} unique snps from {len(unique_snp_ids)} input snp ids with dtype={unique_snp_ids.dtype}."
    )

    repeats = dict()

    if len(bonafide_unique_snps) != len(unique_snp_ids):
        for snp_id, count in zip(bonafide_unique_snps[cnts > 1], cnts[cnts > 1]):
            contig, pos, _, _ = snp_id.split("_")
            logger.warning(
                f"Detected repeated snp_id @ chr{contig}:{pos} with count={count}."
            )
            repeats[snp_id] = 0

    result = []

    for snp_id in unique_snp_ids:
        contig, pos, *_ = snp_id.split("_")

        if snp_id in repeats:
            enum = repeats[snp_id]
            repeats[snp_id] += 1
        else:
            enum = 0

        # TODO HACK JOHN
        contig = contig.replace("chr", "")

        new_snp_id = f"{contig}_{pos}_{enum}"
        result.append(new_snp_id)

    result = np.array(result, dtype=unique_snp_ids.dtype)

    logger.info(f"Mapped input snp ids to enum:\n{result[:5]}")

    return result


# @cacher("processed_input.hdf5")

ProcessedData = namedtuple(
    "ProcessedData",
    [
        "coords",
        "barcodes",
        "adata",
        "exp_counts",
        "cell_snp_Aallele",
        "cell_snp_Ballele",
        "unique_snp_ids",
        "across_slice_adjacency_mat",
    ],
)
"""`cnamaste`'s own return shape, declared here because it declares it inline.

Field for field the same, so a caller cannot tell the two apart by name. The
patch test compares by field rather than by type, since two `namedtuple`s of
the same shape are not the same class.
"""


def _spot_umis(counts: Any) -> np.ndarray:
    """Per-spot totals, whether the counts are dense or sparse.

    One helper rather than four call sites: `np.sum(matrix, axis=1)` on a
    `scipy` sparse matrix returns an `np.matrix` of shape `(n, 1)`, and the
    comparison that follows it then broadcasts into a matrix rather than a
    mask. Flattening here is what lets the caller be written once.
    """
    if sp.issparse(counts):
        return np.asarray(counts.sum(axis=1)).ravel()

    return np.asarray(np.sum(counts, axis=1)).ravel()


def _genes_expressed_in(counts: Any) -> np.ndarray:
    """How many spots express each gene.

    `cnamaste` writes `np.sum(adata.X > 0, axis=0)`, which materializes a second
    matrix of the same shape to count its non-zeros. `getnnz` counts the
    structural non-zeros already stored, so it allocates one vector.

    The two agree only where no stored entry is zero. A matrix read from
    `spaceranger` carries none -- `sc.read_10x_h5` stores what was counted --
    but a filtered view can, so the explicit count is kept as the fallback
    rather than assumed away.
    """
    if sp.issparse(counts):
        stored = counts.data
        if stored.size and not stored.all():
            return np.asarray((counts > 0).sum(axis=0)).ravel()

        return np.asarray(counts.getnnz(axis=0)).ravel()

    return np.asarray(np.sum(counts > 0, axis=0)).ravel()


def _load_allele_matrices(snp_dir: str) -> tuple[Any, Any]:
    """The A and B allele matrices, inflated on two threads.

    `sp.load_npz` is `zlib` inflate and little else: the pair is 20 MB on disk
    and 240 MB inflated, and at 2,500 spots it is 390 ms of the loader with
    the decompressor holding 238 of them. `zlib` releases the GIL, so the two
    files overlap on two threads for the cost of a pool.

    One thread per file rather than per member: the members of one file are a
    120 MB `data` against a 40 MB `indices` and three arrays under a kilobyte,
    so splitting inside a file schedules four idle workers behind one long
    one. The floor either way is the longest single member, since a `deflate`
    stream cannot be split.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = [f"{snp_dir}/cell_snp_{allele}allele.npz" for allele in ("A", "B")]

    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        a_matrix, b_matrix = pool.map(lambda path: sp.load_npz(path).tocsr(), paths)

    return a_matrix, b_matrix


def _without_nan(values: np.ndarray) -> np.ndarray:
    """`values` with any `NaN` replaced by zero, copying only if there is one.

    `np.nan_to_num` copies unconditionally and tests for the infinities too,
    which is 239 ms of the loader on an input that carries neither. The test
    is one pass over the values and the copy is skipped on the common case;
    the branch is kept rather than assumed away because `cnamaste` logs the
    `NaN` fraction, so a file that carries them is a file it expects.
    """
    nan = np.isnan(values)

    if not nan.any():
        return values

    logger.info(f"Found {100.0 * np.mean(nan):.3f}% NaN counts in anndata.")

    return np.where(nan, 0.0, values)


def _spaceranger_counts(
    spaceranger_dir: str, config: Any, *, sparse_counts: bool
) -> Any:
    """The transcript counts, with the integer cast done where they are stored.

    `cnamaste` writes the count layer with `adatatmp.X.toarray()`, tests it with
    `np.isnan`, and casts it with `astype(int)` -- three dense `(spots, genes)`
    arrays to cast values that `spaceranger` stored sparsely. At 2,500 spots
    and 5,983 genes that is 617 ms, 408 of it in the cast alone.

    The values are the same either way: a structural zero is not `NaN` and
    truncates to zero, so casting `.data` and casting the dense array agree
    entry for entry. What differs is that one of them allocates the shape and
    the other the stored values.

    This is the one place the patch reads a file `cnamaste`'s helper would have
    read, rather than calling that helper -- the cost is inside it, so
    orchestrating around it is not available. The two branches, the `.h5` and
    the `.h5ad`, are the branches `get_spaceranger_counts` takes, in its order.
    """
    if not sparse_counts:
        return get_spaceranger_counts(spaceranger_dir)

    import scanpy as sc

    stem = f"{spaceranger_dir}/{config.visium.filtered_feature_name}"

    if Path(f"{stem}.h5").exists():
        adatatmp = sc.read_10x_h5(f"{stem}.h5")
    elif Path(f"{stem}.h5ad").exists():
        adatatmp = sc.read_h5ad(f"{stem}.h5ad")
    else:
        msg = f"{spaceranger_dir} has no {config.visium.filtered_feature_name}.h5(ad)"
        raise RuntimeError(msg)

    counts = adatatmp.X

    if sp.issparse(counts):
        values = _without_nan(counts.data)
        counts = counts.__class__(
            (values.astype(np.int64), counts.indices, counts.indptr),
            shape=counts.shape,
        )
        counts.eliminate_zeros()
    else:
        counts = _without_nan(counts).astype(np.int64)

    adatatmp.layers["count"] = counts
    adatatmp.var_names_make_unique()

    return adatatmp


def _gene_umis(counts: Any) -> np.ndarray:
    """Per-gene totals, whether the counts are dense or sparse."""
    return np.asarray(np.sum(counts, axis=0), dtype=float).ravel()


def _scaled_columns(counts: Any, factors: np.ndarray) -> Any:
    """Each column scaled by its factor, in place, keeping the integer dtype.

    The dense form assigns a float product back into an `int64` array, which
    truncates toward zero. The sparse form assigns into `.data`, which is the
    same array dtype and so truncates the same way -- the equality is the
    reason the two can share one caller.

    **In place, on both paths**, and the result is returned rather than
    discarded only so the caller reads as an assignment. Copying would
    reinstate the allocation this exists to remove, and `cnamaste`'s own form
    -- `layers["count"][:, gene] = ...` -- mutates in place as well.
    """
    if sp.issparse(counts):
        # NB CSR, and not CSC, because it is CSR whose `indices` are column
        #    indices -- CSC's are row indices, and factors read through them
        #    scale the wrong entries where the matrix is not square, or index
        #    out of bounds where it is wider than it is tall.
        scaled = counts.tocsr()
        scaled.data = (scaled.data * factors[scaled.indices]).astype(counts.dtype)
        scaled.eliminate_zeros()

        return scaled.asformat(counts.format)

    counts[:, :] = (counts * factors).astype(counts.dtype)

    return counts


def _range_mask(unique_snp_ids: np.ndarray, ranges: pd.DataFrame) -> np.ndarray:
    """Which SNPs fall outside every filtered range.

    `cnamaste` walks the SNPs in Python with a fast-forward pointer into the
    ranges, calling `ranges.Chr.to_numpy()` **inside** the inner loop, so it
    rebuilds the column once per comparison. This sorts the ranges once and
    finds each SNP's candidate by `searchsorted`.

    The two agree because the ranges are disjoint per chromosome, which is what
    the original's single forward pointer already assumes.
    """
    chromosome = np.array(
        [int(str(snp).split("_")[0]) for snp in unique_snp_ids], dtype=np.int64
    )
    position = np.array(
        [int(str(snp).split("_")[1]) for snp in unique_snp_ids], dtype=np.int64
    )

    keys = np.stack(
        [ranges.Chr.to_numpy().astype(np.int64), ranges.End.to_numpy().astype(np.int64)]
    ).T
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    chr_sorted = keys[order, 0]
    end_sorted = keys[order, 1]
    start_sorted = ranges.Start.to_numpy().astype(np.int64)[order]

    # NB the first range on this chromosome whose end is past the SNP, which is
    #    the one the forward pointer stops at.
    candidate = np.searchsorted(
        chr_sorted * (1 + end_sorted.max()) + end_sorted,
        chromosome * (1 + end_sorted.max()) + position,
        side="right",
    )
    candidate = np.clip(candidate, 0, len(order) - 1)

    inside = (
        (chr_sorted[candidate] == chromosome)
        & (start_sorted[candidate] <= position)
        & (end_sorted[candidate] > position)
    )

    return np.asarray(~inside, dtype=bool)


def load_input_data(
    config: Any,
    alignment_files: Any = None,
    filter_gene_file: Any = None,
    filter_range_file: Any = None,
    normal_idx_file: Any = None,
    min_snp_umis: int = 50,
    min_percent_expressed_spots: float = 5.0e-3,
    *,
    sparse_counts: bool = False,
) -> ProcessedData:
    """What `cnamaste.io.load_input_data` returns, computed in fewer passes.

    Parameters
    ----------
    sparse_counts : bool
        Return the two allele matrices, the count layer and `exp_counts`
        sparse rather than dense. `False`, the default, is what `cnamaste`
        returns and what its callers index; `True` is the measurement of what
        the dense forms cost, and changes the type every downstream consumer
        sees. Under `True`, `exp_counts` **is** `adata.layers["count"]` rather
        than a copy of it, so a consumer that mutates one mutates the other --
        `cnamaste`'s frame is a separate object.

    Raises
    ------
    NotImplementedError
        If `alignment_files` is given, as upstream.
    """
    if alignment_files is not None:
        msg = "Alignment files are not supported."
        raise NotImplementedError(msg)

    df_meta = get_sample_sheet(config.paths.sample_sheet)

    assert np.all(df_meta["snp_dir"] == df_meta["snp_dir"].iloc[0])

    snp_dir = df_meta["snp_dir"].iloc[0]
    known_sample_id = df_meta.sample_id[0] if len(df_meta) == 1 else None

    df_agg_barcode = get_aggregated_barcodes(f"{snp_dir}/barcodes.txt", known_sample_id)
    snp_barcodes = get_barcodes(f"{snp_dir}/barcodes.txt").rename(
        columns={"combined_barcode": "barcodes"}, errors="raise"
    )
    snp_barcodes["barcodes"] = snp_barcodes["barcodes"].map(
        lambda xx: xx.replace("_U1", "")
    )

    unique_snp_ids = map_unique_snps_enum(
        np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    )

    cell_snp_Aallele, cell_snp_Ballele = _load_allele_matrices(snp_dir)

    assert cell_snp_Aallele.shape == cell_snp_Ballele.shape

    # NB upstream writes `(A + B).todense().sum(axis=1)`, which allocates a
    #    dense (spots, snps) matrix to reduce it away on the next call. The
    #    sum is the same; only the intermediate is not built.
    snp_umis_per_spot = np.asarray(
        (cell_snp_Aallele + cell_snp_Ballele).sum(axis=1)
    ).ravel()

    logger.info(
        f"Read cell-snp A,B matrices of shape={cell_snp_Aallele.shape} with "
        f"min={snp_umis_per_spot.min()}, max={snp_umis_per_spot.max()}, "
        f"median={np.median(snp_umis_per_spot)} snp-umis per cell.  "
        f"Found {int(snp_umis_per_spot.sum()):_} snp-umis total."
    )

    adata = None

    for i, sname in enumerate(df_meta.sample_id.to_numpy()):
        index = np.where(df_agg_barcode["sample_id"] == sname)[0]

        df_this_barcode = copy.copy(df_agg_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode

        df_this_pos = get_spatial_positions(df_meta["spaceranger_dir"].iloc[i])
        df_this_pos = get_he_image(df_meta["spaceranger_dir"].iloc[i], pos=df_this_pos)

        adatatmp = _spaceranger_counts(
            df_meta["spaceranger_dir"].iloc[i], config, sparse_counts=sparse_counts
        )

        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()

        if not np.array_equal(idx_argsort, np.arange(len(idx_argsort))):
            adatatmp = adatatmp[idx_argsort, :].copy()

        shared_barcodes = set(df_this_pos.barcode) & set(adatatmp.obs.index)
        isin = adatatmp.obs.index.isin(shared_barcodes)

        logger.info(
            f"Retaining {100.0 * np.mean(isin):.3f}% of spots based on "
            "(in-tissue) position and UMIs."
        )

        if not isin.all():
            adatatmp = adatatmp[isin, :].copy()

        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]
        df_this_pos.barcode = pd.Categorical(
            df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True
        )
        df_this_pos = df_this_pos.sort_values(by="barcode")

        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T

        if "gray" in df_this_pos.columns:
            adatatmp.obsm["he_gray"] = df_this_pos.gray.to_numpy()
        if "label" in df_this_pos.columns:
            adatatmp.obsm["he_label"] = df_this_pos.label.to_numpy()

        adatatmp.obs["sample"] = sname
        adatatmp.obs.index = [f"{x}" for x in adatatmp.obs.index]

        adata = (
            adatatmp
            if adata is None
            else anndata.concat([adata, adatatmp], join="outer")
        )

    assert adata is not None

    shared_barcodes = set(snp_barcodes.barcodes) & set(adata.obs.index)
    isin = snp_barcodes.barcodes.isin(shared_barcodes).to_numpy()

    assert np.any(isin), (
        "Found inconsistent barcodes between SNPs and UMIs, e.g. \n"
        f"{list(snp_barcodes.barcodes)[:5]}\nvs\n{list(adata.obs.index)[:5]}"
    )

    if not isin.all():
        cell_snp_Aallele = cell_snp_Aallele[isin, :]
        cell_snp_Ballele = cell_snp_Ballele[isin, :]
        snp_barcodes = snp_barcodes[isin]

    isin = adata.obs.index.isin(shared_barcodes)

    if not isin.all():
        adata = adata[isin, :].copy()

    idx_argsort = pd.Categorical(
        adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
    ).argsort()

    if not np.array_equal(idx_argsort, np.arange(len(idx_argsort))):
        adata = adata[idx_argsort, :]

    across_slice_adjacency_mat = get_alignments(
        alignment_files, df_meta, df_agg_barcode
    )

    # NB one pass over the counts, where upstream takes four: the UMI filter,
    #    the total, the percentiles and the post-filter median each recompute
    #    it. The filter itself is unchanged -- transcript UMIs and SNP UMIs
    #    both at or above the floor.
    spot_umis = _spot_umis(adata.layers["count"])
    allele_umis = (
        np.asarray(cell_snp_Aallele.sum(axis=1)).ravel()
        + np.asarray(cell_snp_Ballele.sum(axis=1)).ravel()
    )

    indicator = (spot_umis >= min_snp_umis) & (allele_umis >= min_snp_umis)

    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient "
        f"umis and snp-umis (>= {min_snp_umis})."
    )

    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    if across_slice_adjacency_mat is not None:
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    spot_umis = spot_umis[indicator]

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    perc_vals = np.percentile(spot_umis, percentiles)
    pairs = "\n".join(
        f"{p:.3f} [%]\t{v:_.0f}" for p, v in zip(perc_vals, percentiles, strict=False)
    )

    logger.info(f"Found total umi = {int(spot_umis.sum()):_} for input.")
    logger.info(f"Per-spot umi percentiles:\n{pairs}")

    expressed_in = _genes_expressed_in(adata.X)
    indicator = expressed_in >= min_percent_expressed_spots * adata.shape[0]

    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of genes with sufficient "
        f"expression across spots @ {min_percent_expressed_spots} fraction of spots."
    )

    adata = adata[:, indicator]

    if filter_gene_file is not None:
        genes_to_filter = get_filter_genes(filter_gene_file).iloc[:, 0].to_numpy()
        adata = adata[:, ~np.isin(adata.var.index, genes_to_filter)]

    if filter_range_file is not None:
        keep = _range_mask(unique_snp_ids, get_filter_ranges(filter_range_file))

        logger.info(
            f"Retaining {100.0 * np.mean(keep):.2f}% of snps based on input "
            "filter ranges."
        )

        cell_snp_Aallele = cell_snp_Aallele[:, keep]
        cell_snp_Ballele = cell_snp_Ballele[:, keep]
        unique_snp_ids = unique_snp_ids[keep]

    if config.quality.local_outlier_filter:
        gene_umi_counts = _gene_umis(adata.layers["count"])

        clf = LocalOutlierFactor(n_neighbors=200)
        label = clf.fit_predict(gene_umi_counts.reshape(-1, 1))
        to_zero = np.where(label == -1)[0]

        total_umis = gene_umi_counts.sum()
        ratio = gene_umi_counts[to_zero].sum() / total_umis

        logger.info(
            f"Removed {len(to_zero)} outlier genes ({100.0 * ratio:.3f}% of umis) "
            f"based on {clf.__class__.__name__}."
        )

        for rank in np.argsort(-gene_umi_counts[to_zero])[:25]:
            gene_idx = to_zero[rank]
            warning = (
                "WARNING known to be cancerous"
                if exp_cancer_gene(adata.var.index[gene_idx])
                else ""
            )
            logger.info(
                f"  {adata.var.index[gene_idx]:<20} "
                f"{100.0 * gene_umi_counts[gene_idx] / total_umis:6.3f}% UMIs {warning}"
            )

        keep = np.ones(adata.shape[1], dtype=float)
        keep[to_zero] = 0.0

        adata.layers["count"] = _scaled_columns(adata.layers["count"], keep)

    elif config.quality.normalize_gene_outliers:
        percentile = 95
        gene_counts = _gene_umis(adata.layers["count"])

        total_umis = gene_counts.sum()
        top = np.where(gene_counts >= np.percentile(gene_counts, percentile))[0]
        top_umis = gene_counts[top].sum()
        target_umis = (1.0 - percentile / 100) * (total_umis - top_umis)

        over = top[gene_counts[top] > target_umis]

        factors = np.ones(adata.shape[1], dtype=float)
        factors[over] = target_umis / gene_counts[over]

        adata.layers["count"] = _scaled_columns(adata.layers["count"], factors)

        logger.info(
            f"Downsampled top {100.0 - percentile}% genes; originally "
            f"{100.0 * top_umis / total_umis:.3f} [%]."
        )

    if normal_idx_file is not None:
        normal_barcodes = (
            pd.read_csv(normal_idx_file, header=None).iloc[:, 0].to_numpy()
        )

        # NB `.loc` rather than upstream's
        #    `adata.obs["tumor_annotation"][mask] = "normal"`, which is chained
        #    assignment: it writes through an intermediate that pandas 3.0's
        #    copy-on-write makes a copy, so the annotation would silently stop
        #    being applied. Same values today, and reported upstream.
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs.loc[adata.obs.index.isin(normal_barcodes), "tumor_annotation"] = (
            "normal"
        )

    assert adata.layers["count"].shape[0] == cell_snp_Aallele.shape[0]
    assert cell_snp_Aallele.shape[0] == cell_snp_Ballele.shape[0]
    assert len(unique_snp_ids) == cell_snp_Aallele.shape[1]

    # NB the frame costs 679 ms at 2,500 spots and is built eagerly, and its
    #    one live consumer -- `filter_normal_diffexp` -- opens with
    #    `anndata.AnnData(exp_counts)` and `exp_counts.values`, which makes it
    #    dense again. Under `sparse_counts` the matrix is handed back instead.
    if sparse_counts:
        # NB the layer's own format, unconverted. `cnamaste` builds the frame
        #    from CSC because `from_spmatrix` wants a column store; the one
        #    live consumer takes `anndata.AnnData(exp_counts)`, which reads
        #    either, so the conversion is 86 ms bought for the container.
        exp_counts = adata.layers["count"]
    else:
        exp_counts = pd.DataFrame.sparse.from_spmatrix(
            sp.csc_matrix(adata.layers["count"]),
            index=adata.obs.index,
            columns=adata.var.index,
        )

    return ProcessedData(
        adata.obsm["X_pos"],
        adata.obs.index,
        adata,
        exp_counts,
        cell_snp_Aallele if sparse_counts else cell_snp_Aallele.toarray(),
        cell_snp_Ballele if sparse_counts else cell_snp_Ballele.toarray(),
        unique_snp_ids,
        across_slice_adjacency_mat,
    )


def get_sample_list(adata):
    sample_list = [adata.obs["sample"].iloc[0]]

    # NB loop through rows (barcodes x samples) and collect sample names;
    #    assumes sorted by sample and is unique in this case.
    for i in range(1, adata.shape[0]):
        if adata.obs["sample"].iloc[i] != sample_list[-1]:
            logger.warning(
                f"Appending sample_id={adata.obs['sample'].iloc[i]} to sample list."
            )
            sample_list.append(adata.obs["sample"].iloc[i])

    # NB e.g. HT112C1-U1.
    logger.info(f"Found {len(sample_list)} unique samples:\n{sample_list}")

    # NB array: assigns to each transcript row (barcode x sample) unique index according to sample names.
    sample_ids = -np.ones(adata.shape[0], dtype=int)

    for s, sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        sample_ids[index] = s

    assert np.all(
        sample_ids >= 0
    ), f"Failed to assign unique integer to all samples in list. Bug?"

    return sample_list, sample_ids


def read_tumor_prop(adata, config=None):
    if config is None:
        config = get_global_config()

    if config.preprocessing.tumorprop_file is not None:
        logger.info(
            f"Reading pre-processed tumorprop file={config.preprocessing.tumorprop_file}"
        )

        df_tumorprop = pd.read_csv(
            config.preprocessing.tumorprop_file, sep="\t", header=0, index_col=0
        )

        df_tumorprop = df_tumorprop[["Tumor"]]
        df_tumorprop.columns = ["tumor_proportion"]

        assert np.all(
            adata.obs.index == df_tumorprop.index
        ), "Detected mis-alignment of AnnData & tumor prop. barcode/sample ordering."

        adata.obs = adata.obs.join(df_tumorprop)

        return adata.obs["tumor_proportion"]
    else:
        logger.info(f"No (pre-processed) tumorprop. file provided.")

        # np.ones(len(adata.obs.index), dtype=float)
        return
