"""Replaces `cnaster.io.load_input_data` with fewer passes over the counts (#167).

Reads through `cnaster`'s own helpers except `_spaceranger_counts` (#186).
Default return is bitwise `cnaster`'s; `sparse_counts=True` returns the allele
matrices, count layer and `exp_counts` sparse.
"""

from __future__ import annotations

import copy
from collections import namedtuple
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata._core.views import (
    ArrayView,
)  # NB no public name; the type `adata.layers` returns on a view
from cnaster.config import start_time
from cnaster.filter import get_filter_genes
from cnaster.io import (
    get_aggregated_barcodes as _UPSTREAM_AGGREGATED_BARCODES,
)
from cnaster.io import (
    get_alignments,
    get_barcodes,
    get_sample_sheet,
    get_spatial_positions,
    map_unique_snps_enum,
)
from cnaster.logger import get_logger
from cnaster.reference import exp_cancer_gene
from sklearn.neighbors import LocalOutlierFactor

from port.patch.he import he_image

logger = get_logger(__name__, start_time=start_time)

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
"""`cnaster`'s inline return shape, field for field."""


def _spot_umis(counts: Any) -> np.ndarray:
    """Per-spot totals as a flat array, whether the counts are dense or sparse."""
    if sp.issparse(counts):
        return np.asarray(counts.sum(axis=1)).ravel()

    return np.asarray(np.sum(counts, axis=1)).ravel()


def _genes_expressed_in(counts: Any) -> np.ndarray:
    """How many spots express each gene; `getnnz` unless a stored entry is zero."""
    if sp.issparse(counts):
        stored = counts.data
        if stored.size and not stored.all():
            return np.asarray((counts > 0).sum(axis=0)).ravel()

        return np.asarray(counts.getnnz(axis=0)).ravel()

    return np.asarray(np.sum(counts > 0, axis=0)).ravel()


def _load_allele_matrices(snp_dir: str) -> tuple[Any, Any]:
    """The A and B allele matrices, as CSR, inflated on two threads."""
    from concurrent.futures import ThreadPoolExecutor

    paths = [f"{snp_dir}/cell_snp_{allele}allele.npz" for allele in ("A", "B")]

    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        a_matrix, b_matrix = pool.map(lambda path: sp.load_npz(path).tocsr(), paths)

    return a_matrix, b_matrix


def _without_nan(values: np.ndarray) -> np.ndarray:
    """`values` with any `NaN` replaced by zero, copying only if there is one."""
    nan = np.isnan(values)

    if not nan.any():
        return values

    logger.info(f"Found {100.0 * np.mean(nan):.3f}% NaN counts in anndata.")

    return np.where(nan, 0.0, values)


def _spaceranger_counts(
    spaceranger_dir: str, config: Any, *, sparse_counts: bool
) -> Any:
    """The `spaceranger` counts with the `count` layer cast to integer.

    `sparse_counts` casts the stored values only; otherwise the layer is
    `get_spaceranger_counts`'s dense one, accepting a dense `X` (#88).
    """
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

    if not sparse_counts:
        # NB densify only a sparse matrix: upstream's `.toarray()` raises on a
        #    dense `.h5ad` (#88).
        dense = counts.toarray() if sp.issparse(counts) else np.array(counts)
        is_nan = np.isnan(dense)

        if np.any(is_nan):
            dense[is_nan] = 0

        adatatmp.layers["count"] = dense.astype(int)
        adatatmp.var_names_make_unique()

        return adatatmp

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
    """Each column scaled by its factor, truncated to the integer dtype.

    The caller must assign the result: a dense view returns a new array, as
    anndata copies a view on first write (#466, #496).
    """
    if sp.issparse(counts):
        # NB CSR, whose `indices` are column indices.
        scaled = counts.tocsr()
        scaled.data = (scaled.data * factors[scaled.indices]).astype(counts.dtype)
        scaled.eliminate_zeros()

        return scaled.asformat(counts.format)

    if isinstance(counts, ArrayView):
        # NB writing into a view's layer leaves the view unchanged (#496).
        return (np.asarray(counts) * factors).astype(counts.dtype)

    counts[:, :] = (counts * factors).astype(counts.dtype)

    return counts


def filter_ranges(filter_range_file: Any) -> pd.DataFrame:
    """`cnaster.filter.get_filter_ranges`, also reading bare-integer chromosomes (#176).

    Returns integer `Chr`, sorted by `Chr` and `Start`, as `cnaster` does.
    """
    ranges = pd.read_csv(
        filter_range_file, header=None, sep="\t", names=["Chr", "Start", "End"]
    )
    ranges["Chr"] = [
        int(str(x)[3:]) if str(x).startswith("chr") else int(x)
        for x in ranges.Chr.to_numpy()
    ]
    ordered: pd.DataFrame = ranges.sort_values(by=["Chr", "Start"])

    return ordered


def _range_mask(unique_snp_ids: np.ndarray, ranges: pd.DataFrame) -> np.ndarray:
    """Which SNPs survive `cnaster`'s forward-pointer walk over the filter ranges.

    For SNPs in `(chr, pos)` order this is `searchsorted` over the running
    maximum of `(Chr, End)`, exact for overlapping ranges; otherwise the walk.
    """
    chromosome = np.array(
        [int(str(snp).split("_")[0]) for snp in unique_snp_ids], dtype=np.int64
    )
    position = np.array(
        [int(str(snp).split("_")[1]) for snp in unique_snp_ids], dtype=np.int64
    )
    chrs = ranges.Chr.to_numpy().astype(np.int64)
    starts = ranges.Start.to_numpy().astype(np.int64)
    ends = ranges.End.to_numpy().astype(np.int64)

    if len(chrs) == 0:
        return np.ones(len(unique_snp_ids), dtype=bool)

    scale = 1 + int(max(ends.max(), position.max(initial=0)))
    snp_key = chromosome * scale + position
    range_key = chrs * scale + ends

    if np.all(snp_key[1:] >= snp_key[:-1]):
        pointer = np.searchsorted(
            np.maximum.accumulate(range_key), snp_key, side="right"
        )
    else:
        pointer = np.empty(len(snp_key), dtype=np.int64)
        j = 0

        for i, key in enumerate(snp_key):
            while j < len(range_key) and range_key[j] <= key:
                j += 1
            pointer[i] = j

    found = pointer < len(chrs)
    at = np.minimum(pointer, len(chrs) - 1)
    inside = (
        found
        & (chrs[at] == chromosome)
        & (starts[at] <= position)
        & (ends[at] > position)
    )

    return np.asarray(~inside, dtype=bool)


NORMAL_SPOTS: list[np.ndarray] = []
"""Per loaded spot, whether `normal_idx_file` names it, from the last load (#479)."""


def release() -> None:
    """Drop the run's normal spots; `port.pipeline.patched` calls this on exit (#617)."""
    NORMAL_SPOTS.clear()


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
    """`cnaster.io.load_input_data`'s return, computed in fewer passes.

    `sparse_counts` returns the allele matrices, count layer and `exp_counts`
    sparse; `exp_counts` is then `adata.layers["count"]` itself, not a copy.
    Raises `NotImplementedError` for `alignment_files`, as upstream.
    """
    if alignment_files is not None:
        msg = "Alignment files are not supported."
        raise NotImplementedError(msg)

    df_meta = get_sample_sheet(config.paths.sample_sheet)

    if not (np.all(df_meta["snp_dir"] == df_meta["snp_dir"].iloc[0])):  # invariant
        msg = 'expected np.all(df_meta["snp_dir"] == df_meta["snp_dir"].iloc[0])'
        raise AssertionError(msg)

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

    if cell_snp_Aallele.shape != cell_snp_Ballele.shape:  # invariant
        msg = "expected cell_snp_Aallele.shape == cell_snp_Ballele.shape"
        raise AssertionError(msg)

    # NB upstream densifies `(A + B)` to sum it.
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
        df_this_pos = he_image(df_meta["spaceranger_dir"].iloc[i], pos=df_this_pos)

        # NB read and cast sparse on every path; densified once at the end (#488).
        adatatmp = _spaceranger_counts(
            df_meta["spaceranger_dir"].iloc[i], config, sparse_counts=True
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

    if adata is None:  # invariant
        msg = "expected adata is not None"
        raise AssertionError(msg)

    shared_barcodes = set(snp_barcodes.barcodes) & set(adata.obs.index)
    isin = snp_barcodes.barcodes.isin(shared_barcodes).to_numpy()

    if not (np.any(isin)):  # invariant
        msg = "Found inconsistent barcodes between SNPs and UMIs, e.g. \n"
        f"{list(snp_barcodes.barcodes)[:5]}\nvs\n{list(adata.obs.index)[:5]}"
        raise AssertionError(msg)

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

    # NB one pass over the counts, where upstream takes four.
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
        keep = _range_mask(unique_snp_ids, filter_ranges(filter_range_file))

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

    # NB `run_cnaster` never passes `normal_idx_file`; read it from the config (#479).
    if normal_idx_file is None:
        normal_idx_file = getattr(
            getattr(config, "preprocessing", None), "normalidx_file", None
        )

    if normal_idx_file is not None:
        normal_barcodes = (
            pd.read_csv(normal_idx_file, header=None).iloc[:, 0].to_numpy()
        )

        # NB `.loc` rather than upstream's chained assignment, which pandas
        #    3.0 copy-on-write would silently drop.
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs.loc[adata.obs.index.isin(normal_barcodes), "tumor_annotation"] = (
            "normal"
        )

    NORMAL_SPOTS.clear()

    if normal_idx_file is not None:
        NORMAL_SPOTS.append(adata.obs["tumor_annotation"].to_numpy() == "normal")

    if adata.layers["count"].shape[0] != cell_snp_Aallele.shape[0]:  # invariant
        msg = 'expected adata.layers["count"].shape[0] == cell_snp_Aallele.shape[0]'
        raise AssertionError(msg)
    if cell_snp_Aallele.shape[0] != cell_snp_Ballele.shape[0]:  # invariant
        msg = "expected cell_snp_Aallele.shape[0] == cell_snp_Ballele.shape[0]"
        raise AssertionError(msg)
    if len(unique_snp_ids) != cell_snp_Aallele.shape[1]:  # invariant
        msg = "expected len(unique_snp_ids) == cell_snp_Aallele.shape[1]"
        raise AssertionError(msg)

    # NB the frame's one consumer densifies it again; `sparse_counts` skips it.
    stored = adata.layers["count"]

    if not sparse_counts and sp.issparse(stored):
        # NB `cnaster`'s dense `int64` layer, built once (#488).
        dense = adata.layers["count"].toarray()

        if adata.is_view:
            adata = adata.copy()

        adata.layers["count"] = dense

    if sparse_counts:
        # NB the layer's own format, without `cnaster`'s CSC conversion.
        exp_counts = adata.layers["count"]
    else:
        exp_counts = pd.DataFrame.sparse.from_spmatrix(
            sp.csc_matrix(stored),
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


def get_aggregated_barcodes(
    barcode_file: str, known_sample_id: str | None = None
) -> pd.DataFrame:
    """`cnaster.io.get_aggregated_barcodes`, keeping each slice's `sample_id` (#446).

    With a `known_sample_id` this is upstream's, bitwise; without one,
    `sample_id` is the suffix after the last `_`.
    """
    frame = _UPSTREAM_AGGREGATED_BARCODES(barcode_file, known_sample_id)

    combined = frame["combined_barcode"].astype(str)
    if known_sample_id is None and combined.str.contains("_").all():
        frame["sample_id"] = combined.str.rsplit("_", n=1).str[-1].to_numpy()

    return frame


def get_sample_list(adata: Any) -> tuple[list[str], np.ndarray]:
    """`cnaster.io.get_sample_list`, keyed by sample name (#418).

    Bitwise upstream on contiguous rows; interleaved rows (`A, B, A`) give
    `[A, B]` rather than `[A, B, A]`. Recorded while `port.extensions.samples` records.
    """
    from port.extensions.samples import observe, samples_of

    samples = observe(samples_of(adata), adata.obs.index)

    logger.info(f"Found {len(samples.names)} unique samples:\n{list(samples.names)}")

    return samples.pair()
