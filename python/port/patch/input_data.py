"""`cnaster.io.load_input_data`, with the passes it does not need removed.

**Proposed for `cnaster`, written here.** #167: the loader densifies a sparse
matrix to take a row sum, builds three full copies of the count matrix to cast
it, recomputes the same reductions four times, copies the whole `AnnData` once
per filter, and walks the range filter in Python. `port` cannot land the change
(`CLAUDE.md`, **Working against a repository you do not own**), so it is
written here as a patch with its referee beside it.

**What it is not.** Not a reimplementation: every file this reads is read by
`cnaster`'s own helper, imported rather than copied, so the patch is the
orchestration and nothing else. What it changes is which intermediate arrays
exist, never which spots, genes or SNPs survive.

The default return is **bitwise** what `cnaster` returns, field for field,
which `tests/test_load_input_data_patch.py` pins. `sparse_counts=True` returns
the two allele matrices and the count layer as CSR instead, which is a
different type contract and therefore a `cnaster`-side decision -- it is here
so the memory it saves can be measured against the density where it starts to
pay, not because `port` can adopt it.
"""

from __future__ import annotations

import copy
from collections import namedtuple
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cnaster.config import start_time
from cnaster.filter import get_filter_genes, get_filter_ranges
from cnaster.he import get_he_image
from cnaster.io import (
    get_aggregated_barcodes,
    get_alignments,
    get_barcodes,
    get_sample_sheet,
    get_spaceranger_counts,
    get_spatial_positions,
    map_unique_snps_enum,
)
from cnaster.logger import get_logger
from cnaster.reference import exp_cancer_gene
from sklearn.neighbors import LocalOutlierFactor

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
"""`cnaster`'s own return shape, declared here because it declares it inline.

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

    `cnaster` writes `np.sum(adata.X > 0, axis=0)`, which materializes a second
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


def _range_mask(unique_snp_ids: np.ndarray, ranges: pd.DataFrame) -> np.ndarray:
    """Which SNPs fall outside every filtered range.

    `cnaster` walks the SNPs in Python with a fast-forward pointer into the
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
    """What `cnaster.io.load_input_data` returns, computed in fewer passes.

    Parameters
    ----------
    sparse_counts : bool
        Return the two allele matrices as CSR rather than dense. `False`, the
        default, is what `cnaster` returns and what its callers index; `True`
        is the measurement of what the dense return costs, and changes the
        type every downstream consumer sees.

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

    cell_snp_Aallele = sp.load_npz(f"{snp_dir}/cell_snp_Aallele.npz").tocsr()
    cell_snp_Ballele = sp.load_npz(f"{snp_dir}/cell_snp_Ballele.npz").tocsr()

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

        adatatmp = get_spaceranger_counts(df_meta["spaceranger_dir"].iloc[i])

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
        gene_umi_counts = np.asarray(
            np.sum(adata.layers["count"], axis=0), dtype=float
        ).ravel()

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

        adata.layers["count"][:, to_zero] = 0

    elif config.quality.normalize_gene_outliers:
        percentile = 95
        gene_counts = np.asarray(
            np.sum(adata.layers["count"], axis=0), dtype=float
        ).ravel()

        total_umis = gene_counts.sum()
        top = np.where(gene_counts >= np.percentile(gene_counts, percentile))[0]
        top_umis = gene_counts[top].sum()
        target_umis = (1.0 - percentile / 100) * (total_umis - top_umis)

        for gene_idx in top:
            if gene_counts[gene_idx] > target_umis:
                adata.layers["count"][:, gene_idx] = adata.layers["count"][
                    :, gene_idx
                ] * (target_umis / gene_counts[gene_idx])

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
