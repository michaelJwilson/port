"""`cnaster.io.load_input_data`, with the passes it does not need removed.

**Proposed for `cnaster`, written here.** #167: the loader densifies a sparse
matrix to take a row sum, builds three full copies of the count matrix to cast
it, recomputes the same reductions four times, copies the whole `AnnData` once
per filter, and walks the range filter in Python. `port` cannot land the change
(`CLAUDE.md`, **Working against a repository you do not own**), so it is
written here as a patch with its referee beside it.

**What it is not.** Not a reimplementation, with one exception: every file
this reads is read by `cnaster`'s own helper, imported rather than copied, so
the patch is the orchestration and nothing else. The exception is
`_spaceranger_counts`, which reads the counts file itself under
`sparse_counts` because the cost it removes is inside `get_spaceranger_counts`
(#186) and orchestrating around it is not available. What it changes is which
intermediate arrays exist, never which spots, genes or SNPs survive.

The default return is **bitwise** what `cnaster` returns, field for field,
which `tests/test_load_input_data_patch.py` pins.

`sparse_counts=True` is the other half, and it is a type contract rather than
an optimization of the same one: the two allele matrices, the count layer and
`exp_counts` all come back sparse. That removes the loader's three
materializations -- the `pandas` sparse frame, the dense `int` cast, the dense
allele return -- which are 62% of its time and the whole of the difference in
its peak. It is measured in `tests/test_loader_materialization_bench.py` and
pinned in `tests/test_loader_materialization.py`, and it stays a flag because
adopting it is a `cnaster`-side decision about what every consumer indexes.
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

MIRRORS: tuple[str, ...] = ("cnaster.io",)
"""`io.load_input_data`, installed by `SWAPS`.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

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
    the branch is kept rather than assumed away because `cnaster` logs the
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

    `cnaster` writes the count layer with `adatatmp.X.toarray()`, tests it with
    `np.isnan`, and casts it with `astype(int)` -- three dense `(spots, genes)`
    arrays to cast values that `spaceranger` stored sparsely. At 2,500 spots
    and 5,983 genes that is 617 ms, 408 of it in the cast alone.

    The values are the same either way: a structural zero is not `NaN` and
    truncates to zero, so casting `.data` and casting the dense array agree
    entry for entry. What differs is that one of them allocates the shape and
    the other the stored values.

    This is the one place the patch reads a file `cnaster`'s helper would have
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
    reinstate the allocation this exists to remove, and `cnaster`'s own form
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
        Return the two allele matrices, the count layer and `exp_counts`
        sparse rather than dense. `False`, the default, is what `cnaster`
        returns and what its callers index; `True` is the measurement of what
        the dense forms cost, and changes the type every downstream consumer
        sees. Under `True`, `exp_counts` **is** `adata.layers["count"]` rather
        than a copy of it, so a consumer that mutates one mutates the other --
        `cnaster`'s frame is a separate object.

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
        # NB the layer's own format, unconverted. `cnaster` builds the frame
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
