"""The read-depth segment floor (#551): a minimum length and normal UMI per segment.

Referees: the floor's own definition, checked segment by segment
(`analytic`), and a brute-force sum of the normal spots' counts over each
merged bin's genes (`oracle`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from port.extensions.segments import Segmentation


def _table(seed: int = 7) -> pd.DataFrame:
    """Genes on three contigs, one per bin; contig 9 holds a single short gene."""
    rng = np.random.default_rng(seed)
    rows = []

    for contig, n_genes in ((1, 60), (2, 35), (9, 1)):
        starts = np.cumsum(rng.integers(20_000, 400_000, n_genes))
        for k, start in enumerate(starts):
            rows.append(
                {
                    "CHR": contig,
                    "START": int(start),
                    "END": int(start + rng.integers(5_000, 60_000)),
                    "is_interval": True,
                    "gene": f"g{contig}_{k}",
                }
            )

    table = pd.DataFrame(rows)
    table["bin_id"] = np.arange(len(table), dtype=np.float64)
    return table


def _floored(
    min_length: float, min_weight: float, seed: int = 7
) -> tuple[Segmentation, np.ndarray, Segmentation]:
    from port.extensions.segments import Segmentation

    bins = Segmentation.from_table(_table(seed), "bin_id")
    weight = np.random.default_rng(seed).integers(0, 60, bins.genes.n_genes)
    return bins, weight, bins.floored(min_length, weight, min_weight, name="floored")


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("min_length", "min_weight"), [(7.5e5, 100.0), (2e6, 0.0), (0.0, 300.0)]
)
def test_every_floored_segment_meets_both_minimums_but_a_contigs_only_one(
    min_length: float, min_weight: float
) -> None:
    """Each merged segment spans `min_length` bp and holds `min_weight`, save a contig's only segment; the bins refine it and no segment crosses a contig."""
    bins, weight, floored = _floored(min_length, min_weight)
    held = floored.aggregate(weight)
    alone = np.repeat(floored.lengths == 1, floored.lengths)

    assert np.all(alone | (floored.end - floored.start >= min_length))
    assert np.all(alone | (held >= min_weight))
    assert not floored.short(min_length, weight, min_weight).any()
    assert bins.refines(floored)
    assert floored.lengths.size == bins.lengths.size
    assert floored.n_segments < bins.n_segments


@pytest.mark.analytic
def test_a_floor_of_zero_is_the_segmentation_itself() -> None:
    """With both minimums at zero nothing merges."""
    bins, weight, floored = _floored(0.0, 0.0)

    np.testing.assert_array_equal(floored.label, bins.label)


@pytest.mark.analytic
def test_the_lineage_refuses_a_level_under_its_floor_once_set() -> None:
    """Before the floor any level records; after it, the unfloored bins are refused and the floored ones kept."""
    from port.extensions.segments import Lineage

    bins, weight, floored = _floored(7.5e5, 100.0)
    lineage = Lineage(genes=bins.genes)
    lineage.record(bins, "bins")
    lineage.floor = (7.5e5, weight, 100.0)

    with pytest.raises(ValueError, match="under the floor"):
        lineage.record(bins, "bins")
    assert lineage.record(floored, "bins-floored").n_segments == floored.n_segments


@pytest.mark.oracle
def test_floor_bins_counts_normal_umi_as_a_sum_over_the_normal_spots() -> None:
    """`floor_bins`' merged bins each hold 150 normal UMIs by a brute-force sum over normal spots and genes, and every row keeps a bin."""
    import anndata
    import scipy.sparse as sp
    from port.patch.omics.blocks import floor_bins

    table = _table()
    rng = np.random.default_rng(11)
    counts = rng.poisson(0.4, size=(30, len(table)))
    adata = anndata.AnnData(
        X=np.zeros_like(counts, dtype=np.float32),
        var=pd.DataFrame(index=table["gene"].to_numpy()),
    )
    adata.layers["count"] = sp.csr_matrix(counts)
    normal = rng.random(30) < 0.5

    floored = floor_bins(table, adata, normal, min_length=5e5, min_normal_umi=150.0)
    by_gene = counts[normal].sum(axis=0)
    per_bin = pd.Series(by_gene).groupby(floored["bin_id"].to_numpy()).sum()
    contig_of = floored.groupby("bin_id")["CHR"].first()
    alone = contig_of.map(contig_of.value_counts()) == 1

    assert floored["bin_id"].notna().all()
    assert np.all(alone | (per_bin >= 150))


@pytest.mark.infra
@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        ({}, (None, None)),
        ({"min_segment_mb": True}, (0.75, None)),
        ({"min_segment_normal_umi": True}, (None, 300.0)),
        ({"min_segment_mb": 0.5, "min_segment_normal_umi": 600}, (0.5, 600.0)),
        ({"min_segment_mb": False, "min_segment_normal_umi": "none"}, (None, None)),
    ],
)
def test_the_floor_is_off_unless_the_config_sets_it(
    quality: dict[str, object], expected: tuple[float | None, float | None]
) -> None:
    """Absent, `false` or `none` is off; `true` is 0.75 Mb or 300 normal UMIs; a number is itself."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from port.patch.omics.blocks import segment_floor

    previous = get_global_config()
    set_global_config(YAMLConfig({"quality": quality}))
    try:
        assert segment_floor() == expected
    finally:
        set_global_config(previous)


def _greedy_parent(
    bins: Segmentation, weight: np.ndarray, min_length: float, min_weight: float
) -> np.ndarray:
    """The loop `Ragged.floored` replaced (T- #632): each segment's merged parent."""
    contig, start, end = bins.contig, bins.start, bins.end
    held = bins.aggregate(np.asarray(weight, dtype=np.float64))
    parent = np.zeros(bins.n_segments, dtype=np.int64)
    label, first, opened, total, closed = -1, 0, 0, 0.0, True
    for k in range(bins.n_segments):
        if k == 0 or contig[k] != contig[k - 1]:
            first, closed = label + 1, True
        if closed:
            label, opened, total = label + 1, int(start[k]), 0.0
        parent[k] = label
        total += float(held[k])
        closed = end[k] - opened >= min_length and total >= min_weight
        at_boundary = k == bins.n_segments - 1 or contig[k + 1] != contig[k]
        if at_boundary and not closed and label > first:
            parent[parent == label] = label - 1
            label -= 1
    return parent


@pytest.mark.patch
@pytest.mark.parametrize("seed", [7, 11, 23, 31])
@pytest.mark.parametrize(
    ("min_length", "min_weight"),
    [(7.5e5, 100.0), (2e6, 0.0), (0.0, 300.0), (5e6, 900.0)],
)
def test_sals_floor_merges_as_the_loop_it_replaces(
    seed: int, min_length: float, min_weight: float
) -> None:
    """sal #1141's `Ragged.floored` against port's greedy loop: the same parent for every segment."""
    from port.extensions.segments import Segmentation

    bins = Segmentation.from_table(_table(seed), "bin_id")
    weight = np.random.default_rng(seed).integers(0, 60, bins.genes.n_genes)
    floored = bins.floored(min_length, weight, min_weight, name="floored")

    oracle = bins.coarsen(
        _greedy_parent(bins, weight, min_length, min_weight), name="oracle"
    )

    assert floored.n_segments == oracle.n_segments
    for field in ("contig", "start", "end"):
        assert np.array_equal(getattr(floored, field), getattr(oracle, field))
