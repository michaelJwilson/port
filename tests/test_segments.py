"""Segmentations as labellings of the gene rows (#438).

Referees: a brute-force walk over the table for what a labelling says
(`oracle`), and the definitions for what is derived (`analytic`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from port.extensions.segments import Segmentation


def _table(seed: int = 3) -> pd.DataFrame:
    """Genes on three contigs with SNPs inside some, blocks of 1-4 genes."""
    rng = np.random.default_rng(seed)
    rows = []

    for contig, n_genes in ((1, 40), (2, 25), (7, 31)):
        starts = np.cumsum(rng.integers(1_000, 5_000, n_genes))
        for start in starts:
            end = int(start + rng.integers(200, 900))
            rows.append(
                {"CHR": contig, "START": int(start), "END": end, "is_interval": True}
            )
            for _ in range(int(rng.integers(0, 3))):
                pos = int(rng.integers(start + 1, end))
                rows.append(
                    {"CHR": contig, "START": pos, "END": pos + 1, "is_interval": False}
                )

    table = (
        pd.DataFrame(rows)
        .sort_values(["CHR", "START"], kind="stable")
        .reset_index(drop=True)
    )

    genes = table.is_interval.to_numpy()
    contig = table.CHR.to_numpy()
    new_block = genes & (rng.random(len(table)) < 0.4)
    new_block |= genes & np.r_[True, contig[1:] != contig[:-1]]
    table["block_id"] = np.cumsum(new_block) - 1

    return table


def _levels() -> tuple[pd.DataFrame, Segmentation, Segmentation, Segmentation]:
    table = _table()
    blocks = Segmentation.from_table(table, "block_id")
    rng = np.random.default_rng(5)
    kept = blocks.select(rng.random(blocks.n_segments) > 0.2, name="kept")

    # NB bins: a new bin at each contig change, else at random.
    contig = kept.contig
    new_bin = np.r_[True, contig[1:] != contig[:-1]] | (
        rng.random(kept.n_segments) < 0.5
    )
    bins = kept.coarsen(np.cumsum(new_bin) - 1, name="bins")

    return table, blocks, kept, bins


@pytest.mark.oracle
def test_each_level_labels_the_genes_as_the_table_does() -> None:
    """A gene's label is its block's rank; a bin's extent is its first and last gene's."""
    table, blocks, kept, bins = _levels()
    genes = table[table.is_interval]

    np.testing.assert_array_equal(blocks.ids[blocks.label], genes.block_id.to_numpy())

    for segment in range(bins.n_segments):
        members = np.flatnonzero(bins.label == segment)
        span = bins.label[members[0] : members[-1] + 1]
        assert np.all((span == segment) | (span == -1))
        assert bins.start[segment] == genes.START.iloc[members[0]]
        assert bins.end[segment] == genes.END.iloc[members[-1]]

    assert kept.refines(bins)
    assert kept.refines(blocks)
    assert not bins.refines(kept)


@pytest.mark.oracle
def test_aggregate_is_a_groupby_and_broadcast_undoes_it() -> None:
    """Block sums equal pandas' `groupby`; broadcast gives each gene its block."""
    table, blocks, _, _ = _levels()
    genes = table[table.is_interval]
    values = np.random.default_rng(9).normal(size=(len(genes), 2))

    ours = blocks.aggregate(values)
    theirs = pd.DataFrame(values).groupby(genes.block_id.to_numpy()).sum().to_numpy()

    np.testing.assert_allclose(ours, theirs, rtol=1e-14)
    np.testing.assert_array_equal(
        blocks.broadcast(np.arange(blocks.n_segments)), blocks.label
    )


@pytest.mark.analytic
def test_lengths_are_the_contig_runs_and_never_zero() -> None:
    """`lengths` sums to the segment count, one entry per contig present, none zero."""
    _, blocks, kept, bins = _levels()

    for level in (blocks, kept, bins):
        lengths = level.lengths
        assert lengths.sum() == level.n_segments
        assert np.all(lengths > 0)
        assert lengths.size == np.unique(level.contig).size
        np.testing.assert_array_equal(
            np.flatnonzero(level.boundary), np.cumsum(lengths) - 1
        )


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("gap", "not contiguous"),
        ("two contigs", "two contigs"),
        ("order", "not contiguous"),
        ("no gene", "hold no gene"),
    ],
)
def test_a_labelling_that_would_misalign_rows_is_refused(
    defect: str, message: str
) -> None:
    """Refuses non-contiguous genes, cross-contig segments, out-of-order ids and gene-less segments."""

    table = _table()
    labels = table.block_id.to_numpy().copy()
    genes = np.flatnonzero(table.is_interval.to_numpy())
    boundary = int(np.flatnonzero(np.diff(table.CHR.to_numpy()))[0])

    if defect == "gap":
        labels[genes[5]] = labels[genes[20]]
    elif defect == "two contigs":
        labels[boundary + 1 :] -= 1
    elif defect == "order":
        labels = labels.max() - labels
    else:
        snp = int(np.flatnonzero(~table.is_interval.to_numpy())[0])
        labels = labels.astype(float)
        labels[snp] = labels.max() + 1

    table["bad"] = labels

    with pytest.raises(ValueError, match=message):
        Segmentation.from_table(table, "bad")


@pytest.mark.analytic
def test_the_known_range_paths_minus_one_is_refused_by_name() -> None:
    """`block_id = -1` (uncovered rows) is refused by that name, not as a gap."""

    table = _table()
    labels = table.block_id.to_numpy().copy()
    labels[:3] = -1
    table["known"] = labels

    with pytest.raises(ValueError, match="negative id"):
        Segmentation.from_table(table, "known")
