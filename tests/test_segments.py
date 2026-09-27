"""The segment lineage (#438): what each level is, in terms of the rows it came from.

Referees: a brute-force walk over the rows for every composed quantity
(`oracle`), and the definitions for what is derived (`analytic`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from port.extensions.segments import Segmentation


def _table(seed: int = 3) -> pd.DataFrame:
    """Rows on three contigs, sorted, with blocks of 1-4 rows and bins of 1-3 blocks."""
    rng = np.random.default_rng(seed)
    rows = []

    for contig, n_rows in ((1, 40), (2, 25), (7, 31)):
        starts = np.cumsum(rng.integers(100, 1_000, n_rows))
        rows += [
            {"CHR": contig, "START": int(s), "END": int(s + rng.integers(1, 90))}
            for s in starts
        ]

    table = pd.DataFrame(rows)
    change = np.r_[True, table.CHR.to_numpy()[1:] != table.CHR.to_numpy()[:-1]]
    block_start = change | (rng.random(len(table)) < 0.4)
    table["block_id"] = np.cumsum(block_start) - 1

    return table


def _levels() -> tuple[pd.DataFrame, Segmentation, Segmentation, Segmentation]:
    from port.extensions.segments import Segmentation

    table = _table()
    blocks = Segmentation.from_table(table, key="block_id")
    rng = np.random.default_rng(5)
    keep = rng.random(blocks.n_segments) > 0.2

    # NB bins of kept blocks: a new bin at each contig change or at random.
    kept = blocks.select(keep, name="kept")
    change = np.r_[True, kept.contig[1:] != kept.contig[:-1]]
    bins = kept.group(
        np.cumsum(change | (rng.random(kept.n_segments) < 0.5)), name="bins"
    )

    return table, blocks, kept, bins


@pytest.mark.oracle
def test_every_level_is_its_rows() -> None:
    """A segment's composed row span is exactly the rows a brute-force walk assigns it."""
    table, blocks, kept, bins = _levels()
    lo, hi = bins.root_span()

    block_rows = {b: np.flatnonzero(table.block_id.to_numpy() == b) for b in blocks.ids}
    kept_ids = kept.ids

    for segment in range(bins.n_segments):
        members = kept_ids[bins.lo[segment] : bins.hi[segment]]
        rows = np.concatenate([block_rows[b] for b in members])
        assert lo[segment] == rows.min()
        assert hi[segment] == rows.max() + 1
        assert bins.start[segment] == table.START[rows.min()]
        assert bins.end[segment] == table.END[rows.max()]


@pytest.mark.oracle
def test_aggregate_is_a_groupby_and_broadcast_undoes_it() -> None:
    """Sums over each block's rows equal pandas' `groupby`; broadcast returns each row its block's value."""
    table, blocks, _, _ = _levels()
    values = np.random.default_rng(9).normal(size=(len(table), 2))

    ours = blocks.aggregate(values)
    theirs = pd.DataFrame(values).groupby(table.block_id.to_numpy()).sum().to_numpy()

    np.testing.assert_allclose(ours, theirs, rtol=1e-14)

    back = blocks.broadcast(np.arange(blocks.n_segments))
    np.testing.assert_array_equal(back, table.block_id.to_numpy())


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
        ("order", "genomic order"),
    ],
)
def test_a_grouping_that_would_misalign_rows_is_refused(
    defect: str, message: str
) -> None:
    """Non-contiguous members, a segment over two contigs, or ids out of genomic order."""
    from port.extensions.segments import Segmentation

    table = _table()
    labels = table.block_id.to_numpy().copy()
    boundary = int(np.flatnonzero(np.diff(table.CHR.to_numpy()))[0])

    if defect == "gap":
        labels[5] = labels[20]
    elif defect == "two contigs":
        labels[boundary + 1] = labels[boundary]
    else:
        labels = labels.max() - labels

    table["bad"] = labels

    with pytest.raises(ValueError, match=message):
        Segmentation.from_table(table, key="bad")
