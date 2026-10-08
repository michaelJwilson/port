"""`port.patch.omics.blocks`'s count summarizers against `cnaster`'s, bitwise (#198).

Also against an explicit loop on a matrix built to trip a grouped sum, and the
planted per-gene counts summed per block.
"""

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from port.sim.inputs import read_to_bins
from port.sim.run_config import PlantedInstance

pytestmark = pytest.mark.preprocessing

PHASES = ["all-true", "alternating", "all-false"]
"""Phase vectors for the bin summary; the fixture yields only the first."""


@pytest.fixture(scope="module")
def blocked(
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> tuple[Any, Any, Any]:
    """Return the loaded instance, its gene-SNP table with blocks, and the block counts."""
    chain = read_to_bins(planted_instance[2], through="blocks")

    return chain.loaded, chain.table, chain.blocks


def _fields_equal(realized: Any, reference: Any) -> None:
    """Assert every field of a `SpatioGenomicCounts` equal, bitwise (`__iter__` yields values, #196)."""
    for key in reference.keys():  # noqa: SIM118 -- not a dict; see above
        np.testing.assert_array_equal(
            np.asarray(realized[key]), np.asarray(reference[key]), err_msg=key
        )


@pytest.mark.patch
def test_the_block_counts_are_cnasters(blocked: tuple[Any, Any, Any]) -> None:
    """Every field of the block summary equals `cnaster`'s, bitwise."""
    from cnaster.omics import summarize_counts_for_blocks as upstream
    from port.patch.omics.blocks import summarize_counts_for_blocks as patched

    loaded, table, _ = blocked
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)
    arguments = (loaded.adata, *alleles, loaded.unique_snp_ids)

    _fields_equal(patched(table.copy(), *arguments), upstream(table.copy(), *arguments))


@pytest.mark.patch
@pytest.mark.parametrize("phase", PHASES)
def test_the_bin_counts_are_cnasters(blocked: tuple[Any, Any, Any], phase: str) -> None:
    """Every field of the bin summary equals `cnaster`'s, bitwise, under three phases."""
    from cnaster.omics import create_bin_ranges
    from cnaster.omics import summarize_counts_for_bins as upstream
    from port.patch.omics.blocks import summarize_counts_for_bins as patched

    loaded, table, counts = blocked
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

    binned = create_bin_ranges(
        table.copy(),
        loaded.adata,
        *alleles,
        loaded.unique_snp_ids,
        counts.X,
        counts.total_bb_RD,
        counts.lengths,
        secondary_min_umi=1,
        secondary_min_snp_umi=1,
        secondary_min_normal_umi=0,
    )

    n_blocks = counts.X.shape[0]
    indicator = {
        "all-true": np.ones(n_blocks, dtype=bool),
        "alternating": np.arange(n_blocks) % 2 == 0,
        "all-false": np.zeros(n_blocks, dtype=bool),
    }[phase]

    def call(implementation: Any) -> Any:
        return implementation(
            binned.copy(),
            loaded.adata,
            counts.X,
            counts.total_bb_RD,
            indicator,
            nu=1.0,
            logphase_shift=0.0,
            geneticmap_file=None,
        )

    _fields_equal(call(patched), call(upstream))


@pytest.mark.analytic
def test_the_indicator_sums_each_group_and_nothing_else() -> None:
    """The grouped sum equals an explicit loop with a duplicate, an orphan and an empty group."""
    from port.patch.omics.blocks import _group_indicator, _grouped_column_sums

    matrix = np.arange(12, dtype=np.int64).reshape(3, 4)
    rows = np.array([0, 0, 1, 2])
    groups = np.array([0, 0, 0, 1])

    indicator = _group_indicator(rows, groups, 4, 3)
    realized = _grouped_column_sums(matrix, indicator)

    expected = np.stack(
        [
            matrix[:, [0, 1]].sum(axis=1),
            matrix[:, [2]].sum(axis=1),
            np.zeros(3, dtype=np.int64),
        ]
    )

    assert realized.shape == (3, 3)
    np.testing.assert_array_equal(realized, expected)


@pytest.mark.analytic
def test_the_grouped_sum_agrees_sparse_and_dense() -> None:
    """The grouped sum is the same on sparse and dense counts (#186)."""
    from port.patch.omics.blocks import _group_indicator, _grouped_column_sums

    dense = np.arange(40, dtype=np.int64).reshape(5, 8)
    dense[dense % 3 == 0] = 0

    indicator = _group_indicator(np.arange(8), np.array([0, 0, 1, 1, 2, 2, 2, 2]), 8, 3)

    np.testing.assert_array_equal(
        _grouped_column_sums(sp.csr_matrix(dense), indicator),
        _grouped_column_sums(dense, indicator),
    )


@pytest.mark.end2end
@pytest.mark.parametrize("implementation", ["cnaster", "port"])
def test_the_block_counts_are_the_planted_counts(
    planted_instance: PlantedInstance,
    blocked: tuple[Any, Any, Any],
    implementation: str,
) -> None:
    """Both implementations' block counts equal the planted counts summed per block."""
    from port.patch.omics import summarize_counts_for_blocks

    _, pre_image, written, _ = planted_instance
    loaded, table, counts = blocked

    if implementation == "port":
        counts = summarize_counts_for_blocks(
            table,
            loaded.adata,
            loaded.cell_snp_Aallele,
            loaded.cell_snp_Ballele,
            loaded.unique_snp_ids,
        )

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)
    column_of = {str(name): index for index, name in enumerate(names)}
    spot_of = {str(barcode): spot for spot, barcode in enumerate(written.barcodes)}

    rows = np.array([spot_of[str(barcode)] for barcode in loaded.barcodes])
    genes = table[table.is_interval.to_numpy().astype(bool)]

    expected = np.zeros_like(counts.X[:, 0, :])

    for block, frame in genes.groupby("block_id"):
        columns = sorted(
            {column_of[str(name)] for name in frame.gene if name in column_of}
        )

        if columns:
            expected[int(block)] = planted[np.ix_(rows, columns)].sum(axis=1)

    np.testing.assert_array_equal(counts.X[:, 0, :], expected)
