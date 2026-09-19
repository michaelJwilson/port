"""`port.patch.reference.get_reference_genes` against `cnaster`'s (#185).

**13.3x and 55.2 MB to 23.3 MB at 250,000 transcripts, bitwise.** `cnaster`
reads the reference table with `pandas`, builds a frame whose every column is
then taken out again with `.to_numpy()`, and converts the contig names with a
Python loop -- `[int(x[3:]) for x in ...]`, once per transcript.

Read with `polars` and the frame in the middle never exists. What has to be
pinned is that the six columns come back identical: the values, the order, and
the dtypes, since `snp_id` is an all-`None` object column and `CHR` an
integer, and a reader that returned strings or `NaN` would pass a comparison
of values alone.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.test_load_input_data_patch import (
    gate_config,  # noqa: F401  -- used by name, and it needs the one below
    planted_instance,  # noqa: F401  -- `gate_config` resolves it in this module
)

pytestmark = pytest.mark.preprocessing


@pytest.mark.patch
def test_the_reference_table_is_cnasters_table(
    planted_instance: tuple[Any, Any, Any, Any],  # noqa: F811
    gate_config: Any,  # noqa: F811
) -> None:
    """Every column, every dtype, and the index, as a frame."""
    from cnaster.reference import get_reference_genes as upstream
    from port.patch.reference import get_reference_genes as patched

    _, _, written, _ = planted_instance

    pd.testing.assert_frame_equal(
        patched(str(written.hgtable)), upstream(str(written.hgtable))
    )


@pytest.mark.patch
def test_the_reader_drops_what_cnaster_drops(
    tmp_path: Path,
    gate_config: Any,  # noqa: F811
) -> None:
    """**Only chr1 to chr22 survive, and a written table proves it.**

    The fixture's reference carries autosomes alone, so a reader that kept
    everything would agree with `cnaster` on it and disagree on a real one.
    `chrX`, `chrY` and `chrM` are written here for that reason, and the
    contig column is checked as integers rather than as strings -- the parse
    is the other thing the Python loop was doing.
    """
    from cnaster.reference import get_reference_genes as upstream
    from port.patch.reference import get_reference_genes as patched

    path = tmp_path / "hgtable.tsv"
    contigs = ["chr1", "chr7", "chr22", "chrX", "chrY", "chrM", "chr2"]

    pd.DataFrame(
        {
            "name": [f"tx_{index}" for index in range(len(contigs))],
            "name2": [f"gene_{index}" for index in range(len(contigs))],
            "chrom": contigs,
            "cdsStart": np.arange(len(contigs)) * 1_000,
            "cdsEnd": np.arange(len(contigs)) * 1_000 + 500,
        }
    ).to_csv(path, sep="\t", index=False)

    realized = patched(str(path))

    pd.testing.assert_frame_equal(realized, upstream(str(path)))
    np.testing.assert_array_equal(realized.CHR.to_numpy(), [1, 7, 22, 2])
    assert realized.snp_id.isna().all()
    assert realized.is_interval.all()
