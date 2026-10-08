"""`port.patch.reference.get_reference_genes` against `cnaster`'s, bitwise (#185)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from port.sim.run_config import PlantedInstance

pytestmark = pytest.mark.preprocessing


@pytest.mark.patch
def test_the_reference_table_is_cnasters_table(
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> None:
    """Every column, dtype and the index equal `cnaster`'s frame."""
    from cnaster.reference import get_reference_genes as upstream
    from port.patch.reference import get_reference_genes as patched

    _, _, written, _ = planted_instance

    pd.testing.assert_frame_equal(
        patched(str(written.hgtable)), upstream(str(written.hgtable))
    )


@pytest.mark.patch
def test_the_reader_drops_what_cnaster_drops(
    tmp_path: Path,
    gate_config: Any,
) -> None:
    """Drops chrX, chrY and chrM as `cnaster` does, with contigs parsed as integers."""
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
