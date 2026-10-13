"""The phase-switch kernel: cnaster's Python walk against the lineage's vectorized one
(#438).

Both read the map file each call, as `run_cnaster` does; 2,200 and 33,000 (stress)
blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cnaster.recomb import get_sitewise_transmat
from port.patch.recomb import get_sitewise_transmat as port_get_sitewise_transmat
from pytest_benchmark.fixture import BenchmarkFixture

from tests.builders import gene_snp_blocks
from tests.fixtures import recombination_map, tiers

GATE = {"per_contig": 100}
STRESS = {"per_contig": 1_500}
"""22 contigs: 2,200 and 33,000 blocks."""


def _table(per_contig: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    grid = np.arange(1, 59_000_000, 20_000)
    return gene_snp_blocks(
        (contig, int(start), 15_010)
        for contig in range(1, 23)
        for start in np.sort(rng.choice(grid, per_contig, replace=False))
    )


def _cnaster(*args: Any) -> Any:
    return get_sitewise_transmat(*args)


def _port(*args: Any) -> Any:
    return port_get_sitewise_transmat(*args)


@pytest.mark.benchmark
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_cnaster, _port], ids=["cnaster", "port"])
def test_kernel(
    benchmark: BenchmarkFixture,
    arm: Callable[..., Any],
    size: dict[str, int],
    tmp_path: Path,
) -> None:
    table = _table(**size)
    path = recombination_map(tmp_path / "map.tsv", range(1, 23))
    arm("block_id", table.copy(), path, 1.0, -2.0)
    benchmark(arm, "block_id", table.copy(), path, 1.0, -2.0)
