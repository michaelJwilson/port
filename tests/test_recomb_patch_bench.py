"""The phase-switch kernel: `cnaster`'s Python walk against the lineage's vectorized one (#438).

Both read the same map file each call, as `run_cnaster` does. Median:
2,200 blocks 29.4 ms against 10.3 ms; 33,000 blocks (stress) 272.8 ms
against 26.8 ms, 10.2x.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import recombination_map, tiers

GATE = {"per_contig": 100}
STRESS = {"per_contig": 1_500}
"""22 contigs: 2,200 and 33,000 blocks."""


def _table(per_contig: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    block = 0

    for contig in range(1, 23):
        grid = np.arange(1, 59_000_000, 20_000)
        for start in np.sort(rng.choice(grid, per_contig, replace=False)):
            rows += [
                (contig, int(start), int(start) + 10, True, block),
                (contig, int(start) + 5, int(start) + 6, False, block),
                (contig, int(start) + 15_000, int(start) + 15_010, True, block),
            ]
            block += 1

    return pd.DataFrame(
        rows, columns=["CHR", "START", "END", "is_interval", "block_id"]
    )


def _cnaster(*args: Any) -> Any:
    from cnaster.recomb import get_sitewise_transmat

    return get_sitewise_transmat(*args)


def _port(*args: Any) -> Any:
    from port.patch.recomb import get_sitewise_transmat

    return get_sitewise_transmat(*args)


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
