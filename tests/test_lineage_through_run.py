"""Segment lineage through one `run_cnaster_port` run, checked against the run itself (#438)."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from port.extensions.segments import recording
from port.patch import recomb
from port.scripts.run_cnaster import main
from port.sim.run_config import isolated_run, write_for_run

from tests.fixtures import end_to_end_truth


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The run, with its lineage and every kernel it computed."""

    mpl.use("Agg")
    root = tmp_path_factory.mktemp("lineage")
    truth = end_to_end_truth()
    written, config = write_for_run(truth, root, max_iter_outer=1, max_iter=3)

    kernels: list[tuple[str, int, np.ndarray]] = []
    original = recomb.get_sitewise_transmat

    def spy(*args: Any, **kwargs: Any) -> np.ndarray:
        # NB `run_cnaster` passes it positionally once and by name three times.
        key = kwargs.get("segment_key", args[0] if args else None)
        table: Any = kwargs.get("df_gene_snp", args[1] if len(args) > 1 else None)
        result = original(*args, **kwargs)
        kernels.append((str(key), int(table[key].nunique()), result))
        return result

    recomb.get_sitewise_transmat = spy

    try:
        with isolated_run(), warnings.catch_warnings(), recording() as lineage:
            warnings.simplefilter("ignore")
            assert main([str(config)]) == 0
    finally:
        recomb.get_sitewise_transmat = original

    written_table = pd.read_csv(
        next((written.root / "output").rglob("gene_segments.tsv")), sep="\t"
    )
    return {
        "lineage": lineage,
        "kernels": kernels,
        "table": written_table,
    }


@pytest.mark.analytic
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
def test_every_kernel_is_independence_at_every_contig_boundary(
    run: dict[str, Any],
) -> None:
    """Each kernel has one entry per segment, `log 1/2` at each contig end."""
    lineage = run["lineage"]
    kernels = run["kernels"]

    assert len(kernels) == 4, f"{len(kernels)} kernels"

    levels = list(lineage.levels.values())

    for key, n_segments, kernel in kernels:
        level = next(
            level for level in reversed(levels) if level.n_segments == n_segments
        )
        assert kernel.size == level.n_segments, key
        np.testing.assert_array_equal(kernel[level.boundary], np.log(0.5))
        assert np.unique(level.contig).size > 1, "the instance must cross a contig"


@pytest.mark.analytic
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
def test_the_levels_nest_as_the_pipeline_builds_them(run: dict[str, Any]) -> None:
    """Blocks refine bins, filtering subsets, the merge coarsens; `lengths` sum to segment counts."""
    levels = run["lineage"].levels
    names = list(levels)

    assert names[:3] == ["blocks", "bins", "bins-filtered"], names
    blocks, bins, filtered = levels["blocks"], levels["bins"], levels["bins-filtered"]
    merged = levels[names[-1]]

    assert blocks.refines(bins)
    assert filtered.refines(bins)
    assert filtered.refines(merged)
    assert np.all(merged.label[filtered.label == -1] == -1)

    for level in levels.values():
        assert level.lengths.sum() == level.n_segments
        assert np.all(level.lengths > 0)


@pytest.mark.analytic
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
def test_the_run_writes_its_lineage(run: dict[str, Any]) -> None:
    """`gene_segments.tsv` is the recorded lineage, one label column per level."""
    lineage = run["lineage"]
    table = run["table"]
    expected = lineage.table()

    assert list(table.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
