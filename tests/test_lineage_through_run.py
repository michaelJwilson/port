"""The segment lineage through a whole `run_cnaster_port` run (#438).

One run of the entry point on the end-to-end instance, recording every
segmentation the patched stages make and every phase-switch kernel the run
asks for. Checked against the run itself: each kernel is computed on the
level it belongs to, is independence at every contig boundary, and is
Haldane's over its own contig's map everywhere else; the levels nest as the
pipeline builds them; and the table the run writes is the lineage.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The run, with its lineage and every kernel it computed."""
    import matplotlib as mpl
    from port.extensions.segments import recording
    from port.patch import recomb
    from port.scripts.run_cnaster import main

    from tests.fixtures import core_inference_truth
    from tests.run_config import isolated_run, write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    mpl.use("Agg")
    root = tmp_path_factory.mktemp("lineage")
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    config = write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3)

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
    """Four kernels, each one entry per segment of its level, `log 1/2` at each contig end."""
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
    """Blocks refine bins; filtering keeps a subset; the merge coarsens what survived.

    And `lengths` of every level sums to its segment count with no zero -- the
    grid the HMM restarts on.
    """
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
    """`gene_segments.tsv` is the recorded lineage: coordinates and one label column per level."""
    lineage = run["lineage"]
    table = run["table"]
    expected = lineage.table()

    assert list(table.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
