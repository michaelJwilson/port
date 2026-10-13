"""T- #829: `potts_stream` polishes every solver by ICM then sal's label merge, a sampler's inside its anneal."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _patch(seed: int) -> Any:
    """A 10 x 10 hex patch with a 4-label field, as `potts_stream._warm` builds it."""
    from port.studies.potts_stream import hex_graph

    rows, cols = np.divmod(np.arange(100), 10)
    points = np.column_stack([cols + 0.5 * (rows % 2), rows * np.sqrt(3) / 2])
    indptr, indices, weights = hex_graph(points)
    field = np.random.default_rng(seed).normal(size=(100, 4)) * 0.5
    return SimpleNamespace(realization=-1, field=field, planted=field.argmax(1), indptr=indptr, indices=indices,
                           weights=weights, spatial_weight=0.8, n_spots=100)  # fmt: skip


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("solver", "setting"),
    [
        ("sal:wolff-heat-bath", {"t_start": 2.0, "sweeps": 40}),
        ("sal:swendsen-wang-heat-bath", {"t_start": 2.0, "sweeps": 40}),
        ("sal:anneal", {"t_start": 2.0, "sweeps": 40}),
        ("sal:trws", None),
    ],
)
def test_each_polish_stage_never_raises_the_energy(
    solver: str, setting: dict[str, float] | None
) -> None:
    """Raw >= after ICM >= after the merge, and the merge count is the labels it removed."""
    from port.studies.potts_stream import solve_labelling

    row = solve_labelling(_patch(1), solver, 0, setting)

    assert "error" not in row, row.get("trace")
    assert row["energy"] >= row["polished"] - 1e-9 >= row["both"] - 2e-9
    assert row["merges"] >= 0


@pytest.mark.oracle
@pytest.mark.parametrize(
    "solver", ["sal:wolff-heat-bath", "sal:swendsen-wang-heat-bath", "sal:anneal"]
)
def test_an_annealed_runs_merge_stage_is_sals_merge_of_its_icm_stage(
    solver: str,
) -> None:
    """`_sample`'s merge stage (`Polish.ICM_MERGE`) is `merged` of its ICM stage, bitwise: one merge, in or out of the anneal."""
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph
    from port.studies.potts_stream import STAGES, _sample, merged

    patch = _patch(2)
    graph = potts_graph_from(
        CsrGraph(patch.indptr, patch.indices, patch.weights), patch.spatial_weight
    )
    start = np.random.default_rng(3).integers(0, 4, 100)
    run = _sample(
        solver,
        patch.field,
        start,
        np.random.default_rng(4),
        graph,
        {"t_start": 2.0, "sweeps": 40},
    )

    assert tuple(stage.name for stage in run.stages) == STAGES
    np.testing.assert_array_equal(
        run.stages[2].best, merged(graph, patch.field, np.asarray(run.stages[1].best))
    )
