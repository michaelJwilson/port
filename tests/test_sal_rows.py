"""`--sal` rows `run_cnaster_port` substitutes from `snakes_and_ladders` (#312).

Backend equivalence, `cnaster`'s clone floor, and recovery of planted clones (ARI).
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from port.extensions.label_solver import (
    ENVIRONMENT,
    fusion_then_merge,
    sal_icm_sweep,
    solver_for,
)
from port.extensions.sal import SAL_ROWS, sal_options
from port.patch.hmrf.clone_assignment import pipeline_clone_assignment
from port.patch.icm.alpha_expansion import alpha_expansion_sweep, potts_graph_from
from port.pipeline import SWAPS, patched, with_options
from port.scripts.run_cnaster import main
from port.sim.run_config import write_for_run
from port.sim.truth import critical_instance, dev_instance
from sal.backend import Backend
from sal.search.alpha_expansion import alpha_expansion
from sal.search.icm import iterated_conditional_modes
from sal.sim.potts import energy
from sklearn.metrics import adjusted_rand_score

from tests.fixtures import (
    partition_ari,
    planted_blocky_field,
    run_planted_core_inference,
)


@pytest.mark.backend
@pytest.mark.parametrize("n_states", [3, 6])
def test_the_rust_cut_returns_the_python_cuts_labelling(n_states: int) -> None:
    """Rust cut equals the Python cut bitwise on 30 x 30 at three and six labels."""

    field, graph, start, beta = planted_blocky_field(30, n_states, seed=4, beta=0.6)
    python, rust = start.copy(), start.copy()

    alpha_expansion_sweep(field, graph, python, beta, backend=Backend.PYTHON)
    alpha_expansion_sweep(field, graph, rust, beta, backend=Backend.RUST)

    np.testing.assert_array_equal(rust, python)


@pytest.mark.backend
def test_the_numba_descent_returns_the_python_descents_labelling() -> None:
    """`icm-numba` equals sal's Python sweep bitwise through port's adapter (#264)."""

    field, graph, start, beta = planted_blocky_field(30, 4, seed=2, beta=0.6)
    compiled = start.copy()
    sal_icm_sweep(field, graph, compiled, beta)

    python = iterated_conditional_modes(
        potts_graph_from(graph, beta),
        field,
        np.random.default_rng(0),
        start=start.copy(),
        backend=Backend.PYTHON,
    )

    np.testing.assert_array_equal(compiled, np.asarray(python.labelling))


@pytest.mark.analytic
def test_the_fusion_is_no_worse_than_either_proposal() -> None:
    """The fused labelling's energy is at most both proposals' (sal #1125)."""

    field, graph, _, beta = planted_blocky_field(40, 16, seed=9, beta=0.6)
    start = np.arange(1600, dtype=np.int64) % 16
    potts = potts_graph_from(graph, beta)
    values = np.asarray(field, dtype=np.float64)

    expanded = energy(
        potts,
        values,
        alpha_expansion(
            potts, values, start=start.copy(), backend=Backend.RUST
        ).labelling,
    )
    descended = energy(
        potts,
        values,
        iterated_conditional_modes(
            potts,
            values,
            np.random.default_rng(0),
            start=np.argmax(values, axis=1).astype(np.int64),
        ).labelling,
    )
    fused = fusion_then_merge(field, graph, start.copy(), beta, min_clone_spots=1)

    assert fused.cost <= min(expanded, descended) + 1e-9


@pytest.mark.infra
def test_the_flag_binds_the_rows_solver_and_leaves_the_default_cnasters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--sal` binds the row's labelling; unbound, the row runs `cnaster`'s ICM."""

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    default = (
        inspect.signature(pipeline_clone_assignment).parameters["label_solver"].default
    )

    assert sal_options() == {"label_solver": "alpha-rust-fuse-merge"}
    assert SAL_ROWS[-1].solver == "alpha-rust-fuse-merge"
    assert solver_for(default) == "icm"


@pytest.mark.infra
def test_list_prints_the_sal_row(capsys: pytest.CaptureFixture[str]) -> None:
    """`--list` names each row with its ticket and the flag that adds it."""

    assert main(["--list"]) == 0

    printed = capsys.readouterr().out

    assert "cnaster.icm.icm_sweep_deque <- search.alpha_expansion" in printed
    assert "(#410, accuracy; --sal to add)" in printed


@pytest.mark.end2end
def test_sal_recovers_the_critical_instance(cnaster_config: None) -> None:
    """ARI 1.000 against the planted clones on the critical instance (M = K = 2, S = 500)."""

    truth = critical_instance()
    row = with_options(
        tuple(swap for swap in SWAPS if swap.name == "pipeline_clone_assignment"),
        "port.patch.hmrf:pipeline_clone_assignment",
        **sal_options(),
    )

    with patched(row):
        result = run_planted_core_inference(truth, max_iter_outer=1, max_iter=3)

    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert partition_ari(truth.labels, fitted) == pytest.approx(1.0)


@pytest.mark.end2end
@pytest.mark.release
def test_sal_recovers_the_planted_clones_on_the_dev_instance(tmp_path: Path) -> None:
    """ARI at least 0.99 against the planted labels on the dev instance (#312, #632)."""

    truth = dev_instance()
    written, config = write_for_run(
        truth, tmp_path, max_iter_outer=1, max_iter=3, n_states=5
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main([str(config), "--sal"])

    labels = pd.read_csv(
        next((tmp_path / "output").rglob("clone_labels.tsv")), sep="\t", comment="#"
    )
    column = next(c for c in labels.columns if "clone" in c.lower())
    spots = labels["barcode"].str.slice(2, 7).astype(int).to_numpy()
    fitted = np.empty(truth.labels.size, dtype=object)
    fitted[spots] = labels[column].astype(str).to_numpy()

    assert adjusted_rand_score(truth.labels, fitted.astype(str)) >= 0.99
