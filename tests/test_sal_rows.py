"""`--sal` rows `run_cnaster_port` substitutes from `snakes_and_ladders` (#312).

Backend equivalence, `cnaster`'s clone floor, and recovery of planted clones (ARI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import (
    partition_ari,
    planted_blocky_field,
    run_planted_core_inference,
)


@pytest.mark.backend
@pytest.mark.parametrize("n_states", [3, 6])
def test_the_rust_cut_returns_the_python_cuts_labelling(n_states: int) -> None:
    """Rust cut equals the Python cut bitwise on 30 x 30 at three and six labels."""
    from port.patch.icm.alpha_expansion import alpha_expansion_sweep
    from sal.backend import Backend

    field, graph, start, beta = planted_blocky_field(30, n_states, seed=4, beta=0.6)
    python, rust = start.copy(), start.copy()

    alpha_expansion_sweep(field, graph, python, beta, backend=Backend.PYTHON)
    alpha_expansion_sweep(field, graph, rust, beta, backend=Backend.RUST)

    np.testing.assert_array_equal(rust, python)


@pytest.mark.backend
def test_the_numba_descent_returns_the_python_descents_labelling() -> None:
    """`icm-numba` equals sal's Python sweep bitwise through port's adapter (#264)."""
    from port.extensions.label_solver import sal_icm_sweep
    from port.patch.icm.alpha_expansion import potts_graph_from
    from sal.backend import Backend
    from sal.search.icm import iterated_conditional_modes

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


@pytest.mark.smoke
@pytest.mark.merge
def test_the_sequence_keeps_cnasters_clone_floor() -> None:
    """No returned clone is under `min_clone_spots` at coupling 0.6 (#45)."""
    from port.patch.icm.alpha_expansion import alpha_expansion_sweep
    from port.sandbox.extensions.label_solvers import expansion_then_floor

    field, graph, _, beta = planted_blocky_field(40, 16, seed=9, beta=0.6)
    start = np.arange(1600, dtype=np.int64) % 16

    alone = start.copy()
    alpha_expansion_sweep(field, graph, alone, beta)
    sizes_alone = np.bincount(alone, minlength=16)

    np.random.seed(0)  # noqa: NPY002 -- cnaster's sweep reads the legacy state
    floored = start.copy()
    expansion_then_floor(field, graph, floored, beta, min_clone_spots=200)
    sizes = np.bincount(floored, minlength=16)

    assert np.any((sizes_alone > 0) & (sizes_alone < 200)), (
        "the fixture no longer exercises the floor"
    )
    assert np.all((sizes == 0) | (sizes >= 200)), f"clone sizes {sizes}"


@pytest.mark.smoke
def test_the_merge_keeps_cnasters_clone_floor_without_cnaster() -> None:
    """`alpha-rust-merge` leaves no clone under `min_clone_spots`, using sal alone."""
    from port.patch.icm.alpha_expansion import alpha_expansion_sweep
    from port.sandbox.extensions.label_solvers import expansion_then_merge

    field, graph, _, beta = planted_blocky_field(40, 16, seed=9, beta=0.6)
    start = np.arange(1600, dtype=np.int64) % 16

    alone = start.copy()
    alpha_expansion_sweep(field, graph, alone, beta)
    merged = start.copy()
    expansion_then_merge(field, graph, merged, beta, min_clone_spots=200)

    sizes_alone = np.bincount(alone, minlength=16)
    sizes = np.bincount(merged, minlength=16)

    assert np.any((sizes_alone > 0) & (sizes_alone < 200))
    assert np.all((sizes == 0) | (sizes >= 200)), f"clone sizes {sizes}"


@pytest.mark.analytic
def test_the_fusion_is_no_worse_than_either_proposal() -> None:
    """The fused labelling's energy is at most both proposals' (sal #1125)."""
    from port.extensions.label_solver import fusion_then_merge
    from port.patch.icm.alpha_expansion import potts_graph_from
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion
    from sal.search.icm import iterated_conditional_modes
    from sal.sim.potts import energy

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


@pytest.mark.analytic
def test_the_argmax_descent_is_the_argmax_without_coupling() -> None:
    """At `beta = 0` the descent returns `np.argmax` of the field."""
    from port.sandbox.extensions.label_solvers import sal_icm_argmax_sweep

    field, graph, start, _ = planted_blocky_field(20, 5, seed=4, beta=0.6)
    labelling = start.copy()
    sal_icm_argmax_sweep(field, graph, labelling, 0.0, min_clone_spots=1)

    np.testing.assert_array_equal(labelling, np.argmax(field, axis=1))


@pytest.mark.smoke
def test_the_argmax_descent_keeps_the_clone_floor() -> None:
    """No clone the argmax row returns is under `min_clone_spots`."""
    from port.sandbox.extensions.label_solvers import sal_icm_argmax_sweep

    field, graph, start, beta = planted_blocky_field(40, 16, seed=9, beta=0.6)
    labelling = start.copy()
    sal_icm_argmax_sweep(field, graph, labelling, beta, min_clone_spots=200)
    sizes = np.bincount(labelling, minlength=16)

    assert np.all((sizes == 0) | (sizes >= 200)), f"clone sizes {sizes}"


@pytest.mark.infra
def test_the_flag_binds_the_rows_solver_and_leaves_the_default_cnasters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--sal` binds the row's labelling; unbound, the row runs `cnaster`'s ICM."""
    import inspect

    from port.extensions.label_solver import ENVIRONMENT, solver_for
    from port.extensions.sal import SAL_ROWS, sal_options
    from port.patch.hmrf.clone_assignment import pipeline_clone_assignment

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
    from port.scripts.run_cnaster import main

    assert main(["--list"]) == 0

    printed = capsys.readouterr().out

    assert "cnaster.icm.icm_sweep_deque <- search.alpha_expansion" in printed
    assert "(#410, accuracy; --sal to add)" in printed


@pytest.mark.end2end
def test_sal_recovers_the_critical_instance(cnaster_config: None) -> None:
    """ARI 1.000 against the planted clones on the critical instance (M = K = 2, S = 500)."""
    from port.extensions.sal import sal_options
    from port.pipeline import SWAPS, patched, with_options
    from port.sim.truth import critical_instance

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
    import warnings

    import pandas as pd
    from port.scripts.run_cnaster import main
    from port.sim.run_config import write_for_run
    from port.sim.truth import dev_instance
    from sklearn.metrics import adjusted_rand_score

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
