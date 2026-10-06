"""`--sal`: the rows `run_cnaster_port` substitutes from `snakes_and_ladders` (#312).

One row is admitted, the clone labelling. `snakes_and_ladders`' alpha
expansion finds the lower-energy basin, then `cnaster`'s ICM applies its
200-spot floor. What pins it:

- the Rust minimum cut returns the Python cut's labelling, so the speed it
  buys changes no answer (`backend`);
- the sequence keeps `cnaster`'s floor, the property alpha expansion alone
  lacks and the one the whole-run recovery depends on (`smoke`: it reaches
  `cnaster`, and the floor is `cnaster`'s contract);
- end to end on the dev instance, `--sal` recovers the planted clones at
  ARI 1.000 against the default's 0.919 (`end2end`, `release`);
- the same on the critical instance, per pull request (`end2end`);
- the flag's own mechanics (`infra`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.test_alpha_expansion import _lattice


@pytest.mark.backend
@pytest.mark.parametrize("n_states", [3, 6])
def test_the_rust_cut_returns_the_python_cuts_labelling(n_states: int) -> None:
    """Bitwise, on a 30 x 30 lattice at three and six labels."""
    from port.patch.icm.alpha_expansion import alpha_expansion_sweep
    from sal.backend import Backend

    field, graph, start, beta = _lattice(30, n_states, seed=4, beta=0.6)
    python, rust = start.copy(), start.copy()

    alpha_expansion_sweep(field, graph, python, beta, backend=Backend.PYTHON)
    alpha_expansion_sweep(field, graph, rust, beta, backend=Backend.RUST)

    np.testing.assert_array_equal(rust, python)


@pytest.mark.backend
def test_the_numba_descent_returns_the_python_descents_labelling() -> None:
    """`icm-numba`, measured and selectable, against sal's Python sweep.

    sal states the compiled sweep bitwise with its Python loop (#264); this
    pins that through `port`'s adapter, on 30 x 30 and four labels.
    """
    from port.extensions.label_solver import sal_icm_sweep
    from port.patch.icm.alpha_expansion import potts_graph_from
    from sal.backend import Backend
    from sal.search.icm import iterated_conditional_modes

    field, graph, start, beta = _lattice(30, 4, seed=2, beta=0.6)
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
    """No clone the sequence returns is under `min_clone_spots`.

    Sixteen labels over 1,600 spots start the problem with clones of 100,
    under the 200-spot floor, which is how the RDR stage starts on the dev
    instance. At coupling 0.6 alpha expansion alone leaves four clones over
    it and twelve of 38 to 84 spots; the ICM that follows merges them into
    three, as `cnaster` does. The global RNG is seeded because `cnaster`'s
    sweep draws from it (#45).

    **The floor is `cnaster`'s only when its sweep edits.** At coupling 1.0
    and above the first ICM epoch changes nothing, the merge is never
    reached, and clones of 1 to 19 spots survive `icm_sweep_deque` itself --
    so the sequence keeps the floor exactly as far as `cnaster` does, and
    this coupling is chosen where `cnaster` applies it.
    """
    from port.extensions.label_solver import expansion_then_floor
    from port.patch.icm.alpha_expansion import alpha_expansion_sweep

    field, graph, _, beta = _lattice(40, 16, seed=9, beta=0.6)
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
    """`alpha-rust-merge` leaves no clone under `min_clone_spots`, from sal alone.

    The fixture above, where alpha expansion alone leaves twelve clones of 38
    to 84 spots: sal's `merge_small_labels` dissolves each, and no global RNG
    is touched because none of `cnaster`'s sweep runs.
    """
    from port.extensions.label_solver import expansion_then_merge
    from port.patch.icm.alpha_expansion import alpha_expansion_sweep

    field, graph, _, beta = _lattice(40, 16, seed=9, beta=0.6)
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
    """Before the floor, the fused labelling's energy is at most both proposals'.

    The roof dual's labelled part is an autarky (sal #1125), so
    `E(fuse(a, b)) <= min(E(a), E(b))` whatever the two are. Checked on the
    floor fixture with a floor of one spot, so the merge dissolves nothing
    and only descends further from the fusion.
    """
    from port.extensions.label_solver import fusion_then_merge
    from port.patch.icm.alpha_expansion import potts_graph_from
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion
    from sal.search.icm import iterated_conditional_modes
    from sal.sim.potts import energy

    field, graph, _, beta = _lattice(40, 16, seed=9, beta=0.6)
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
    """At `beta = 0` the MAP labelling is each site's best clone, and nothing moves it.

    With no coupling every site's conditional mode is its own field argmax,
    which is where the descent starts, so it takes no step; a floor of one
    spot dissolves nothing. Against `np.argmax` directly.
    """
    from port.extensions.label_solver import sal_icm_argmax_sweep

    field, graph, start, _ = _lattice(20, 5, seed=4, beta=0.6)
    labelling = start.copy()
    sal_icm_argmax_sweep(field, graph, labelling, 0.0, min_clone_spots=1)

    np.testing.assert_array_equal(labelling, np.argmax(field, axis=1))


@pytest.mark.smoke
def test_the_argmax_descent_keeps_the_clone_floor() -> None:
    """No clone the argmax row returns is under `min_clone_spots`."""
    from port.extensions.label_solver import sal_icm_argmax_sweep

    field, graph, start, beta = _lattice(40, 16, seed=9, beta=0.6)
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
    """ARI 1.000 on the early gate's instance, with the row's labelling installed.

    `M = K = 2`, `G = 1,000`, `S = 500`, through `run_core_inference` with
    `port`'s `pipeline_clone_assignment` (the `SWAPS` row `--sal` reads) and
    `sal_options()` bound, so alpha expansion and `cnaster`'s floor label every
    spot. The per-pull-request form of the dev claim below.
    """
    from port.extensions.sal import sal_options
    from port.pipeline import SWAPS, patched, with_options
    from port.sim.truth import critical_instance

    from tests.test_core_inference_end_to_end import _adjusted_rand_index, _run

    truth = critical_instance()
    row = with_options(
        tuple(swap for swap in SWAPS if swap.name == "pipeline_clone_assignment"),
        "port.patch.hmrf:pipeline_clone_assignment",
        **sal_options(),
    )

    with patched(row):
        result = _run(truth, max_iter_outer=1, max_iter=3)

    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert _adjusted_rand_index(truth.labels, fitted) == pytest.approx(1.0)


@pytest.mark.end2end
@pytest.mark.release
def test_sal_recovers_the_planted_clones_on_the_dev_instance(tmp_path: Path) -> None:
    """ARI 1.000 against the planted labels, stated at 0.99.

    The dev instance at the figures' configuration (one outer iteration,
    three EM iterations, five states). Realized three times in three runs
    (#312); stated at 0.99 so a spot or two of drift fails no-one. The
    default arm was pinned below 0.99 while it reached 0.919; it reaches 1.0
    on main at sal `3ad4b04` and `b61dfba` alike (PR #633), so that pin and
    its run are retired (T- #632).
    """
    import warnings

    import pandas as pd
    from port.scripts.run_cnaster import main
    from port.sim.inputs import write_tmp_inputs
    from port.sim.run_config import write_run_cnaster_config
    from port.sim.truth import dev_instance
    from port.sim.unsegment import unsegment
    from sklearn.metrics import adjusted_rand_score

    truth = dev_instance()
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = write_run_cnaster_config(
        written, truth, max_iter_outer=1, max_iter=3, n_states=5
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
