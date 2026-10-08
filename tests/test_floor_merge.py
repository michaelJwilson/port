"""Clone-size floor and refinement mask (#348): `cnaster`'s collapse and port's merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _problem(sizes: list[int], seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A field favouring each spot's own clone by 5, over sizes given."""
    rng = np.random.default_rng(seed)
    assignment = np.repeat(np.arange(len(sizes)), sizes)
    field = rng.normal(0.0, 0.1, (assignment.size, len(sizes)))
    field[np.arange(assignment.size), assignment] += 5.0
    return field, assignment


@pytest.mark.bug
def test_cnasters_floor_moves_every_undersized_clone_into_the_one_over_it() -> None:
    """`cnaster`'s floor of 20 over 25 + 15 x 9 spots leaves one clone; fails when fixed."""
    from port.patch.icm.interface import CsrGraph, icm_sweep
    from scipy.sparse import csr_matrix

    field, assignment = _problem([25] + [9] * 15)
    graph = CsrGraph.from_matrix(csr_matrix((assignment.size, assignment.size)))
    np.random.seed(0)  # noqa: NPY002 - cnaster's sweep draws from it

    icm_sweep(field, graph, assignment, 0.0, min_clone_spots=20)

    assert np.unique(assignment).size == 1


@pytest.mark.patch
def test_the_floor_merges_smallest_first_and_stops_when_every_clone_clears_it() -> None:
    """`floor_clones` on the same problem keeps seven clones of at least 20 spots."""
    from port.patch.icm.floor import floor_clones

    field, assignment = _problem([25] + [9] * 15)
    emptied = floor_clones(field, assignment, 20)
    counts = np.bincount(assignment)
    kept = counts[counts > 0]

    assert emptied == 16 - kept.size
    assert kept.min() >= 20
    assert kept.size > 1


@pytest.mark.patch
def test_the_floor_does_not_cross_a_masked_boundary() -> None:
    """An undersized clone with no unmasked partner keeps its spots."""
    from port.patch.icm.floor import floor_clones

    field, assignment = _problem([30, 5, 30])
    field[assignment == 1, 0] = -np.inf
    field[assignment == 1, 2] = -np.inf

    floor_clones(field, assignment, 20)

    assert (assignment[30:35] == 1).all()


@pytest.mark.infra
def test_the_mask_is_handed_only_to_the_problem_it_describes() -> None:
    from port.patch.hmrf import refinement

    mask = np.zeros((4, 3), dtype=bool)
    mask[:2, :2] = True
    mask[2:, 2] = True
    refinement._KEPT[:] = [mask]

    try:
        assert refinement.mask_for(np.array([0, 1, 2, 2]), 3) is mask
        assert refinement.mask_for(np.array([0, 2, 2, 2]), 3) is None
        assert refinement.mask_for(np.array([0, 1, 1]), 3) is None
        assert refinement.mask_for(np.array([0, 1, 1, 1]), 2) is None
    finally:
        refinement.forget()

    assert refinement.mask_for(np.array([0, 1, 2, 2]), 3) is None


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_refinement_start_is_upstreams_and_its_mask_is_kept() -> None:
    """The wrapper returns `cnaster`'s three values unchanged and keeps the mask."""
    from types import SimpleNamespace

    import cnaster.spatial as upstream
    from port.patch.hmrf.refinement import (
        forget,
        initialize_rdr_clone_refininement,
        mask_for,
    )

    rng = np.random.default_rng(5)
    coords = np.column_stack(np.unravel_index(np.arange(200), (10, 20)))
    baf = np.repeat([0, 1], 100)
    counts = np.full((50, 200), 30.0)
    config = SimpleNamespace(
        hmrf=SimpleNamespace(n_clones_rdr=2), hmm=SimpleNamespace(gmm_random_state=0)
    )
    del rng

    theirs = upstream.initialize_rdr_clone_refininement(baf, coords, counts, 50, config)
    ours = initialize_rdr_clone_refininement(baf, coords, counts, 50, config)

    try:
        np.testing.assert_array_equal(ours[0], theirs[0])
        np.testing.assert_array_equal(ours[1], theirs[1])
        assert ours[2] == theirs[2]
        assert mask_for(ours[0], ours[2]) is not None
    finally:
        forget()


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_floor_is_cnasters_unless_the_config_sets_one() -> None:
    """Unset `hmrf.min_spots_per_clone` keeps `cnaster`'s floor (#348, #403, #517)."""
    import inspect

    from cnaster.config import get_global_config
    from cnaster.icm import icm_sweep_deque
    from port.patch.hmrf.clone_assignment import pipeline_clone_assignment
    from port.patch.icm.floor import CNASTER_FLOOR, configured_floor

    default = inspect.signature(icm_sweep_deque).parameters["min_clone_spots"].default
    assert default == CNASTER_FLOOR

    section = getattr(get_global_config(), "hmrf", None)
    key = getattr(section, "min_spots_per_clone", None)
    assert configured_floor() == (CNASTER_FLOOR if key is None else int(key))

    parameters = inspect.signature(pipeline_clone_assignment).parameters
    assert parameters["floor_merge"].default is False


@pytest.mark.patch
def test_the_mask_keeps_the_columns_cnaster_relabels_survivors_to() -> None:
    """`compact` keeps mask columns in `run_core_inference`'s ascending relabel order."""
    from port.patch.hmrf import refinement

    mask = np.eye(4, dtype=bool)
    assignment = np.array([3, 0, 3, 0])
    survivors, relabelled = np.unique(assignment, return_inverse=True)

    refinement._KEPT[:] = [mask]
    try:
        refinement.compact(assignment)
        kept = refinement._KEPT[0]
    finally:
        refinement.forget()

    np.testing.assert_array_equal(kept, mask[:, survivors])
    assert kept.shape[1] == relabelled.max() + 1


@pytest.mark.infra
@pytest.mark.parametrize("flag", ["--refinement-mask", "--floor-merge"])
def test_no_patch_refuses_a_flag_nothing_would_read(
    flag: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-patch` refuses the mask and floor flags, which only port's assignment reads (#466)."""
    import cnaster.scripts.run_cnaster as pipeline
    from port.scripts.run_cnaster import main

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    ran: list[bool] = []
    monkeypatch.setattr(pipeline, "run_cnaster", lambda *_: ran.append(True))

    with pytest.raises(SystemExit):
        main(["--no-patch", flag, "--no-rust", str(config)])

    assert not ran


@pytest.mark.smoke
@pytest.mark.parametrize("flag", ["mask", "floor", "shift"])
def test_a_delegated_assignment_says_it_drops_the_mask_floor_or_shift(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a tumour proportion the call delegates to `cnaster` and says what it drops (#466)."""
    import contextlib

    from port.patch.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf import clone_assignment, refinement
    from port.pipeline import with_attributes

    said: list[str] = []
    monkeypatch.setattr(clone_assignment, "UPSTREAM", lambda *_, **__: "cnaster")
    monkeypatch.setattr(clone_assignment.logger, "warning_once", said.append)

    with contextlib.ExitStack() as stack:
        if flag == "mask":
            refinement._KEPT.append(np.ones((4, 2), dtype=bool))
            stack.callback(refinement.forget)

        # NB untyped: the nine positional inputs are never read on this path.
        assign: Any = clone_assignment.pipeline_clone_assignment
        result = assign(
            *[None] * 9,
            single_tumor_prop=np.ones(4),
            hmmclass=with_attributes(hmm_nophasing, apply_logmu_shift=flag == "shift"),
            floor_merge=flag == "floor",
        )

    named = {"mask": "--refinement-mask", "floor": "--floor-merge", "shift": "--shift"}

    assert result == "cnaster"
    assert len(said) == 1
    assert named[flag] in said[0]
    assert [name for name in named.values() if name in said[0]] == [named[flag]]


@pytest.mark.merge
@pytest.mark.end2end
def test_sal_recovers_dev_where_the_hard_mask_froze_the_baf_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--sal` recovers `dev`'s planted clones where the hard mask froze the BAF boundary (#467)."""
    import numpy as np
    from port.patch.hmrf import refinement
    from port.qa.audit import audit_truth
    from port.sim import truth as sim_truth

    soft, _ = audit_truth(sim_truth.dev_instance(), ["--sal"])

    assert soft.ari >= 0.99, soft.ari

    monkeypatch.setattr(refinement, "MASK_PENALTY", np.inf)
    hard, _ = audit_truth(sim_truth.dev_instance(), ["--sal"])

    assert hard.ari < 0.9, hard.ari


@pytest.mark.oracle
def test_the_floor_is_sals_oracle_on_random_problems() -> None:
    """`floor_clones` equals `sal`'s oracle `_floor_smallest_first` on 300 random problems (#777)."""
    from port.patch.icm.floor import floor_clones
    from sal.search.icm import _floor_smallest_first

    rng = np.random.default_rng(0)
    for _ in range(300):
        n, q = int(rng.integers(5, 300)), int(rng.integers(2, 9))
        field = rng.normal(size=(n, q)) * rng.choice([0.1, 1.0, 10.0])
        field[rng.random((n, q)) < rng.uniform(0.0, 0.33)] = -np.inf
        start = rng.choice(q, n, p=rng.dirichlet(np.full(q, 0.3)))
        floor = int(rng.integers(1, max(2, n // 2)))
        ours, theirs = start.copy(), start.copy()
        floor_clones(field, ours, floor)
        _floor_smallest_first(theirs, field, floor)

        np.testing.assert_array_equal(ours, theirs)
