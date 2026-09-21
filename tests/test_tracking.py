"""`--track` records what a stage did, and changes nothing when it is off.

**#251.** `snakes_and_ladders.track` is the store and `port.tracking` is the
seam: which `cnaster` stage a number belongs to, and how to open a store
without making `aim` mandatory. These pin the three properties the rest of
#251 will be built on.

`infra`: they assert `port`'s own seam, not a `cnaster` result. The one
exception is marked `warning`, because what it pins is a refusal.

**The property that matters most is the one that is hardest to see**: a run
without `--track` must be the run that existed before this module. That is
asserted here by `record_label_sweep` returning `False` and writing nothing,
and end to end by `tests/test_patched_entry_point.py`'s bitwise comparison,
which would fail if any recorded quantity were computed into the run.
"""

from __future__ import annotations

import pytest
from port.tracking import MISSING_AIM, record_label_sweep, tracking
from snakes_and_ladders.track import MemoryRun, TrackedOptimization, current, track


def sweep(
    tracked: TrackedOptimization | None = None,
    *,
    step: int = 0,
    solver: str = "icm",
    cost: float = -25_666_347.4,
    niter: int = 10,
    n_clones: int = 4,
    n_spots: int = 1980,
) -> bool:
    """One sweep's return, at the figures #247 measured on the stress instance.

    Real numbers rather than round ones: a cost of `-1.0` would pass a test
    that silently dropped the sign, and the sign is what #249 got wrong.

    A function rather than a dict splatted into the call, so every keyword is
    typed at the call site -- `**{...}` widens to `object` and would let this
    file pass `mypy --strict` while handing the seam the wrong types.
    """
    return record_label_sweep(
        step,
        solver=solver,
        cost=cost,
        niter=niter,
        n_clones=n_clones,
        n_spots=n_spots,
        tracked=tracked,
    )


@pytest.mark.infra
def test_an_untracked_run_records_nothing() -> None:
    """Outside a `track()` block the seam is inert and says so.

    The whole basis of the bitwise claim. `current()` is the null
    optimization, so the return is `False` and the caller assembles no
    diagnostics -- which is why the guard is in `record_label_sweep` rather
    than left to `record`'s own first line.
    """
    assert current().is_null

    assert sweep() is False


@pytest.mark.infra
def test_a_tracked_sweep_writes_the_series_a_reader_needs() -> None:
    """Every number the stage returned, under Aim's own names and context.

    `MemoryRun` is upstream's in-process store, so this needs no `aim` and no
    files. The context is asserted because two solvers in one run must be two
    series rather than one series with a jump in it.
    """
    store = MemoryRun()
    tracked = TrackedOptimization(store)

    assert sweep(tracked, step=7) is True

    context = {"stage": "clone_assignment", "solver": "icm"}

    assert store.last("objective", context) == pytest.approx(-25_666_347.4)
    assert store.last("niter", context) == pytest.approx(10.0)
    assert store.last("n_clones", context) == pytest.approx(4.0)
    assert store.last("n_spots", context) == pytest.approx(1980.0)


@pytest.mark.infra
def test_the_objective_keeps_cnasters_sign() -> None:
    """Recorded unnegated, because the name says nothing about the sign.

    `cnaster` maximizes its cost. A seam that negated it here would produce a
    series reading as improvement while the fit got worse -- which is exactly
    the defect #249 fixed in the label solver, and the reason this is pinned
    rather than assumed.
    """
    store = MemoryRun()
    tracked = TrackedOptimization(store)

    sweep(tracked, cost=-100.0)

    recorded = store.last("objective", {"stage": "clone_assignment", "solver": "icm"})

    assert recorded == pytest.approx(-100.0), (
        f"recorded {recorded}, so the sign was flipped on the way in"
    )


@pytest.mark.infra
def test_two_solvers_are_two_series() -> None:
    """The context separates them, so neither reads as the other's history."""
    store = MemoryRun()
    tracked = TrackedOptimization(store)

    sweep(tracked, solver="icm")
    sweep(tracked, solver="alpha", niter=2)

    assert store.last("niter", {"stage": "clone_assignment", "solver": "icm"}) == 10.0
    assert store.last("niter", {"stage": "clone_assignment", "solver": "alpha"}) == 2.0


@pytest.mark.infra
def test_the_block_binds_and_unbinds() -> None:
    """`track()` is what makes `current()` non-null, and only for the block.

    A store left bound would make every later run in the process record into
    it, which for a test session is every run after this one.
    """
    store = MemoryRun()

    with track(store):
        assert not current().is_null

    assert current().is_null


@pytest.mark.warning
def test_tracking_refuses_rather_than_recording_nothing() -> None:
    """Without `aim`, `--track` stops the run instead of running untracked.

    A user who asked to record a run and got an unrecorded one holds a
    measurement they cannot tell from a recorded one, which is worse than a
    run that did not start. Skipped where the extra *is* installed, because
    there the refusal cannot happen and asserting it would be asserting the
    environment.
    """
    try:
        import aim  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        pytest.skip("the `track` extra is installed, so there is no refusal to pin")

    with (
        pytest.raises(SystemExit) as refused,
        tracking("/tmp/port-track-should-not-exist"),
    ):
        pass

    assert str(refused.value) == MISSING_AIM
    assert "--extra track" in MISSING_AIM
