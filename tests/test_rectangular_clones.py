"""`initialize_rectangular_clones` against `cnaster`'s, which can loop forever (#304, #692).

`tests/data/` holds the dev run's captured first and third call arguments (#298).
"""

from __future__ import annotations

import itertools
import signal
from typing import Any

import numpy as np
import pytest

from tests import TESTS
from tests.adapters import square_coords

DATA = TESTS / "data"

TRY_CAP = 3_000
"""`randint` calls before a seed counts as not returning (false miss < 1e-16 at 4 clones)."""

SEEDS = 120
"""Seeds per `(input, n_clones)` in the equivalence sweep: 1,800 calls."""


N_CLONES = (2, 3, 4, 5, 6)
"""Clone counts in the sweep: four blocks to three, nine blocks from five."""


def _coords(name: str) -> np.ndarray:
    coords: np.ndarray = np.load(DATA / f"{name}.npz")["coords"]

    return coords


def _assert_same(ours: Any, theirs: Any) -> None:
    np.testing.assert_array_equal(ours[1], theirs[1])
    assert ours[1].dtype == theirs[1].dtype

    for mine, reference in zip(ours[0], theirs[0], strict=True):
        np.testing.assert_array_equal(mine, reference)


class _Exhausted(Exception):
    """`cnaster`'s loop drew `TRY_CAP` assignments without returning."""


def _upstream_capped(
    monkeypatch: pytest.MonkeyPatch, coords: np.ndarray, n_clones: int, seed: int
) -> Any:
    """`cnaster`'s call, or `None` past `TRY_CAP` draws; delegates each draw unchanged."""
    from cnaster.spatial import initialize_rectangular_clones as upstream

    original = np.random.randint
    calls = [0]

    def capped(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] > TRY_CAP:
            raise _Exhausted
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np.random, "randint", capped)
        try:
            return upstream(coords, n_clones, random_state=seed)
        except _Exhausted:
            return None


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.parametrize(
    ("coords", "n_clones", "seed"),
    [
        ("returns", 4, 0),
        ("grid", 4, 1),
        ("grid", 2, 3),
        ("grid", 3, 0),
        ("grid", 1, 0),
    ],
)
def test_where_cnaster_returns_it_returns_the_same(
    coords: str, n_clones: int, seed: int
) -> None:
    """Bitwise equal to `cnaster` where it returns: the dev first call and a 12 x 40 band."""
    from cnaster.spatial import initialize_rectangular_clones as upstream
    from port.patch.spatial import initialize_rectangular_clones as replacement

    points = (
        _coords("rectangular_returns") if coords == "returns" else square_coords(12, 40)
    )

    _assert_same(
        replacement(points, n_clones, random_state=seed),
        upstream(points, n_clones, random_state=seed),
    )


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.merge
def test_bitwise_on_every_seed_cnaster_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1,800 seeds: bitwise where `cnaster` returns, a redraw exactly where it cannot."""
    from port.patch.spatial import initialize_rectangular_clones as replacement
    from sal.opt.termination import Stop

    inputs = {
        "hang": _coords("rectangular_hang"),
        "returns": _coords("rectangular_returns"),
        "grid": square_coords(12, 40),
    }
    returned = refused = 0

    for (name, points), n_clones, seed in itertools.product(
        inputs.items(), N_CLONES, range(SEEDS)
    ):
        theirs = _upstream_capped(monkeypatch, points, n_clones, seed)
        ours = replacement(points, n_clones, random_state=seed)

        assert ours.termination.reason is Stop.CONVERGED, (name, n_clones, seed)

        if theirs is None:
            assert ours.redraws >= 1, (name, n_clones, seed)
            refused += 1
        else:
            assert ours.redraws == 0, (name, n_clones, seed)
            assert ours.termination.iterations == 1
            _assert_same(ours, theirs)
            returned += 1

    assert returned + refused == len(inputs) * len(N_CLONES) * SEEDS
    assert refused > 0, "the sweep never reached the case cnaster loops on"


@pytest.mark.oracle
@pytest.mark.parametrize("n_clones", [2, 3, 4, 5])
def test_feasibility_agrees_with_enumerating_every_assignment(n_clones: int) -> None:
    """`admits_assignment` against enumerating every surjection of blocks onto clones."""
    from port.patch.spatial import admits_assignment

    p = int(np.ceil(np.sqrt(n_clones)))
    maps = np.stack(
        np.unravel_index(np.arange(n_clones**p**2), (n_clones,) * p**2), axis=1
    ).astype(np.int8)
    members = [maps == clone for clone in range(n_clones)]
    surjective = np.logical_and.reduce([member.any(axis=1) for member in members])
    members = [member[surjective].astype(np.int64) for member in members]
    generator = np.random.default_rng(692 + n_clones)
    admitted = 0

    for trial in range(200):
        n_spots = int(generator.integers(n_clones, 400))
        shares = generator.dirichlet(np.ones(p**2) * (0.3 if trial % 4 else 10))
        sizes = generator.multinomial(n_spots, shares)
        floor = 0.2 * n_spots / n_clones

        least = np.min([member @ sizes for member in members], axis=0)
        expected = bool((least > floor).any())

        assert admits_assignment(sizes, n_clones, floor) is expected, sizes
        admitted += expected

    assert 0 < admitted < 200, "the sizes never reached one of the two answers"


def _within(seconds: int, call: Any) -> bool:
    """Whether `call` returns inside `seconds`, by `SIGALRM`."""

    def expire(*_: Any) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)

    try:
        call()
    except TimeoutError:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    return True


@pytest.mark.bug
@pytest.mark.merge
def test_cnaster_does_not_return_on_the_captured_input() -> None:
    """`cnaster` runs past 5 s on the third captured call; fails when fixed."""
    from cnaster.spatial import initialize_rectangular_clones as upstream

    points = _coords("rectangular_hang")

    assert not _within(5, lambda: upstream(points, 4, random_state=0))


@pytest.mark.analytic
def test_the_dev_blocks_are_refused_and_a_redraw_passes() -> None:
    """On that input [194, 3, 77, 23] is refused and one redraw passes `cnaster`'s acceptance test."""
    from port.patch.spatial import admits_assignment, initialize_rectangular_clones
    from sal.opt.termination import Stop

    assert not admits_assignment(np.array([194, 3, 77, 23]), 4, 0.2 * 297 / 4)

    points = _coords("rectangular_hang")
    returned: list[Any] = []

    assert _within(
        20, lambda: returned.append(initialize_rectangular_clones(points, 4))
    )

    result = returned[0]
    index, labels = result

    assert result.termination.reason is Stop.CONVERGED
    assert (result.termination.iterations, result.redraws) == (2, 1)
    assert sorted(np.concatenate(index).tolist()) == list(range(len(points)))
    assert min(len(spots) for spots in index) > 0.2 * len(points) / 4
    np.testing.assert_array_equal(
        np.bincount(labels, minlength=4), [len(s) for s in index]
    )


@pytest.mark.analytic
def test_a_one_row_strip_returns_bands_and_says_infeasible() -> None:
    """A one-row strip falls back to bands of 50 and reports `Stop.INFEASIBLE` (#248)."""
    from port.patch.spatial import RECTANGLE_REDRAWS, initialize_rectangular_clones
    from sal.opt.termination import Stop

    points = square_coords(1, 200)
    returned: list[Any] = []

    assert _within(
        20, lambda: returned.append(initialize_rectangular_clones(points, 4))
    )

    result = returned[0]
    index, labels = result

    assert result.termination.reason is Stop.INFEASIBLE
    assert result.termination.iterations == RECTANGLE_REDRAWS + 1
    assert sorted(np.concatenate(index).tolist()) == list(range(len(points)))
    assert [len(spots) for spots in index] == [50, 50, 50, 50]
    assert (np.diff(labels[np.argsort(points[:, 1])]) >= 0).all(), "bands along x"
