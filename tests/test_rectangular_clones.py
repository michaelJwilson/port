"""`initialize_rectangular_clones`, which `cnaster` can loop in forever (#304).

The two inputs under `tests/data/` are the arguments the dev instance's run
passed on its first and third calls, captured after #298's normal clone made
the third one hang: 297 spots into four clones, where the Dirichlet-drawn
boundaries leave a block too small for any assignment to pass `cnaster`'s
20 per cent test.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent / "data"


def _coords(name: str) -> np.ndarray:
    coords: np.ndarray = np.load(DATA / f"{name}.npz")["coords"]

    return coords


def _grid(rows: int, columns: int) -> np.ndarray:
    return np.array([(r, c) for r in range(rows) for c in range(columns)])


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
    """Bitwise, the index lists and the labels, where `cnaster` returns.

    The dev instance's first call, and a 12 x 40 band at 1 to 4 clones. The
    replacement consumes the random stream in `cnaster`'s order, so where
    `cnaster` succeeds within `RECTANGLE_TRIES` the two are the same draws.
    """
    from cnaster.spatial import initialize_rectangular_clones as upstream
    from port.patch.spatial import initialize_rectangular_clones as replacement

    points = _coords("rectangular_returns") if coords == "returns" else _grid(12, 40)

    theirs = upstream(points, n_clones, random_state=seed)
    ours = replacement(points, n_clones, random_state=seed)

    np.testing.assert_array_equal(ours[1], theirs[1])

    for mine, reference in zip(ours[0], theirs[0], strict=True):
        np.testing.assert_array_equal(mine, reference)


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
def test_cnaster_does_not_return_on_the_captured_input() -> None:
    """`cnaster`'s loop, on the third call's arguments, runs past 5 s.

    Written to fail when `cnaster` redraws its boundaries or bounds the loop.
    A returning call takes milliseconds on 297 spots, so five seconds is a
    thousandfold margin rather than a guess.
    """
    from cnaster.spatial import initialize_rectangular_clones as upstream

    points = _coords("rectangular_hang")

    assert not _within(5, lambda: upstream(points, 4, random_state=0))


@pytest.mark.analytic
def test_it_returns_a_partition_that_passes_cnasters_own_test() -> None:
    """On the input `cnaster` loops on: every spot once, no clone below 20%.

    `cnaster`'s own acceptance test, applied to the answer: each clone holds
    more than a fifth of an equal share. Realized 84, 108, 60 and 45 spots
    of 297, against a threshold of 14.85.
    """
    from port.patch.spatial import initialize_rectangular_clones

    points = _coords("rectangular_hang")
    returned: list[Any] = []

    assert _within(
        20, lambda: returned.append(initialize_rectangular_clones(points, 4))
    )

    index, labels = returned[0]

    assert sorted(np.concatenate(index).tolist()) == list(range(len(points)))
    assert min(len(spots) for spots in index) > 0.2 * len(points) / 4
    np.testing.assert_array_equal(
        np.bincount(labels, minlength=4), [len(s) for s in index]
    )
