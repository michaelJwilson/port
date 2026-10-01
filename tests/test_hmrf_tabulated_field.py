"""The tabulated field against the fused one, bitwise (#433).

`fused_spot_clone_field` is pinned bitwise to `cnaster`'s two-step in
`tests/test_hmrf_fused_field.py`; this pins the tabulated kernel bitwise to
the fused one, so the chain reaches `cnaster` without a tolerance. The
referee is the call it replaces, so the marker is `patch`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.fixtures import SpotCloneField, spot_clone_field


def _arguments(fixture: SpotCloneField, weight: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.log_mu,
        fixture.alphas,
        fixture.p_binom,
        fixture.taus,
        fixture.pred,
        weight,
    )


def _both(fixture: SpotCloneField, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from port.patch.hmrf.fused_field import fused_spot_clone_field
    from port.patch.hmrf.tabulated_field import tabulated_spot_clone_field

    # NB `Any`: the kernels are `numba` dispatchers, whose stubs take no
    #    positional unpacking.
    fused: Any = fused_spot_clone_field
    tabulated: Any = tabulated_spot_clone_field
    shape = (fixture.n_spots, fixture.n_clones)
    arguments = _arguments(fixture, weight)

    return (
        fused(*arguments, np.empty(shape)),
        tabulated(*arguments, np.empty(shape)),
    )


@pytest.mark.patch
@pytest.mark.parametrize(
    ("n_states", "n_clones"), [(7, 3), (5, 5), (3, 1)], ids=["7x3", "5x5", "3x1"]
)
def test_the_tabulated_field_is_bitwise_the_fused_one(
    n_states: int, n_clones: int
) -> None:
    """Identical output under a per-spot RDR weight off its default."""
    fixture = spot_clone_field(n_states=n_states, n_clones=n_clones)
    weight = np.random.default_rng(3).uniform(0.5, 1.5, fixture.n_spots)

    fused, tabulated = _both(fixture, weight)

    np.testing.assert_array_equal(tabulated, fused)


@pytest.mark.patch
def test_the_edges_cnaster_scores_zero_are_zero_here_too() -> None:
    """No baseline, no trials, and more successes than trials: each scores 0.

    Put where they are read -- the first bin of every spot -- so a kernel that
    tabulated past them, or indexed a negative `n - k`, would differ or fail.
    """
    fixture = spot_clone_field(n_states=4, n_clones=2)
    fixture.base_nb_mean[0, ::3] = 0.0
    fixture.total_bb_RD[0, 1::3] = 0.0
    fixture.counts_bb[0, 2::3] = fixture.total_bb_RD[0, 2::3] + 1.0
    weight = np.ones(fixture.n_spots)

    fused, tabulated = _both(fixture, weight)

    np.testing.assert_array_equal(tabulated, fused)


@pytest.mark.patch
def test_counts_that_cannot_index_a_table_go_to_the_fused_kernel() -> None:
    """A non-integer count is scored by the fused kernel, not rounded into a row."""
    from port.patch.hmrf.fused_field import fused_spot_clone_field
    from port.patch.hmrf.tabulated_field import spot_clone_field as dispatch

    dispatched: Any = dispatch
    fused: Any = fused_spot_clone_field

    fixture = spot_clone_field(n_states=4, n_clones=2)
    fixture.counts_nb[0, 0] += 0.5
    weight = np.ones(fixture.n_spots)
    shape = (fixture.n_spots, fixture.n_clones)

    np.testing.assert_array_equal(
        dispatched(*_arguments(fixture, weight), np.empty(shape)),
        fused(*_arguments(fixture, weight), np.empty(shape)),
    )


@pytest.mark.patch
def test_the_one_pass_integer_check_decides_as_the_three_pass_one() -> None:
    """`_integral` against `_integral_reference`, the decision bit for bit (#569).

    Integer and float counts, strided views, a fraction, a negative, a NaN,
    `LIMIT` and one below it, and an empty array: every way the table can
    be refused, and the ways it cannot.
    """
    from port.patch.hmrf.tabulated_field import LIMIT, _integral, _integral_reference

    rng = np.random.default_rng(5)
    counts = rng.poisson(3, (40, 2, 70))
    floats = counts.astype(np.float64)
    cases = [
        counts[:, 0, :],
        counts[:, 1, :],
        floats[:, 1, :],
        floats[:, 1, :] + 0.5,
        -counts[:, 0, :] - 1,
        np.where(rng.random((40, 70)) < 0.01, np.nan, floats[:, 0, :]),
        np.full((3, 4), float(LIMIT)),
        np.full((3, 4), float(LIMIT - 1)),
        np.empty((0, 5)),
        floats[:, 0, 0],
    ]
    for case in cases:
        assert _integral(case) is _integral_reference(case)
    assert [_integral(c) for c in cases] == [
        True, True, True, False, False, False, False, True, True, True
    ]  # fmt: skip
