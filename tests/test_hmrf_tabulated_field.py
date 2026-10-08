"""Tabulated field against the fused one, bitwise (#433); the referee is the call it replaces."""

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
    """No baseline, no trials, and successes over trials each score 0, as in `cnaster`."""
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
