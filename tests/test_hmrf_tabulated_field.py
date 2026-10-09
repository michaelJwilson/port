"""Tabulated field against the fused one, bitwise (#433); the referee is the call it replaces."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pytest
from port.patch.hmrf.fused_field import fused_spot_clone_field
from port.patch.hmrf.tabulated_field import spot_clone_field as dispatch

from tests.fixtures import (
    SpotCloneField,
    fused_field_arguments,
    fused_field_of,
    spot_clone_field,
)


def _both(fixture: SpotCloneField, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return fused_field_of(fixture, weight), fused_field_of(
        fixture, weight, tabulated=True
    )


def _off_default(n_states: int, n_clones: int) -> tuple[SpotCloneField, np.ndarray]:
    """A per-spot RDR weight off its default."""
    fixture = spot_clone_field(n_states=n_states, n_clones=n_clones)
    return fixture, np.random.default_rng(3).uniform(0.5, 1.5, fixture.n_spots)


def _zero_edges() -> tuple[SpotCloneField, np.ndarray]:
    """No baseline, no trials, and successes over trials, which `cnaster` scores 0."""
    fixture = spot_clone_field(n_states=4, n_clones=2)
    fixture.base_nb_mean[0, ::3] = 0.0
    fixture.total_bb_RD[0, 1::3] = 0.0
    fixture.counts_bb[0, 2::3] = fixture.total_bb_RD[0, 2::3] + 1.0
    return fixture, np.ones(fixture.n_spots)


@pytest.mark.patch
@pytest.mark.parametrize(
    "build",
    [
        *(partial(_off_default, s, c) for s, c in ((7, 3), (5, 5), (3, 1))),
        _zero_edges,
    ],
    ids=["7x3", "5x5", "3x1", "zero-edges"],
)
def test_the_tabulated_field_is_bitwise_the_fused_one(
    build: Callable[[], tuple[SpotCloneField, np.ndarray]],
) -> None:
    """Identical output under a per-spot RDR weight off its default, and the edges `cnaster` scores 0 are 0 here too."""
    fixture, weight = build()

    fused, tabulated = _both(fixture, weight)

    np.testing.assert_array_equal(tabulated, fused)


@pytest.mark.patch
def test_counts_that_cannot_index_a_table_go_to_the_fused_kernel() -> None:
    """A non-integer count is scored by the fused kernel, not rounded into a row."""

    dispatched: Any = dispatch
    fused: Any = fused_spot_clone_field

    fixture = spot_clone_field(n_states=4, n_clones=2)
    fixture.counts_nb[0, 0] += 0.5
    weight = np.ones(fixture.n_spots)
    shape = (fixture.n_spots, fixture.n_clones)

    np.testing.assert_array_equal(
        dispatched(*fused_field_arguments(fixture, weight), np.empty(shape)),
        fused(*fused_field_arguments(fixture, weight), np.empty(shape)),
    )
