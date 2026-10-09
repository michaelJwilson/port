"""The set-aside `--sal` label solver rows keep `cnaster`'s clone floor (#312).

`port.sandbox.extensions.label_solvers`; the live rows are `tests.test_sal_rows`'.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.patch.icm.alpha_expansion import alpha_expansion_sweep
from port.sandbox.extensions.label_solvers import (
    expansion_then_floor,
    expansion_then_merge,
    sal_icm_argmax_sweep,
)

from tests.fixtures import (
    planted_blocky_field,
)


@pytest.mark.analytic
@pytest.mark.merge
def test_the_sequence_keeps_cnasters_clone_floor() -> None:
    """No returned clone is under `min_clone_spots` at coupling 0.6 (#45)."""

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


@pytest.mark.analytic
def test_the_merge_keeps_cnasters_clone_floor_without_cnaster() -> None:
    """`alpha-rust-merge` leaves no clone under `min_clone_spots`, using sal alone."""

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
def test_the_argmax_descent_is_the_argmax_without_coupling() -> None:
    """At `beta = 0` the descent returns `np.argmax` of the field."""

    field, graph, start, _ = planted_blocky_field(20, 5, seed=4, beta=0.6)
    labelling = start.copy()
    sal_icm_argmax_sweep(field, graph, labelling, 0.0, min_clone_spots=1)

    np.testing.assert_array_equal(labelling, np.argmax(field, axis=1))


@pytest.mark.analytic
def test_the_argmax_descent_keeps_the_clone_floor() -> None:
    """No clone the argmax row returns is under `min_clone_spots`."""

    field, graph, start, beta = planted_blocky_field(40, 16, seed=9, beta=0.6)
    labelling = start.copy()
    sal_icm_argmax_sweep(field, graph, labelling, beta, min_clone_spots=200)
    sizes = np.bincount(labelling, minlength=16)

    assert np.all((sizes == 0) | (sizes >= 200)), f"clone sizes {sizes}"
