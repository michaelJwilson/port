"""`port.sandbox.extensions.shared_decode` on `tests.test_integer_copy_patch`'s planted pairs (#362)."""

from __future__ import annotations

import pytest
from port.sandbox.extensions.shared_decode import shared_decode

from tests.test_integer_copy_patch import BASE, HIGH, _bulk


@pytest.mark.end2end
@pytest.mark.parametrize("extra", HIGH, ids=str)
def test_the_likelihood_decodes_the_planted_pairs_under_a_cap_of_twelve(
    extra: tuple[int, int],
) -> None:
    """Every planted pair exactly, at a stated cap of 12 and the shift held at 0."""

    bulk, path = _bulk(extra)
    decoded = shared_decode(
        [(path, bulk, 0.0)],
        n_states=5,
        normal=0,
        max_total_copy=12,
    )

    assert [(int(a), int(b)) for a, b in decoded.states] == [*BASE, extra]


@pytest.mark.analytic
def test_the_normal_state_is_one_one_by_definition() -> None:
    """A state named normal decodes `(1, 1)` though its counts say `(2, 1)`."""

    bulk, path = _bulk(HIGH[0])
    decoded = shared_decode(
        [(path, bulk, 0.0)],
        n_states=5,
        normal=1,
        max_total_copy=12,
    )

    assert tuple(decoded.states[1]) == (1, 1)
    assert tuple(decoded.states[0]) == (1, 1)
