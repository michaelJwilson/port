"""`hmrf.fixed_assignment` holds the clones where they started (#362).

`cnaster` reads the flag in its label solve (`hmrf.py:287`) and `port`'s
`pipeline_clone_assignment` in its own (`clone_assignment.py:424`): set, no
ICM move, floor merge, pairwise merge or clone loss runs. The oracle arm of
`tests.sim_audit` rests on it, so it is pinned here on a start that is
deliberately not the planted labelling -- a solve that ran would move it.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from tests.adapters import from_core_inference_truth
from tests.fixtures import core_inference_truth


@contextmanager
def _fixed() -> Iterator[None]:
    """`hmrf.fixed_assignment` set on the installed configuration, restored after."""
    from cnaster.config import get_global_config

    hmrf = get_global_config().hmrf
    previous = hmrf.fixed_assignment
    hmrf.fixed_assignment = True

    try:
        yield
    finally:
        hmrf.fixed_assignment = previous


def _run(start: list[np.ndarray], port: bool) -> Any:
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.pipeline import SWAPS, patched

    truth = core_inference_truth(
        n_clones=3, n_states=4, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    kwargs = from_core_inference_truth(truth).as_kwargs()
    kwargs["initial_clone_index"] = start

    with warnings.catch_warnings(), _fixed():
        warnings.simplefilter("ignore")

        if port:
            with patched(SWAPS):
                from cnaster import hmrf

                return hmrf.run_core_inference(**kwargs, hmmclass=hmm_nophasing)

        from cnaster.hmrf import run_core_inference

        return run_core_inference(**kwargs, hmmclass=hmm_nophasing)


@pytest.mark.patch
@pytest.mark.parametrize("port", [False, True], ids=["cnaster", "port"])
def test_a_fixed_assignment_returns_the_start(cnaster_config: None, port: bool) -> None:
    """Three vertical bands, not the planted clones, come back spot for spot."""
    n_spots = 25 * 40
    start = np.arange(n_spots) * 3 // n_spots
    result = _run([np.flatnonzero(start == c) for c in range(3)], port)

    np.testing.assert_array_equal(np.asarray(result["new_assignment"]), start)
