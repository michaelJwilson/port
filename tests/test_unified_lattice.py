"""`port.patch.lattice`'s one recursion against cnaster's four, bitwise (#205).

State-space width and site-dependence of the transition are arguments; nothing is
reassociated, so the same `logsumexp` runs in the same order.
"""

from dataclasses import replace

import numpy as np
import pytest

from tests.builders import (
    LatticeInputs,
    cnaster_lattice,
    random_lattice,
    unified_lattice,
)

SPOTS = 3
"""More than one, so both spot sums are non-trivial."""


def _inputs(n_states: int, *, phased: bool, seed: int = 5) -> LatticeInputs:
    """Ragged segments, a proper transition, and a drawn (site-varying) switch kernel."""
    return random_lattice(n_states, (7, 11, 5), SPOTS, phased=phased, seed=seed)


_cnaster, _unified = cnaster_lattice, unified_lattice


@pytest.mark.patch
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
@pytest.mark.parametrize("n_states", [2, 5])
def test_the_unified_recursion_is_cnasters_bitwise(
    which: str, phased: bool, n_states: int
) -> None:
    """All four of cnaster's recursions from one kernel, bitwise."""
    inputs = _inputs(n_states, phased=phased)

    expected = _cnaster(which, inputs, phased=phased)
    actual = _unified(which, inputs, phased=phased)

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), (
        f"{which} differs by at most {np.abs(actual - expected).max():.3e}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("n_states", [2, 5])
def test_the_state_axis_decides_which_chain_is_being_run(n_states: int) -> None:
    """`is_phased` reads the chain from `n_states` and the emission, and refuses a
    mismatch.
    """
    from port.patch.lattice import is_phased

    unphased = _inputs(n_states, phased=False)
    phased = _inputs(n_states, phased=True)

    assert not is_phased(unphased.log_emission, n_states)
    assert is_phased(phased.log_emission, n_states)

    with pytest.raises(ValueError, match="which is neither"):
        is_phased(np.zeros((3 * n_states, 4, 1)), n_states)


@pytest.mark.smoke
@pytest.mark.parametrize("n_states", [2, 5])
def test_the_two_spot_sums_agree_bitwise(n_states: int) -> None:
    """cnaster's whole-block and per-row spot sums agree bitwise under `numba`."""
    from port.patch.lattice import spot_sums_agree

    inputs = _inputs(n_states, phased=True)

    assert spot_sums_agree(inputs.log_emission[:, 0, :])
    assert spot_sums_agree(np.ascontiguousarray(inputs.log_emission[:, 0, :]))


@pytest.mark.smoke
def test_the_backward_pass_does_not_read_the_start_probability() -> None:
    """`log_startprob` is accepted but unread, as in cnaster."""
    inputs = _inputs(4, phased=True)

    with_start = _unified("backward_lattice", inputs, phased=True)

    scrambled = replace(inputs, log_startprob=inputs.log_startprob[::-1].copy())

    assert np.array_equal(
        with_start, _unified("backward_lattice", scrambled, phased=True)
    )
