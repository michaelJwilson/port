"""`port.patch.lattice`'s one recursion against cnaster's four, bitwise (#205).

State-space width and site-dependence of the transition are arguments; nothing is
reassociated, so the same `logsumexp` runs in the same order.
"""

from dataclasses import dataclass

import numpy as np
import pytest

SPOTS = 3
"""More than one, so both spot sums are non-trivial."""


@dataclass(frozen=True)
class LatticeInputs:
    """What either recursion takes, at `cnaster`'s shapes."""

    lengths: np.ndarray
    log_transmat: np.ndarray
    log_startprob: np.ndarray
    log_emission: np.ndarray
    log_sitewise_transmat: np.ndarray
    n_states: int


def _inputs(n_states: int, *, phased: bool, seed: int = 5) -> LatticeInputs:
    """Ragged segments, a proper transition, and a drawn (site-varying) switch kernel."""
    generator = np.random.default_rng(seed)

    lengths = np.array([7, 11, 5], dtype=np.int64)
    n_obs = int(lengths.sum())
    rows = 2 * n_states if phased else n_states

    transition = generator.random((n_states, n_states)) + 0.5
    transition /= transition.sum(axis=1, keepdims=True)

    start = generator.random(n_states) + 0.5
    start /= start.sum()

    return LatticeInputs(
        lengths=lengths,
        log_transmat=np.log(transition),
        log_startprob=np.log(start),
        log_emission=generator.normal(-2.0, 1.5, (rows, n_obs, SPOTS)),
        log_sitewise_transmat=np.log(generator.uniform(1e-4, 0.4, n_obs)),
        n_states=n_states,
    )


def _cnaster(which: str, inputs: LatticeInputs, *, phased: bool) -> np.ndarray:
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased

    klass = hmm_phased if phased else hmm_nophasing
    recursion = getattr(klass, which)

    result: np.ndarray = recursion(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        inputs.log_emission,
        inputs.log_sitewise_transmat,
    )
    return result


def _unified(which: str, inputs: LatticeInputs, *, phased: bool) -> np.ndarray:
    from port.patch import lattice

    recursion = getattr(lattice, which)

    result: np.ndarray = recursion(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        inputs.log_emission,
        inputs.log_sitewise_transmat,
        inputs.n_states,
        phased,
    )
    return result


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

    scrambled = LatticeInputs(
        lengths=inputs.lengths,
        log_transmat=inputs.log_transmat,
        log_startprob=inputs.log_startprob[::-1].copy(),
        log_emission=inputs.log_emission,
        log_sitewise_transmat=inputs.log_sitewise_transmat,
        n_states=inputs.n_states,
    )

    assert np.array_equal(
        with_start, _unified("backward_lattice", scrambled, phased=True)
    )
