"""One recursion for `cnaster`'s four, bitwise (#205).

**Four recursions, two arguments.** `hmm_nophasing` runs a `K`-state chain
under a transition that does not move along it; `hmm_phased` overrides both
passes to run a `2K`-state chain under a transition rebuilt per site. The
difference is how wide the state space is and whether the transition depends
on the site, and `port.patch.lattice` takes both as arguments.

**Bitwise is the bar and it is reached**, which is what makes this a
simplification rather than a rewrite: nothing is reassociated, so the same
`logsumexp` runs over the same buffer in the same order.

One hypothesis died on the way and is recorded rather than carried
forward. `cnaster` initializes its two chains with different spot sums --
one over the whole state block, one row at a time -- and floating-point
addition is not associative, so an earlier draft kept both forms to protect
the bitwise claim. Under `numba` the two reductions agree to the bit, so one
form serves both and `test_the_two_spot_sums_agree_bitwise` is what would
catch a release that changed it.
"""

from dataclasses import dataclass

import numpy as np
import pytest

SPOTS = 3
"""More than one, because a single spot makes both spot sums trivial."""


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
    """Ragged segments, a proper transition, and a switch kernel that moves.

    The sitewise probability is drawn rather than held constant: a constant
    one would make the phased transition site-independent, and the phased
    half of the claim would hold for the wrong reason.
    """
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
    """All four of `cnaster`'s recursions, from one kernel, to the last bit.

    Bitwise rather than to a tolerance, and that is the whole claim: a
    tolerance would leave open whether the unified form reassociated
    something, which is the one thing a recursion collapsing four cases must
    not do.
    """
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
    """`is_phased` reads the chain off the emission, and refuses the rest.

    `cnaster` recovers `n_states` from the emission by halving it, which
    cannot express an unphased chain on an even number of states. Passing
    `n_states` instead makes the two cases distinguishable, and a state axis
    that is neither is a caller error rather than a silent halving.
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
    """Why the unified recursion needs one initialization and not two.

    `cnaster` initializes its two chains differently: `hmm_nophasing` sums
    the spot axis with `np.sum(..., axis=1)` over the whole state block,
    `hmm_phased` sums one state's row at a time. Floating-point addition is
    not associative, so an earlier draft of `port.patch.lattice` kept both
    forms rather than risk the bitwise claim on a reassociation.

    It did not need to. Under `numba` the two reductions agree to the bit,
    on the contiguous block the unphased chain hands them and on the strided
    view the phased one does. The hypothesis is recorded here rather than
    carried forward, and this test is what would catch a `numba` release
    that changed it -- which would break the four bitwise claims above
    without touching `port`.

    `smoke` because the referee is the implementation itself: two of
    `numba`'s reductions agreeing says nothing about whether either is the
    sum `cnaster` should be taking.
    """
    from port.patch.lattice import spot_sums_agree

    inputs = _inputs(n_states, phased=True)

    assert spot_sums_agree(inputs.log_emission[:, 0, :])
    assert spot_sums_agree(np.ascontiguousarray(inputs.log_emission[:, 0, :]))


@pytest.mark.smoke
def test_the_backward_pass_does_not_read_the_start_probability() -> None:
    """`log_startprob` is in the signature and out of the recursion.

    `cnaster` takes it in both passes and reads it in one. Kept so the two
    are interchangeable at a call site, and pinned here so a reader does not
    have to infer it from the body -- and so a future edit that started
    reading it would fail rather than change a number quietly.
    """
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
