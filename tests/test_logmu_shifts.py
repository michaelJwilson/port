"""`compute_logmu_shifts`, against its own identities.

The quantity issue #5 is about: a per-clone normalizer that `cnaster`
computes and never applies. It is a pure function of its arguments, so it
can be pinned now and the pin stands when the shift is finally used.

Agreement with the vectorized form is `tests/test_logmu_shift.py`'s claim,
against `port.patch.hmm_nophasing.logmu_shift.shifts`.
"""

import numpy as np
import pytest


def draw(seed: int, n_states: int, clone_lengths: list[int]) -> tuple[np.ndarray, ...]:
    """Arguments for one call, seeded."""
    rng = np.random.default_rng(seed)
    n_segments = int(np.sum(clone_lengths))
    log_mus = rng.normal(size=n_states)
    copy_states = rng.integers(0, n_states, size=n_segments)
    weights = rng.random(n_segments)
    return (
        log_mus,
        copy_states,
        np.log(weights / weights.sum()),
        np.array(clone_lengths),
    )


@pytest.mark.smoke
def test_is_constant_within_a_clone() -> None:
    """One shift per clone, broadcast over its positions.

    The property the emission depends on: the shift scales a clone's whole
    profile, so a value varying inside a clone would be a different model,
    not a different number.
    """
    from cnaster.hmm_nophasing import compute_logmu_shifts

    clone_lengths = [6, 9, 5]
    shifts = compute_logmu_shifts(
        *draw(seed=2, n_states=3, clone_lengths=clone_lengths)
    )

    start = 0
    for length in clone_lengths:
        within = shifts[start : start + length]
        np.testing.assert_array_equal(within, np.full(length, within[0]))
        start += length


@pytest.mark.analytic
@pytest.mark.parametrize("offset", [-2.0, 0.5, 3.0])
def test_shifts_with_log_mu(offset: float) -> None:
    """Scaling every mean by a constant moves the shift by its logarithm.

    This is why the quantity removes a scale rather than changing a shape:
    `mu -> c mu` sends the normalizer to `c` times itself and the debiased
    `log_mu - shift` is unchanged, which is the invariant #5 asserts of the
    fitted parameters.
    """
    from cnaster.hmm_nophasing import compute_logmu_shifts

    log_mus, copy_states, normal_log_lambda, clone_lengths = draw(
        seed=9, n_states=3, clone_lengths=[8, 8]
    )
    base = compute_logmu_shifts(log_mus, copy_states, normal_log_lambda, clone_lengths)
    moved = compute_logmu_shifts(
        log_mus + offset, copy_states, normal_log_lambda, clone_lengths
    )

    np.testing.assert_allclose(moved, base + offset, rtol=0.0, atol=1e-12)


@pytest.mark.analytic
def test_normalised_weights_and_one_state_give_that_state() -> None:
    """With one state and weights summing to one, the shift is that state's mean.

    A case whose answer is known without computing it: every term is the
    same `log_mu`, and `logsumexp` over weights that sum to one returns it.
    """
    from cnaster.hmm_nophasing import compute_logmu_shifts

    n_segments = 12
    log_mus = np.array([1.75])
    copy_states = np.zeros(n_segments, dtype=int)
    weights = np.full(n_segments, 1.0 / n_segments)

    shifts = compute_logmu_shifts(
        log_mus, copy_states, np.log(weights), np.array([n_segments])
    )

    np.testing.assert_allclose(shifts, np.full(n_segments, 1.75), rtol=0.0, atol=1e-12)
