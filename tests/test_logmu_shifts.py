"""`compute_logmu_shifts`, against a vectorized reference and its own identities.

The quantity issue #5 is about: a per-clone normalizer that `cnaster`
computes and never applies. It is a pure function of its arguments, so it
can be pinned now and the pin stands when the shift is finally used.

The reference below is the vectorized form of the same definition, which is
also the form the docstring inside `compute_logmu_shifts` sketches in a
comment block that was never adopted.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

SELF_TRANSITION_STATES = 4


def reference_shifts(
    log_mus: np.ndarray,
    copy_states: np.ndarray,
    normal_log_lambda: np.ndarray,
    clone_lengths: np.ndarray,
) -> np.ndarray:
    """Per clone, the log of `sum_b lambda_b mu_state(b)`: **one value each**.

    `cnaster`'s `port` branch returns `(n_clones,)`; the branch before it
    broadcast the same value over each clone's segments and returned
    `(n_segments,)`. The quantity is unchanged -- a clone's normalizer is one
    number either way -- and #259 stage 1 moved the pin, so this reference
    moved with it.
    """
    shifts = np.empty(len(clone_lengths), dtype=np.float64)
    start = 0
    for clone, length in enumerate(clone_lengths):
        stop = start + length
        terms = log_mus[copy_states[start:stop]] + normal_log_lambda[start:stop]
        shifts[clone] = logsumexp(terms)
        start = stop
    return shifts


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


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize(
    "clone_lengths", [[10], [10, 10], [1, 19], [7, 3, 10]], ids=str
)
@pytest.mark.parametrize("n_states", [1, SELF_TRANSITION_STATES])
def test_matches_the_vectorized_reference(
    clone_lengths: list[int], n_states: int
) -> None:
    """The loop computes what the closed form computes."""
    from cnaster.hmm_nophasing import compute_logmu_shifts

    args = draw(seed=5, n_states=n_states, clone_lengths=clone_lengths)
    np.testing.assert_allclose(
        compute_logmu_shifts(*args), reference_shifts(*args), rtol=0.0, atol=1e-12
    )


@pytest.mark.smoke
def test_there_is_one_shift_per_clone() -> None:
    """One value per clone, and the rank says which model this is.

    The property the emission depends on: the shift scales a clone's whole
    profile, so a value varying *inside* a clone would be a different model.
    The `port` branch expresses that by returning one number per clone rather
    than by broadcasting it over the clone's segments, and the rank is what
    #258 and #259 settle on -- `(n_clones,)`, not `(n_states, n_clones)` and
    not `(n_segments,)`.
    """
    from cnaster.hmm_nophasing import compute_logmu_shifts

    clone_lengths = [6, 9, 5]
    shifts = compute_logmu_shifts(
        *draw(seed=2, n_states=3, clone_lengths=clone_lengths)
    )

    assert shifts.shape == (len(clone_lengths),)
    assert np.all(np.isfinite(shifts))


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

    np.testing.assert_allclose(shifts, np.full(1, 1.75), rtol=0.0, atol=1e-12)
