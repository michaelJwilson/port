"""The phased HMM's posteriors from sal's Kronecker-switch ragged kernel (#426, sal #1133).

Referee: `cnaster.hmm_phased`'s forward and backward lattices normalized as
`get_state_posteriors` does, to 1e-9. Not on the run's path (#318).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp


def _instance(
    n_states: int, lengths: tuple[int, ...], spots: int, seed: int = 0
) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    n_obs = int(np.sum(lengths))
    transition = 0.9 * np.eye(n_states) + 0.1 * rng.dirichlet(
        np.ones(n_states), n_states
    )
    return (
        np.asarray(lengths),
        np.log(transition),
        np.log(np.full(n_states, 1 / n_states)),
        rng.normal(-3, 2, (2 * n_states, n_obs, spots)),
        np.log(rng.uniform(1e-3, 0.3, n_obs)),
    )


@pytest.mark.oracle
@pytest.mark.cnaster
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize(
    "shape", [(5, (30, 50, 20), 2), (4, (1, 17, 3), 1)], ids=["gate", "short"]
)
def test_the_kronecker_posteriors_are_cnasters_lattices(
    diagonal: bool, shape: tuple[int, tuple[int, ...], int]
) -> None:
    """Both variants, including a one-position segment, to 1e-9."""
    import cnaster.hmm_phased as phased
    from port.qa.kronecker_posteriors import kronecker_state_posteriors

    arguments = _instance(*shape)
    model = phased.hmm_phased()
    forward = model.forward_lattice(*arguments, diagonal)
    backward = model.backward_lattice(*arguments, diagonal)
    theirs = forward + backward
    theirs = theirs - logsumexp(theirs, axis=0)

    ours = kronecker_state_posteriors(*arguments, diagonal=diagonal)

    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9)
    np.testing.assert_allclose(logsumexp(ours, axis=0), 0.0, atol=1e-12)


@pytest.mark.patch
@pytest.mark.parametrize("n_states", range(1, 10))
def test_kronecker_order_is_the_reorder_it_replaces(n_states: int) -> None:
    """sal #1144's `kronecker_order` against the index port built (T- #632): a permutation, so bitwise."""
    from sal.likelihood.ragged import kronecker_order

    ours = (2 * np.arange(n_states)[None, :] + np.arange(2)[:, None]).reshape(-1)
    order = kronecker_order(2 * n_states, layer_major=True)

    assert np.array_equal(np.argsort(order), ours)
    assert np.array_equal(order[ours], np.arange(2 * n_states))
