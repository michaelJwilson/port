"""The per-clone library shift (#392 stage 3): `log Z_c = log sum_g lambda_g
mu_{s_c(g)}`, against brute force, the pin's loop, and its identities."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.special
from sim.truth import Truth, planted

from cnamaste.hmm_nophasing import compute_logmu_shifts, neutral_state, shifts

TOLERANCE = 1e-12
"""Absolute, in log units: one `logsumexp` over at most 240 terms."""


def _problem(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(log_mu, states, log_lambda, lengths)`: three clones of unequal length."""
    rng = np.random.default_rng(seed)
    lengths = np.array([40, 25, 60])
    log_mu = rng.normal(0.0, 0.5, 5)
    states = rng.integers(0, 5, lengths.sum())
    log_lambda = np.concatenate([np.log(rng.dirichlet(np.ones(n))) for n in lengths])
    return log_mu, states, log_lambda, lengths


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_clones_shift_is_the_logsumexp_over_its_own_segments(seed: int) -> None:
    log_mu, states, log_lambda, lengths = _problem(seed)
    bounds = np.concatenate([[0], np.cumsum(lengths)])

    expected = [
        scipy.special.logsumexp(log_mu[states[a:b]] + log_lambda[a:b])
        for a, b in itertools.pairwise(bounds)
    ]
    np.testing.assert_allclose(
        shifts(log_mu, states, log_lambda, lengths), expected, rtol=0, atol=TOLERANCE
    )


@pytest.mark.oracle
def test_the_shifts_are_the_pins_loop_one_value_per_clone() -> None:
    """`compute_logmu_shifts` writes each clone's value over its segments."""
    log_mu, states, log_lambda, lengths = _problem(3)
    per_segment = compute_logmu_shifts(log_mu, states, log_lambda, lengths)
    np.testing.assert_allclose(
        np.repeat(shifts(log_mu, states, log_lambda, lengths), lengths),
        per_segment,
        rtol=0,
        atol=TOLERANCE,
    )


@pytest.mark.analytic
def test_a_constant_on_every_rate_moves_every_shift_by_that_constant() -> None:
    """`mu -> e^k mu` sends `Z_c -> e^k Z_c`: the shifted rate `mu / Z_c` has no scale."""
    log_mu, states, log_lambda, lengths = _problem(4)
    base = shifts(log_mu, states, log_lambda, lengths)
    moved = shifts(log_mu + 1.7, states, log_lambda, lengths)
    np.testing.assert_allclose(moved - base, 1.7, rtol=0, atol=TOLERANCE)


@pytest.mark.analytic
def test_shifted_rates_hold_each_clones_expected_library_at_one() -> None:
    """`sum_g lambda_g mu_{s_c(g)} / Z_c = 1` when each clone's `lambda` sums to one."""
    log_mu, states, log_lambda, lengths = _problem(5)
    log_z = np.repeat(shifts(log_mu, states, log_lambda, lengths), lengths)
    terms = np.exp(log_mu[states] - log_z + log_lambda)
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    totals = np.add.reduceat(terms, bounds[:-1])
    np.testing.assert_allclose(totals, 1.0, rtol=1e-12)


@pytest.fixture(scope="module")
def truth() -> Truth:
    return planted(n_clones=3, n_states=4, lattice=(12, 10), n_obs=60, n_segments=3)


@pytest.mark.end2end
def test_the_pinned_state_is_the_normal_clones_neutral_state(truth: Truth) -> None:
    """Planted state 0, under any relabelling of the states."""
    assert neutral_state(truth.log_mu, truth.p_binom, truth.states.T) == 0

    order = np.array([2, 0, 3, 1])
    relabelled = np.argsort(order)[truth.states.T]
    assert neutral_state(truth.log_mu[order], truth.p_binom[order], relabelled) == 1


@pytest.mark.analytic
def test_without_a_path_the_pinned_state_is_the_lowest_balanced_rate() -> None:
    log_mu = np.log(np.array([2.0, 0.8, 1.0, 0.5]))
    p_binom = np.array([0.5, 0.52, 0.49, 0.9])
    assert neutral_state(log_mu, p_binom) == 1
