"""The transition matrix and the forward/backward lattices, against brute-force
enumeration of every path and against the identities any HMM satisfies."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest
import scipy.special

from cnamaste.hmm_nophasing import get_log_transmat, hmm_nophasing
from cnamaste.hmm_phased import hmm_phased

TOLERANCE = 1e-10
"""Absolute, in nats: a sum of at most a few hundred float64 terms."""


def _problem(
    n_states: int, lengths: list[int], n_spots: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_obs = sum(lengths)
    log_emission = np.log(rng.uniform(0.05, 1.0, (n_states, n_obs, n_spots)))
    log_startprob = np.log(rng.dirichlet(np.ones(n_states)))
    log_transmat = get_log_transmat(n_states, 0.8)
    return np.array(lengths), log_transmat, log_startprob, log_emission


def _brute_force(
    lengths: np.ndarray,
    log_start: np.ndarray,
    log_trans: Callable[[int], np.ndarray],
    log_emit_site: np.ndarray,
) -> tuple[float, np.ndarray]:
    """`log P(obs)` and state marginals by enumerating every path per segment.

    `log_trans(t)` is the transition into site `t` from `t - 1`, so a
    site-dependent chain is enumerated the same way as a constant one.
    """
    n_states, n_obs = log_emit_site.shape
    marginals = np.zeros((n_states, n_obs))
    total = 0.0
    start = 0

    for length in lengths:
        sites = range(start, start + int(length))
        scores: list[float] = []
        paths = list(itertools.product(range(n_states), repeat=int(length)))
        for path in paths:
            w = log_start[path[0]] + log_emit_site[path[0], start]
            for offset in range(1, int(length)):
                t = start + offset
                w += (
                    log_trans(t)[path[offset - 1], path[offset]]
                    + log_emit_site[path[offset], t]
                )
            scores.append(float(w))

        weights = np.array(scores)
        norm = scipy.special.logsumexp(weights)
        total += norm
        posterior = np.exp(weights - norm)
        for path, weight in zip(paths, posterior, strict=True):
            for offset, t in enumerate(sites):
                marginals[path[offset], t] += weight
        start += int(length)

    return total, marginals


@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [1, 2, 5])
def test_transition_rows_are_distributions_with_the_stated_self_transition(
    n_states: int,
) -> None:
    t = 0.97
    log_transmat = get_log_transmat(n_states, t)
    np.testing.assert_allclose(
        scipy.special.logsumexp(log_transmat, axis=1), 0.0, atol=1e-15
    )

    if n_states > 1:
        np.testing.assert_allclose(np.exp(np.diag(log_transmat)), t, rtol=1e-15)
        off = np.exp(log_transmat[~np.eye(n_states, dtype=bool)])
        np.testing.assert_allclose(off, (1 - t) / (n_states - 1), rtol=1e-14)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1])
def test_unphased_lattices_are_brute_force_enumeration(seed: int) -> None:
    lengths, log_transmat, log_startprob, log_emission = _problem(3, [4, 3], 2, seed)
    site = log_emission.sum(axis=2)
    expected, marginals = _brute_force(
        lengths, log_startprob, lambda _: log_transmat, site
    )

    none = np.zeros(site.shape[1])
    log_alpha = hmm_nophasing.forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, none
    )
    log_beta = hmm_nophasing.backward_lattice(
        lengths, log_transmat, log_startprob, log_emission, none
    )

    ends = np.cumsum(lengths) - 1
    forward = sum(scipy.special.logsumexp(log_alpha[:, end]) for end in ends)
    assert abs(forward - expected) < TOLERANCE

    starts = np.concatenate([[0], ends[:-1] + 1])
    backward = sum(
        scipy.special.logsumexp(log_startprob + site[:, s] + log_beta[:, s])
        for s in starts
    )
    assert abs(backward - expected) < TOLERANCE

    log_gamma = hmm_nophasing().get_state_posteriors(
        lengths, log_transmat, log_startprob, log_emission, none
    )
    np.testing.assert_allclose(np.exp(log_gamma), marginals, rtol=0, atol=1e-12)


@pytest.mark.oracle
def test_phased_lattices_are_brute_force_enumeration_over_state_and_phase() -> None:
    """Paired states `(k, h)`: the copy chain times a per-site phase flip.

    `P((k, h) -> (k', h')) = T[k, k'] * (s_t if h != h' else 1 - s_t)`, with
    `s_t` the switch probability of the site transitioned **from**, and a
    uniform start over phase.
    """
    n_states = 2
    lengths, log_transmat, log_startprob, _ = _problem(n_states, [3, 3], 2, 7)
    rng = np.random.default_rng(8)
    n_obs = int(lengths.sum())
    log_emission = np.log(rng.uniform(0.05, 1.0, (2 * n_states, n_obs, 2)))
    switch = rng.uniform(0.01, 0.3, n_obs)
    log_switch = np.log(switch)

    def paired(t: int) -> np.ndarray:
        s = switch[t - 1]
        phase = np.log(np.array([[1 - s, s], [s, 1 - s]]))
        flips: np.ndarray = np.kron(phase, np.ones((n_states, n_states)))
        copies: np.ndarray = np.tile(log_transmat, (2, 2))
        paired: np.ndarray = flips + copies
        return paired

    log_start = np.log(0.5) + np.tile(log_startprob, 2)
    site = log_emission.sum(axis=2)
    expected, _ = _brute_force(lengths, log_start, paired, site)

    log_alpha = hmm_phased.forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_switch
    )
    log_beta = hmm_phased.backward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_switch
    )

    ends = np.cumsum(lengths) - 1
    starts = np.concatenate([[0], ends[:-1] + 1])
    forward = sum(scipy.special.logsumexp(log_alpha[:, end]) for end in ends)
    backward = sum(
        scipy.special.logsumexp(log_start + site[:, s] + log_beta[:, s]) for s in starts
    )

    assert abs(forward - expected) < TOLERANCE
    assert abs(backward - expected) < TOLERANCE


@pytest.mark.analytic
def test_segments_are_independent_chains() -> None:
    """Restarting at each segment: the joint lattice is the segments' lattices, concatenated."""
    lengths, log_transmat, log_startprob, log_emission = _problem(4, [5, 2, 6], 3, 3)
    none = np.zeros(log_emission.shape[1])
    joint = hmm_nophasing.forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, none
    )

    start = 0
    for length in lengths:
        part = log_emission[:, start : start + length]
        alone = hmm_nophasing.forward_lattice(
            np.array([length]), log_transmat, log_startprob, part, none[:length]
        )
        np.testing.assert_array_equal(joint[:, start : start + length], alone)
        start += length
