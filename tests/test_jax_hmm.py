"""`port.qa.jax_hmm` objective against `cnaster`'s own kernels, to stated tolerances (#287)."""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d, hmm_nophasing
from port.patch.hmm_nophasing import shifts
from port.qa.jax_hmm import emission, marginal_negative_log_likelihood, shifted_rates
from scipy.special import logsumexp


def _instance(
    n_states: int = 4, n_obs: int = 60, seed: int = 31
) -> dict[str, np.ndarray]:
    """One clone-stacked sequence, with the parameters at the fit's shape, in `emission`'s order."""
    generator = np.random.default_rng(seed)

    exposure = generator.integers(20, 120, n_obs).astype(np.float64)
    trials = generator.integers(10, 60, n_obs).astype(np.float64)

    return {
        "log_mu": generator.normal(0.0, 0.3, size=(n_states, 1)),
        "alphas": np.full((n_states, 1), 0.18),
        "p_binom": generator.uniform(0.2, 0.8, size=(n_states, 1)),
        "taus": np.full((n_states, 1), 22.0),
        "counts_nb": generator.poisson(exposure).astype(np.float64),
        "base_nb_mean": exposure,
        "counts_bb": generator.binomial(trials.astype(int), 0.42).astype(np.float64),
        "total_bb_RD": trials,
    }


def _cnaster_emission(instance: dict[str, np.ndarray]) -> np.ndarray:
    """Per-state `_nb_logpmf_1d` plus `_bb_logpmf_1d`, `(n_states, n_obs)`."""
    n_states, n_obs = instance["log_mu"].shape[0], instance["counts_nb"].size
    theirs = np.zeros((n_states, n_obs))

    for state in range(n_states):
        read_depth, allele = np.zeros(n_obs), np.zeros(n_obs)
        _nb_logpmf_1d(
            instance["counts_nb"],
            instance["base_nb_mean"],
            float(np.exp(instance["log_mu"][state, 0])),
            float(instance["alphas"][state, 0]),
            read_depth,
        )
        _bb_logpmf_1d(
            instance["counts_bb"],
            instance["total_bb_RD"],
            float(instance["p_binom"][state, 0]),
            float(instance["taus"][state, 0]),
            allele,
        )
        theirs[state] = read_depth + allele

    return theirs


@pytest.mark.oracle
def test_the_jax_emission_is_cnasters() -> None:
    """Both channels match `_nb_logpmf_1d` and `_bb_logpmf_1d` within 1e-10 absolute."""

    instance = _instance()
    theirs = _cnaster_emission(instance)
    ours = np.asarray(emission(*instance.values()))

    assert ours.shape == theirs.shape

    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-10)


@pytest.mark.oracle
def test_the_jax_forward_is_cnasters() -> None:
    """Marginal likelihood matches `hmm_nophasing.forward_lattice` within 1e-9 relative."""

    instance = _instance()
    n_states = instance["log_mu"].shape[0]
    n_obs = instance["counts_nb"].size
    lengths = np.array([n_obs // 2, n_obs - n_obs // 2], dtype=np.int64)

    dense = _cnaster_emission(instance)

    generator = np.random.default_rng(3)
    log_startprob = np.log(np.full(n_states, 1.0 / n_states))

    raw = generator.uniform(0.5, 1.5, size=(n_states, n_states)) + 4.0 * np.eye(
        n_states
    )
    log_transmat = np.log(raw / raw.sum(axis=1, keepdims=True))

    lattice = hmm_nophasing.forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        dense[:, :, None],
        np.zeros((n_obs, 2)),
    )

    ends = np.cumsum(lengths) - 1
    theirs = -float(np.sum(logsumexp(np.asarray(lattice)[:, ends], axis=0)))

    scores = emission(*instance.values())

    ours = float(
        marginal_negative_log_likelihood(scores, log_startprob, log_transmat, lengths)
    )

    assert np.isfinite(theirs), "upstream's lattice did not produce a normalizer"

    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=0.0)


@pytest.mark.oracle
def test_the_jax_shift_is_the_patched_one() -> None:
    """`shifted_rates` matches `port.patch.hmm_nophasing.shifts` within 1e-12."""

    generator = np.random.default_rng(13)
    n_states, lengths = 4, np.array([9, 5, 11], dtype=np.int64)
    n_segments = int(lengths.sum())

    log_mu = generator.normal(0.0, 0.4, size=(n_states, 1))
    copy_states = generator.integers(0, n_states, size=n_segments).astype(np.int64)
    normal_log_lambda = np.log(
        generator.random(n_segments) / n_segments,
    )

    theirs = shifts(log_mu[:, 0], copy_states, normal_log_lambda, lengths)
    ours = np.asarray(shifted_rates(log_mu, copy_states, normal_log_lambda, lengths))

    assert ours.shape == (lengths.size, n_states), (
        f"one debiased vector per clone, got {ours.shape}"
    )

    np.testing.assert_allclose(
        ours, log_mu[:, 0][None, :] - theirs[:, None], rtol=0.0, atol=1e-12
    )


@pytest.mark.analytic
def test_the_shift_removes_the_overall_scale() -> None:
    """Adding a constant to every rate leaves the debiased rates unchanged (model invariant)."""

    generator = np.random.default_rng(23)
    n_states, lengths = 3, np.array([7, 7], dtype=np.int64)
    n_segments = int(lengths.sum())

    log_mu = generator.normal(0.0, 0.4, size=n_states)
    copy_states = generator.integers(0, n_states, size=n_segments).astype(np.int64)
    normal_log_lambda = np.log(generator.random(n_segments) / n_segments)

    base = np.asarray(shifted_rates(log_mu, copy_states, normal_log_lambda, lengths))

    for offset in (-1.5, 0.75, 4.0):
        moved = np.asarray(
            shifted_rates(log_mu + offset, copy_states, normal_log_lambda, lengths)
        )

        np.testing.assert_allclose(moved, base, rtol=0.0, atol=1e-12)
