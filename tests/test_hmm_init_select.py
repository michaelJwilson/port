"""Initializer backends scored by one referee, and the selection over them (#229 stages 3, 4; #230).

`patch` tests check agreement; the sal recovery tests are `oracle` against planted
states.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from port.qa.emission_family import count_pair_family
from sal.emissions import CountPairEmission

EXPOSURE, TRIALS = 40.0, 60.0
N_OBS = 400


@pytest.fixture
def drawn() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts from one known state, at constant exposure and trials."""
    rng = np.random.default_rng(3)
    totals = rng.poisson(EXPOSURE * np.exp(0.0), size=N_OBS).astype(float)
    successes = rng.binomial(int(TRIALS), 0.32, size=N_OBS).astype(float)

    X = np.stack([totals, successes], axis=1).reshape(N_OBS, 2, 1)

    return X, np.full(N_OBS, EXPOSURE), np.full(N_OBS, TRIALS)


@pytest.mark.patch
def test_the_parameter_map_round_trips(
    drawn: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """`count_pair_family` and the backend's inverse round-trip the parameters."""
    log_mu = np.array([-0.3, 0.0, 0.45])
    alphas = np.array([0.08, 0.12, 0.05])
    p_binom = np.array([0.50, 0.32, 0.18])
    taus = np.array([30.0, 22.0, 45.0])

    family = count_pair_family(
        log_mu, alphas, p_binom, taus, exposure=EXPOSURE, trials=TRIALS
    )

    back_log_mu = np.log(np.asarray(family.total.mean).reshape(-1) / EXPOSURE)
    back_alphas = 1.0 / np.asarray(family.total.dispersion).reshape(-1)
    back_p = np.asarray(family.rate).reshape(-1)
    back_taus = np.asarray(family.concentration).reshape(-1)

    assert np.allclose(back_log_mu, log_mu, rtol=0, atol=1e-12)
    assert np.allclose(back_alphas, alphas, rtol=0, atol=1e-12)
    assert np.allclose(back_p, p_binom, rtol=0, atol=1e-12)
    assert np.allclose(back_taus, taus, rtol=0, atol=1e-12)


@pytest.mark.oracle
def test_sals_density_with_a_covariate_is_cnasters() -> None:
    """sal's covariate density equals `cnaster`'s NB and BB bin for bin, within 1e-10 (sal #1083)."""

    rng = np.random.default_rng(0)
    n = 400
    log_mu = np.array([-0.3, 0.0, 0.5])
    alphas = np.array([0.05, 0.1, 0.2])
    p_binom = np.array([0.5, 0.2, 0.35])
    taus = np.array([30.0, 40.0, 50.0])
    exposure = rng.uniform(20, 200, n)
    trials = rng.integers(0, 60, n).astype(float)
    totals = rng.poisson(exposure).astype(float)
    successes = np.minimum(rng.poisson(trials * 0.4), trials).astype(float)

    family = CountPairEmission(
        dispersion=1 / alphas,
        mean=np.exp(log_mu),
        alpha=p_binom * taus,
        beta=(1 - p_binom) * taus,
        trials=np.full(3, 10.0),
        joint=False,
    )
    ours = np.asarray(
        family.log_density(
            torch.as_tensor(np.column_stack([totals, successes])),
            covariate=torch.as_tensor(np.column_stack([exposure, trials])),
        )
    ).T

    theirs = np.zeros((3, n))
    for state in range(3):
        nb, bb = np.zeros(n), np.zeros(n)
        _nb_logpmf_1d(totals, exposure, float(np.exp(log_mu[state])), alphas[state], nb)
        _bb_logpmf_1d(successes, trials, p_binom[state], taus[state], bb)
        theirs[state] = nb + bb

    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-10)
