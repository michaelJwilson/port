"""`cnaster`'s coded NB/BB emission, scored by sal's coded log-emission (#425, sal #1340).

All states at once in Rust over distinct counts (T- #776), with `cnaster`'s
edge behaviour: zero exposure scores 0, dispersions floored, a rate `mu <= 0`
scores 0, `tau = inf` is the binomial. To a tolerance, not bitwise, so selected
by `hmm_nophasing`'s `emission_kernels="sal"` rather than a `SWAPS` row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from port.patch.emission import DISPERSION_FLOOR, nb_size

__all__ = ["bb_states", "coded_emission", "nb_states"]


def nb_states(
    obs: np.ndarray,
    exposure: np.ndarray,
    mu: np.ndarray,
    dispersions: np.ndarray,
) -> np.ndarray:
    """`(K, n)`: every state's `_nb_logpmf_1d` in one call at `r = nb_size(alpha)`; a rate <= 0 scores 0."""
    from sal.emissions.coded import Dense, log_emission
    from sal.emissions.counts import NegativeBinomialEmission

    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    covariate = np.asarray(exposure, dtype=np.float64).reshape(-1)
    if not covariate.any():
        # NB sal raises where no observation is observed; each scores 0.
        return np.zeros((mu.size, covariate.size))
    dead = mu <= 0.0
    family = NegativeBinomialEmission(
        nb_size(np.asarray(dispersions, dtype=np.float64).reshape(-1)),
        np.where(dead, 1.0, mu),
    )
    scores = log_emission(
        family,
        Dense(np.asarray(obs), covariate),
    )
    scores[dead] = 0.0
    return scores


def bb_states(
    obs: np.ndarray,
    trials: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """`(K, n)`: every state's `_bb_logpmf_1d` in one call; `a`, `b` floored, `tau = inf` the binomial at `p`."""
    from sal.emissions.coded import Dense, log_emission
    from sal.emissions.counts import (
        BetaBinomialEmission,
        RateConcentrationBetaBinomialEmission,
    )

    p = np.asarray(p_binom, dtype=np.float64).reshape(-1)
    covariate = np.asarray(trials, dtype=np.float64).reshape(-1)
    if not covariate.any():
        return np.zeros((p.size, covariate.size))  # NB as :func:`nb_states`
    tau = np.broadcast_to(np.asarray(taus, dtype=np.float64).reshape(-1), p.shape)
    observations = Dense(np.asarray(obs), covariate)
    limit = np.isinf(tau)
    out = np.empty((p.size, observations.counts.size))
    # NB sal reads trials from the covariate; the family's `trials` is a placeholder.
    if not limit.all():
        finite = ~limit
        a = np.maximum(p[finite] * tau[finite], DISPERSION_FLOOR)
        b = np.maximum((1.0 - p[finite]) * tau[finite], DISPERSION_FLOOR)
        out[finite] = log_emission(
            BetaBinomialEmission(np.ones(a.size), a, b), observations
        )
    if limit.any():
        out[limit] = log_emission(
            RateConcentrationBetaBinomialEmission(
                np.ones(int(limit.sum())), p[limit], tau[limit]
            ),
            observations,
        )
    return out


def coded_emission(
    nbEncoder: Any,
    bbEncoder: Any,
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    *,
    clone_stack: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """`hmm_nophasing.compute_emission_probability_nb_betabinom_coded`, unshifted, all states per spot in one call."""
    n_spots = nbEncoder.n_spots

    if bbEncoder.n_spots != n_spots:  # invariant
        msg = "the encoders must cover the same spots"
        raise AssertionError(msg)

    rdr, baf = [], []

    for spot in range(n_spots):
        mu = np.exp(np.asarray(log_mu, dtype=np.float64)[:, spot])
        alpha = np.asarray(alphas, dtype=np.float64)[:, spot]
        p = np.asarray(p_binom, dtype=np.float64)[:, spot]
        tau = np.asarray(taus, dtype=np.float64)[:, spot]

        nb_endog = np.asarray(nbEncoder.get_unique_obs(spot), dtype=np.float64)
        nb_exposure = np.asarray(nbEncoder.get_unique_total(spot), dtype=np.float64)
        bb_endog = np.asarray(bbEncoder.get_unique_obs(spot), dtype=np.float64)
        bb_trials = np.asarray(bbEncoder.get_unique_total(spot), dtype=np.float64)

        rdr_unique = nb_states(nb_endog, nb_exposure, mu, alpha)
        baf_unique = bb_states(bb_endog, bb_trials, p, tau)

        rdr.append(nbEncoder.decode_array(rdr_unique, spot))
        baf.append(bbEncoder.decode_array(baf_unique, spot))

    if clone_stack:
        return np.concatenate(rdr, axis=1), np.concatenate(baf, axis=1)

    return np.stack(rdr, axis=2), np.stack(baf, axis=2)
