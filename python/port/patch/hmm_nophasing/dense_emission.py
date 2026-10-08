"""`cnaster`'s coded NB/BB emission, scored by sal's coded log-emission (#425, sal #1340).

`_nb_logpmf_1d` and `_bb_logpmf_1d` score one state over a vector of counts,
`lgamma` per score. `sal.emissions.coded.log_emission(family, Dense(...))`
scores every state of the family at once in Rust, from tables over the
distinct counts only (port T- #719: 154,549 rows to 7,109), each completed
in the family's own order (sal #1334, #1336). The same edge behaviour as
`cnaster`'s:

- a zero exposure or trial count scores 0, which sal reads as unobserved;
- `alpha` floored at 1e-10 for the negative binomial, and the beta-binomial's
  `a`, `b` floored at `DISPERSION_FLOOR` (`port.patch.emission`);
- a non-positive rate `mu` scores 0 everywhere, as `exposure * mu <= 0` does.

The tables are sal's `scaled_rising_table`, the one evaluation every site
scores (T- #776). A state at `tau = inf` is the binomial at `p`, which sal's
rate-concentration family scores (sal #1340); the rest are sal's `(a, b)`
family at port's floors.

To a tolerance, not bitwise, so it is `--sal`'s rather than a `SWAPS` row;
#244 is why a tolerance is measured end to end before it is anything else.
Selected by the `hmm_nophasing` row's `emission_kernels="sal"` option, not a
name rebind: `cnaster`'s compiled kernels call `_nb_logpmf_1d` as a global.
:func:`coded_emission` is upstream's coded method with every state scored in
one call per spot, which is where the speed is.
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
    """`(K, n)`: every state's `_nb_logpmf_1d` in one call; a rate <= 0 scores 0.

    `sal.emissions.coded.log_emission` on sal's negative binomial at
    `r = nb_size(alpha)` (`inf`, the Poisson, at `alpha <= 0`).
    """
    from sal.emissions.coded import Dense, log_emission
    from sal.emissions.counts import NegativeBinomialEmission

    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    covariate = np.asarray(exposure, dtype=np.float64).reshape(-1)
    if not covariate.any():
        # NB sal's coded route raises where no observation is observed
        #    (`count_log_factor` reshapes an empty table); each scores 0.
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
    """`(K, n)`: every state's `_bb_logpmf_1d` in one call.

    `sal.emissions.coded.log_emission` on sal's beta-binomial at
    `a = max(p tau, floor)`, `b = max((1 - p) tau, floor)`; a state at
    `tau = inf` on sal's rate-concentration family, the binomial at `p`.
    """
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
    # NB sal reads each observation's trial count from the covariate; the
    #    family's per-state `trials` is a placeholder it requires.
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
    """`hmm_nophasing.compute_emission_probability_nb_betabinom_coded`, unshifted.

    Upstream's loop over spots, decode and stack, with each spot's states
    scored together: one NB and one BB family of `n_states` per spot.
    """
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
