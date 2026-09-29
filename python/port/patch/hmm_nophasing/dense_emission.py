"""`cnaster`'s coded NB/BB emission, scored by sal's dense log-emission (#425).

`_nb_logpmf_1d` and `_bb_logpmf_1d` score one state over a vector of counts,
`lgamma` per score. `sal.emissions.dense.log_emission` scores every state of
the same family at once from sal's tables in Rust (sal #1132), in the
family's own completion order (`Order.FAMILY`). The same edge behaviour as
`cnaster`'s:

- a zero exposure or trial count scores 0, which sal reads as unobserved;
- `alpha` floored at 1e-10 for the negative binomial, and the beta-binomial's
  `alpha`, `beta` floored at `EPS`;
- a non-positive rate `mu` scores 0 everywhere, as `exposure * mu <= 0` does.

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

from port.patch.hmm_nophasing.gradient import DISPERSION_FLOOR

__all__ = ["bb_states", "coded_emission", "nb_states"]


def nb_states(
    obs: np.ndarray, exposure: np.ndarray, mu: np.ndarray, dispersions: np.ndarray
) -> np.ndarray:
    """`(K, n)`: every state's `_nb_logpmf_1d` in one call; a rate <= 0 scores 0."""
    from sal.emissions import NegativeBinomialEmission
    from sal.emissions.dense import Order, log_emission

    mu = np.asarray(mu, dtype=np.float64)
    dead = mu <= 0.0
    family = NegativeBinomialEmission(
        dispersion=1.0
        / np.maximum(np.asarray(dispersions, dtype=np.float64), DISPERSION_FLOOR),
        mean=np.where(dead, 1.0, mu),
    )
    scores = log_emission(
        family,
        np.asarray(obs, dtype=np.float64),
        np.asarray(exposure, dtype=np.float64)[:, None],
        order=Order.FAMILY,
    )
    scores[dead] = 0.0
    return scores


def bb_states(
    obs: np.ndarray, trials: np.ndarray, p_binom: np.ndarray, taus: np.ndarray
) -> np.ndarray:
    """`(K, n)`: every state's `_bb_logpmf_1d` in one call."""
    from sal.emissions import BetaBinomialEmission
    from sal.emissions.dense import Order, log_emission

    p = np.asarray(p_binom, dtype=np.float64)
    t = np.asarray(taus, dtype=np.float64)
    family = BetaBinomialEmission(
        alpha=np.maximum(p * t, DISPERSION_FLOOR),
        beta=np.maximum((1.0 - p) * t, DISPERSION_FLOOR),
        trials=np.ones_like(p),
    )
    return log_emission(
        family,
        np.asarray(obs, dtype=np.float64),
        np.asarray(trials, dtype=np.float64)[:, None],
        order=Order.FAMILY,
    )


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
