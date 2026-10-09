"""`cnaster`'s emission parameters as a `sal.emissions.CountPairEmission` (#232).

Exact map: `r = 1 / alpha`, `mean = exposure * exp(log_mu)`, `a = p * tau`,
`b = (1 - p) * tau`. Holds only when `base_nb_mean` and `total_bb_RD` are
constant across bins (#57, #65); otherwise raises. Referee:
`tests/test_emission_mixture_oracle.py`.
"""

from __future__ import annotations

import numpy as np
from sal.emissions import CountPairEmission

from port.patch.emission import DISPERSION_FLOOR

__all__ = [
    "CovariateNotConstant",
    "constant_covariate",
    "count_pair_family",
]


class CovariateNotConstant(ValueError):
    """The per-bin exposure or trial count varies, so upstream has no form."""


def constant_covariate(values: np.ndarray, name: str, *, rtol: float = 1e-12) -> float:
    """The single finite value `values` takes; `rtol` absorbs float64 noise only."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        msg = f"{name} has no finite entries"
        raise CovariateNotConstant(msg)

    low, high = float(finite.min()), float(finite.max())
    scale = max(abs(low), abs(high), 1.0)

    if (high - low) / scale > rtol:
        msg = (
            f"{name} varies over [{low:.6g}, {high:.6g}]; upstream's family "
            "carries one value per state and cnaster carries one per bin, so "
            "there is no exact correspondence here (#57, #65)"
        )
        raise CovariateNotConstant(msg)

    return float(finite[0])


def count_pair_family(
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    *,
    exposure: np.ndarray | float,
    trials: np.ndarray | float,
) -> CountPairEmission:
    """`cnaster`'s emission parameters as an independent-form upstream family.

    Parameters are per state; a `(n_states, n_spots)` array must have one spot.
    """
    parameters = [
        np.asarray(p, dtype=np.float64) for p in (log_mu, alphas, p_binom, taus)
    ]

    flattened = []

    for name, values in zip(
        ("log_mu", "alphas", "p_binom", "taus"), parameters, strict=True
    ):
        if values.ndim == 2 and values.shape[1] != 1:
            msg = f"{name} carries {values.shape[1]} spots; one state needs one value"
            raise ValueError(msg)

        flattened.append(values[:, 0] if values.ndim == 2 else values)

    log_mu_k, alphas_k, p_binom_k, taus_k = flattened
    n_states = log_mu_k.size

    exposure_value = constant_covariate(np.asarray(exposure), "base_nb_mean")
    trials_value = constant_covariate(np.asarray(trials), "total_bb_RD")

    return CountPairEmission(
        # NB r = 1 / alpha with `hmm_nophasing._nb_logpmf_1d`'s floor.
        dispersion=1.0 / np.maximum(alphas_k, DISPERSION_FLOOR),
        mean=exposure_value * np.exp(log_mu_k),
        alpha=p_binom_k * taus_k,
        beta=(1.0 - p_binom_k) * taus_k,
        trials=np.full(n_states, trials_value),
        joint=False,
    )
