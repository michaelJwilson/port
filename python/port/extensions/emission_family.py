"""`cnaster`'s emission parameters, as a `snakes_and_ladders` family.

**#232, #229 stage 4.** `cnaster` writes its own negative-binomial and
beta-binomial densities (`hmm_nophasing._nb_logpmf_1d`, `_bb_logpmf_1d`) and
its own mixture initializer on top of them. Upstream ships
`sal.emissions.CountPairEmission` -- "a total count and the
successes within it, as one observation" -- which is the same model, and
`opt/emission_mixture` fits a mixture of it. `CLAUDE.md` asks an optimization
to be considered first as functionality that exists upstream; this module is
the seam that makes that possible.

## The parameter map

`cnaster` and upstream parameterize the same two laws differently, and the
map is exact rather than approximate:

| `cnaster` | upstream | relation |
| --- | --- | --- |
| `alphas[k]` | `dispersion[k]` | `r = 1 / alpha` (`hmm_nophasing.py:48`) |
| `log_mu[k]`, `base_nb_mean[i]` | `mean[k]` | `lambda = exposure * exp(log_mu)` |
| `p_binom[k]`, `taus[k]` | `alpha[k]`, `beta[k]` | `a = p * tau`, `b = (1 - p) * tau` |
| `total_bb_RD[i]` | `trials[k]` | equal only when constant |

Measured: 1.4e-13 maximum absolute difference in log-density over 600 values,
which is float64 round-off from a different summation order rather than a
modelling difference. `tests/test_emission_mixture_oracle.py` is the referee.

## The regime, and why it is not the whole problem

**The last row is the catch.** `cnaster` carries an exposure per *bin* and a
trial count per *bin*; `CountPairEmission` carries a mean per *state* and, in
its independent form, a trial count per *state*. So the correspondence holds
exactly when `base_nb_mean` and `total_bb_RD` are constant across bins, and
not otherwise.

That is #57 and #65 -- "upstream's covariate stops at the emission family" --
arriving at the initializer. It is a difference in the problem rather than a
defect on either side, stated here because `CLAUDE.md` requires a difference
in regime to be named where the work is. `constant_covariate` is what refuses
to pretend otherwise: outside the regime this raises rather than silently
fitting a model the data did not come from.

The joint form does not rescue it. There the beta-binomial's trials *are* the
drawn total, and `cnaster`'s `total_bb_RD` is SNP-covering depth while
`X[:, 0]` is total depth -- two different quantities, so coupling them would
be a third model rather than either of these.
"""

from __future__ import annotations

import numpy as np
from sal.emissions import CountPairEmission

__all__ = [
    "CovariateNotConstant",
    "constant_covariate",
    "count_pair_family",
]


class CovariateNotConstant(ValueError):
    """The per-bin exposure or trial count varies, so upstream has no form.

    Raised rather than approximated. A family fitted to a mean that ignores a
    varying exposure is a fit to a different model, and reporting it beside
    `cnaster`'s would be comparing two answers to two questions.
    """


def constant_covariate(values: np.ndarray, name: str, *, rtol: float = 1e-12) -> float:
    """The single value `values` takes, or a refusal naming the spread.

    `rtol` is not a tolerance on the science: it absorbs the float64 noise a
    covariate picks up from being computed rather than declared. A covariate
    that genuinely varies fails this by orders of magnitude, not by ulps.
    """
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
    """`cnaster`'s emission parameters as an upstream family.

    The independent form, because `cnaster`'s two channels are independent
    given the state: its beta-binomial's trials are `total_bb_RD`, which is
    not the negative binomial's drawn total.

    Parameters are taken per state, so a `(n_states, n_spots)` array is
    accepted only when it carries one spot -- the shape `clone_stack_obs`
    produces, and the only one where a state has a single parameter.
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
        # NB r = 1 / alpha, which `hmm_nophasing._nb_logpmf_1d:48` computes
        #    with the same floor. Reproduced rather than chosen.
        dispersion=1.0 / np.maximum(alphas_k, 1.0e-10),
        mean=exposure_value * np.exp(log_mu_k),
        alpha=p_binom_k * taus_k,
        beta=(1.0 - p_binom_k) * taus_k,
        trials=np.full(n_states, trials_value),
        joint=False,
    )
