"""Integer copies from the fit's error bars: every `(A, B)` they admit (#353).

`port.extensions.integer_copy.decode_copy_state` is the paper's decoding: a
state's `(mubar, p)` and its covariance give the **set** of integer pairs
inside the credible region, not one winner. `port.extensions.parameter_errors`
computes the covariance from the observed information of the objective the
HMM maximized. Neither was reached by a run. This joins them, and
`run_cnaster_port --copy-errors` calls it on the final fit.

## The scale comes from the pin, and the neutral state is (1, 1)

`mubar = (A + B) / 2` holds only on the de-biased scale. With the per-clone
`logmu_shift` folded in (`port.patch.hmm_nophasing`, #276), the likelihood is
flat along `mu -> c mu`, and `port.patch.hmrf.core_inference.pin_neutral`
fixes `c` by setting the normal clone's dominant balanced state (#299's
`neutral_state`) to `mu = 1`. So:

- the covariance is taken **in the pinned coordinates**: the neutral `log mu`
  is held at 0 and the rest differentiated, through the shift, whose
  normalizer is a function of every rate (`parameter_errors`' Jacobian);
- the neutral state's `mu` is 1 exactly and carries no error, so its
  decoding conditions on `mubar = 1`: the pairs of total 2 its allele
  fraction admits, `(1, 1)` or `(2, 0)`, by a one-dimensional test at the
  same level.

Without the shift the fit's scale is the baseline's, not the pin's, and the
decode would compare `(A + B) / 2` against a rate with an unknown per-clone
factor; `run_cnaster_port` refuses `--copy-errors` without it.

## Allele fractions are folded

Phasing makes the allele label arbitrary, so `p` is folded to the minor
fraction and the lattice is the unphased one (`acn_lattice(phased=False)`,
`A >= B`, `p = B / (A + B)`). Folding flips the sign of the `(mu, p)`
covariance where it applies and leaves the variances alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "Captured",
    "PinnedErrors",
    "copy_sets",
    "pinned_errors",
    "pinned_objective",
    "pseudobulk",
    "write_copy_sets",
]


class Captured(NamedTuple):
    """What `run_core_inference` was given, and what it returned."""

    single_X: np.ndarray
    lengths: np.ndarray
    single_base_nb_mean: np.ndarray
    single_total_bb_RD: np.ndarray
    result: Any


class PinnedErrors(NamedTuple):
    """One fit's `(mu, minor p)` per state, with their pinned covariance.

    `covariance` is `(n_states, 2, 2)` in `(mu, minor p)`; the neutral state's
    `mu` row and column are zero. `decrement` is the Newton decrement
    `g' S g`, which says whether the point is an optimum of the objective the
    covariance is the curvature of.
    """

    mu: np.ndarray
    minor: np.ndarray
    flipped: np.ndarray
    covariance: np.ndarray
    neutral: int
    decrement: float


def _column(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def pseudobulk(captured: Captured) -> dict[str, np.ndarray]:
    """The clone-summed inputs the HMM scored, stacked clone after clone.

    `run_core_inference` sums each clone's spots and concatenates the clones
    along the genome (`clone_stack_obs`), so the objective is one sequence of
    `n_clones * n_obs` with `lengths` tiled. The assignment is the fit's own.
    """
    assignment = np.asarray(captured.result["new_assignment"], dtype=np.int64)
    clones = np.unique(assignment)

    def summed(values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [values[..., assignment == c].sum(axis=-1) for c in clones]
        )

    return {
        "counts_nb": summed(captured.single_X[:, 0, :]),
        "counts_bb": summed(captured.single_X[:, 1, :]),
        "base_nb_mean": summed(captured.single_base_nb_mean),
        "total_bb_RD": summed(captured.single_total_bb_RD),
        "lengths": np.tile(captured.lengths, clones.size),
        "n_clones": np.asarray(clones.size),
    }


def pinned_objective(captured: Captured, free: np.ndarray) -> Any:
    """The HMM's negative log-likelihood over `(log mu_free, logit p, log alpha, log tau)`.

    The objective the HMM maximized, rebuilt in `jax`: per clone, the rates
    `log mu_k - log sum_g lambda_g mu_{s_c(g)}` over the fit's own decoded
    path, with `lambda` built as `hmrf.py:476` builds it. The path is held
    fixed, so the shift's derivative is through the rates alone. States not
    in `free` have `log mu = 0`: that is the pin, applied to the parameters.
    Dispersions are shared, as configured, and transitions held at the fit.
    """
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jsp

    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood

    result = captured.result
    n_states = _column(result["new_log_mu"]).size
    log_startprob = _column(result["new_log_startprob"])
    log_transmat = np.asarray(result["new_log_transmat"], dtype=np.float64)

    inputs = pseudobulk(captured)
    profile = captured.single_base_nb_mean.sum(axis=1)
    log_lambda = np.log(profile / profile.sum())
    path = np.asarray(result["pred_cnv"], dtype=np.int64)
    n_obs, n_clones = path.shape

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        rates = jnp.zeros(n_states).at[free].set(theta[: free.size])  # noqa: PD008
        dispersions = jnp.full(n_states, jnp.exp(theta[-2]))
        probabilities = jax.nn.sigmoid(theta[free.size : free.size + n_states])
        concentrations = jnp.full(n_states, jnp.exp(theta[-1]))

        blocks = []

        for clone in range(n_clones):
            shift = jsp.logsumexp(rates[path[:, clone]] + log_lambda)
            rows = slice(clone * n_obs, (clone + 1) * n_obs)
            blocks.append(
                emission(
                    rates - shift,
                    dispersions,
                    probabilities,
                    concentrations,
                    inputs["counts_nb"][rows],
                    inputs["base_nb_mean"][rows],
                    inputs["counts_bb"][rows],
                    inputs["total_bb_RD"][rows],
                )
            )

        return marginal_negative_log_likelihood(
            jnp.concatenate(blocks, axis=1),
            log_startprob,
            log_transmat,
            inputs["lengths"],
        )

    return objective


def coordinates(
    rates: np.ndarray, p: np.ndarray, free: np.ndarray, alpha: float, tau: float
) -> np.ndarray:
    """`theta` for :func:`pinned_objective`, from rates and allele fractions."""
    return np.concatenate(
        [rates[free], np.log(p / (1.0 - p)), [np.log(alpha), np.log(tau)]]
    )


def pinned_covariance(
    objective: Any,
    theta: np.ndarray,
    free: np.ndarray,
    mu: np.ndarray,
    p_binom: np.ndarray,
    flipped: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, 2, 2)` in `(mu, minor p)` at `theta`, and its decrement.

    A state outside `free` has no `mu` error.
    """
    import jax
    import jax.numpy as jnp

    from port.extensions.parameter_errors import parameter_errors

    n_states = mu.size
    estimate = parameter_errors(objective, theta)

    gradient = np.asarray(jax.grad(objective)(jnp.asarray(theta)))
    decrement = float(gradient @ estimate.covariance @ gradient)

    rates = np.zeros((n_states, n_states))
    rates[np.ix_(free, free)] = estimate.covariance[: free.size, : free.size]
    cross = np.zeros(n_states)
    cross[free] = [
        estimate.covariance[index, free.size + state]
        for index, state in enumerate(free)
    ]
    alleles = estimate.covariance[
        free.size : free.size + n_states, free.size : free.size + n_states
    ]
    slope = p_binom * (1.0 - p_binom)

    covariance = np.zeros((n_states, 2, 2))

    for k in range(n_states):
        off = cross[k] * mu[k] * slope[k]
        off = -off if flipped[k] else off
        covariance[k] = [
            [rates[k, k] * mu[k] ** 2, off],
            [off, alleles[k, k] * slope[k] ** 2],
        ]

    return covariance, decrement


def pinned_errors(captured: Captured) -> PinnedErrors:
    """The fit's `(mu, minor p)` and their covariance in the pinned coordinates.

    `mu` is `exp(new_log_mu)` as the pipeline returns it, pinned so #299's
    neutral state is 1. The neutral state is found again here with the same
    function, on the same path, so the coordinates held fixed are the ones
    the pin fixed.
    """
    from port.patch.hmm_nophasing.shifted_emission import neutral_state

    result = captured.result
    log_mu = _column(result["new_log_mu"])
    p_binom = _column(result["new_p_binom"])
    alpha = float(_column(result["new_alphas"])[0])
    tau = float(_column(result["new_taus"])[0])
    n_states = log_mu.size
    mu = np.exp(log_mu)

    flipped = p_binom > 0.5
    minor = np.where(flipped, 1.0 - p_binom, p_binom)

    neutral = neutral_state(log_mu, p_binom, np.asarray(result["pred_cnv"]))
    free = np.array([k for k in range(n_states) if k != neutral])

    covariance, decrement = pinned_covariance(
        pinned_objective(captured, free),
        coordinates(log_mu, p_binom, free, alpha, tau),
        free,
        mu,
        p_binom,
        flipped,
    )

    return PinnedErrors(mu, minor, flipped, covariance, int(neutral), decrement)


def _neutral_set(minor: float, variance: float, level: float) -> Any:
    """The neutral state: `mubar = 1` exactly, so the total is 2 and `p` decides."""
    from scipy.stats import chi2

    from port.extensions.integer_copy import IntegerCopyResult

    pairs = ((1, 1), (2, 0))
    implied = np.array([0.5, 0.0])
    distances = (implied - minor) ** 2 / variance
    order = np.argsort(distances, kind="stable")
    threshold = float(chi2.ppf(level, 1))

    return IntegerCopyResult(
        best=pairs[int(order[0])],
        consistent=tuple(pairs[int(i)] for i in order if distances[i] <= threshold),
        distance=float(distances[order[0]]),
        threshold=threshold,
        level=level,
    )


def copy_sets(
    errors: PinnedErrors,
    *,
    level: float = 0.95,
    max_allele_copy: int | None = None,
    max_total_copy: int | None = None,
) -> list[Any]:
    """Every `(A, B)` each state's error bars admit, at `level`.

    The caps default to the configured ones (`int_copy_num.max_total_copy`,
    read by `port.patch.integer_copy.configured_caps`, else `cnaster`'s 5 and
    6), so the set is drawn from the lattice the run's own integer decoder
    searches.
    """
    from port.extensions.integer_copy import acn_lattice, decode_copy_state
    from port.patch.integer_copy import configured_caps

    allele, total = configured_caps()
    lattice = acn_lattice(
        max_allele_copy=allele if max_allele_copy is None else max_allele_copy,
        max_total_copy=total if max_total_copy is None else max_total_copy,
        phased=False,
    )

    decoded = []

    for state in range(errors.mu.size):
        if state == errors.neutral:
            decoded.append(
                _neutral_set(
                    float(errors.minor[state]),
                    float(errors.covariance[state, 1, 1]),
                    level,
                )
            )
            continue

        decoded.append(
            decode_copy_state(
                [errors.mu[state], errors.minor[state]],
                errors.covariance[state],
                level=level,
                lattice=lattice,
            )
        )

    return decoded


def copy_set_table(errors: PinnedErrors, decoded: list[Any]) -> pd.DataFrame:
    """One row per `(state, A, B)` in a state's set; a state whose set is empty
    has one row with `A` and `B` empty, so it is reported rather than dropped."""
    rows = []

    for state, result in enumerate(decoded):
        base = {
            "state": state,
            "neutral": state == errors.neutral,
            "mu": float(errors.mu[state]),
            "p_minor": float(errors.minor[state]),
            "sigma_mu": float(np.sqrt(errors.covariance[state, 0, 0])),
            "sigma_p": float(np.sqrt(errors.covariance[state, 1, 1])),
            "best_A": result.best[0],
            "best_B": result.best[1],
            "best_distance": result.distance,
            "threshold": result.threshold,
            "level": result.level,
            "set_size": len(result.consistent),
        }

        if not result.consistent:
            rows.append({**base, "A": pd.NA, "B": pd.NA})

        for a, b in result.consistent:
            rows.append({**base, "A": a, "B": b})

    return pd.DataFrame(rows)


def write_copy_sets(run: Path, captured: Captured, *, level: float = 0.95) -> Path:
    """Write `cnv_copy_sets.tsv` into `run`, and return its path."""
    errors = pinned_errors(captured)
    table = copy_set_table(errors, copy_sets(errors, level=level))
    path = Path(run) / "cnv_copy_sets.tsv"

    with path.open("w") as handle:
        handle.write(
            f"# every (A, B) inside the {level:.0%} credible region of each "
            f"fitted state (#353); state {errors.neutral} is pinned to mu = 1; "
            f"Newton decrement {errors.decrement:.3e}\n"
        )
        table.to_csv(handle, sep="\t", index=False)

    return path
