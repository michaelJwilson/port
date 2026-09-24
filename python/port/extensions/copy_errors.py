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


BASE_FLOOR = 1e-12
"""The least expected depth a bin is given, so a zero baseline differentiates."""

VARIANCE_FLOOR = 1e-12
"""Added to each variance, so a held parameter's region is defined."""

BOUNDARY = 1e-3
"""An allele fraction this close to 0 or 1 is held rather than estimated."""

EPS_P = 1e-6
"""How close to 0 or 1 an allele fraction is taken, for a finite logit."""


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
        # NB a bin with no normal baseline expects no depth; its NB term is then
        #    `0 * log 0`, whose gradient is `nan`. Floored, it scores a zero
        #    count as certain to within `BASE_FLOOR`, and differentiates.
        "base_nb_mean": np.maximum(summed(captured.single_base_nb_mean), BASE_FLOOR),
        "total_bb_RD": summed(captured.single_total_bb_RD),
        "lengths": np.tile(captured.lengths, clones.size),
        "n_clones": np.asarray(clones.size),
    }


def pinned_objective(
    captured: Captured,
    free: np.ndarray,
    purity: np.ndarray | None = None,
    shares: np.ndarray | None = None,
    held: np.ndarray | None = None,
) -> Any:
    """The HMM's negative log-likelihood over `(log mu_free, logit p, log alpha, log tau)`.

    The objective the HMM maximized, rebuilt in `jax`: per clone, the rates
    `log mu_k - log sum_g lambda_g mu_{s_c(g)}` over the fit's own decoded
    path, with `lambda` built as `hmrf.py:476` builds it. The path is held
    fixed, so the shift's derivative is through the rates alone. States not
    in `free` have `log mu = 0`: that is the pin, applied to the parameters.
    Dispersions are shared, as configured, and transitions held at the fit.

    With `purity`, one tumour fraction per clone, each state's `(mu, p)` is the
    tumour component's: clone `c` sees depth `rho_c mu + 1 - rho_c` and allele
    share `(2 rho_c mu p + 1 - rho_c) / (2 rho_c mu + 2 (1 - rho_c))`, its
    spots being `rho_c` tumour and the rest normal `(1, 1)` (#367). The
    fractions are held, as `lattice_decode` fitted them.
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
    estimated = np.arange(n_states) if shares is None else np.asarray(shares)
    fixed = np.full(n_states, 0.5) if held is None else np.asarray(held)

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        rates = jnp.zeros(n_states).at[free].set(theta[: free.size])  # noqa: PD008
        dispersions = jnp.full(n_states, jnp.exp(theta[-2]))
        probabilities = (
            jnp.asarray(fixed)
            .at[estimated]
            .set(
                jax.nn.sigmoid(theta[free.size : free.size + estimated.size])
            )
        )
        concentrations = jnp.full(n_states, jnp.exp(theta[-1]))

        blocks = []

        for clone in range(n_clones):
            if purity is None:
                observed, shares = rates, probabilities
            else:
                rho = float(purity[clone])
                tumour = rho * jnp.exp(rates)
                observed = jnp.log(tumour + 1.0 - rho)
                shares = (2.0 * tumour * probabilities + 1.0 - rho) / (
                    2.0 * tumour + 2.0 * (1.0 - rho)
                )

            shift = jsp.logsumexp(observed[path[:, clone]] + log_lambda)
            rows = slice(clone * n_obs, (clone + 1) * n_obs)
            blocks.append(
                emission(
                    observed - shift,
                    dispersions,
                    shares,
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
    rates: np.ndarray,
    p: np.ndarray,
    free: np.ndarray,
    alpha: float,
    tau: float,
    shares: np.ndarray | None = None,
) -> np.ndarray:
    """`theta` for :func:`pinned_objective`, from rates and allele fractions.

    An allele fraction of exactly 0 or 1 -- an LOH state on a pure sample --
    has no finite logit; it is taken at `EPS_P` from the boundary.
    """
    share = np.clip(p, EPS_P, 1.0 - EPS_P)
    share = share if shares is None else share[shares]
    return np.concatenate(
        [rates[free], np.log(share / (1.0 - share)), [np.log(alpha), np.log(tau)]]
    )


def pinned_covariance(
    objective: Any,
    theta: np.ndarray,
    free: np.ndarray,
    mu: np.ndarray,
    p_binom: np.ndarray,
    flipped: np.ndarray,
    shares: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """`(n_states, 2, 2)` in `(mu, minor p)` at `theta`, and its decrement.

    A state outside `free` has no `mu` error, and one outside `shares` no `p`
    error: those parameters are held.
    """
    import jax
    import jax.numpy as jnp

    from port.extensions.parameter_errors import parameter_errors

    n_states = mu.size
    # NB the dispersions are held at `theta`'s: they are nuisance parameters
    #    to the decode, and at the binomial limit (a pure sample's allele
    #    counts) `log tau` has no curvature at all, so the full information is
    #    singular there. The errors are conditional on them.
    dispersions = jnp.asarray(theta[-2:])

    def conditional(point: Any) -> Any:
        return objective(jnp.concatenate([point, dispersions]))

    theta = np.asarray(theta[:-2])
    estimate = parameter_errors(conditional, theta)

    gradient = np.asarray(jax.grad(conditional)(jnp.asarray(theta)))
    decrement = float(gradient @ estimate.covariance @ gradient)

    rates = np.zeros((n_states, n_states))
    rates[np.ix_(free, free)] = estimate.covariance[: free.size, : free.size]
    estimated = np.arange(n_states) if shares is None else np.asarray(shares)
    column = {int(state): free.size + i for i, state in enumerate(estimated)}
    cross = np.zeros(n_states)
    cross[free] = [
        estimate.covariance[index, column[int(state)]] if int(state) in column else 0.0
        for index, state in enumerate(free)
    ]
    alleles = np.zeros((n_states, n_states))
    block = estimate.covariance[
        free.size : free.size + estimated.size, free.size : free.size + estimated.size
    ]
    alleles[np.ix_(estimated, estimated)] = block
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


def _refit(objective: Any, theta: np.ndarray) -> np.ndarray:
    """The objective's optimum from `theta`, paths held, by L-BFGS on its gradient."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def fun(point: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(point))
        return float(value), np.asarray(gradient, dtype=np.float64)

    result = minimize(fun, theta, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return np.asarray(result.x, dtype=np.float64)


def pinned_errors(captured: Captured, purity: np.ndarray | None = None) -> PinnedErrors:
    """The fit's `(mu, minor p)` and their covariance in the pinned coordinates.

    With `purity` the objective mixes each clone's tumour fraction in
    (:func:`pinned_objective`), the rates and fractions are refitted under it,
    and `(mu, p)` and the covariance are the tumour component's.

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

    path = np.asarray(result["pred_cnv"])
    neutral = neutral_state(log_mu, p_binom, path)
    visited = set(np.unique(path % n_states).tolist())
    # NB what the data identify: a state no bin visits has no information,
    #    and an allele fraction at its boundary (an LOH state on a pure
    #    sample) has no curvature in its logit. Both are held.
    free = np.array([k for k in range(n_states) if k != neutral and k in visited])
    shares = np.array(
        [
            k
            for k in range(n_states)
            if k in visited and BOUNDARY < p_binom[k] < 1.0 - BOUNDARY
        ]
    )
    held = np.clip(p_binom, EPS_P, 1.0 - EPS_P)
    objective = pinned_objective(captured, free, purity, shares, held)
    theta = coordinates(log_mu, p_binom, free, alpha, tau, shares)

    # NB the curvature is taken at this objective's own optimum, paths held:
    #    the pipeline's fit is not one to within the conditioning check (on the
    #    pure easy fixture its information has an eigenvalue ratio of -1e-6),
    #    and with `purity` the fractions, fitted first (`lattice_decode`),
    #    move the optimum further.
    theta = _refit(objective, theta)

    log_mu[free] = theta[: free.size]
    log_mu[neutral] = 0.0
    p_binom = held.copy()
    p_binom[shares] = 1.0 / (1.0 + np.exp(-theta[free.size : free.size + shares.size]))
    mu = np.exp(log_mu)
    flipped = p_binom > 0.5
    minor = np.where(flipped, 1.0 - p_binom, p_binom)

    covariance, decrement = pinned_covariance(
        objective, theta, free, mu, p_binom, flipped, shares
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

        # NB a held parameter (`pinned_errors`) is known exactly; its variance
        #    is floored so the region is defined and admits only its value.
        covariance = errors.covariance[state] + np.eye(2) * VARIANCE_FLOOR
        decoded.append(
            decode_copy_state(
                [errors.mu[state], errors.minor[state]],
                covariance,
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
