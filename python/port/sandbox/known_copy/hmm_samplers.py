"""Copy-state starts sampled on the HMM's own negative log-likelihood (#540).

Ticket: #540 -- copy-state starts at known clones, polished by Baum-Welch
  (`tests/studies/copy_state_stream.py`).
Measurement: `docs/study-copy-states.md`: each start's gap in log-likelihood
  and share of rows off their planted state, before and after Baum-Welch.
Exit: retire with the study; a start it finds better graduates through
  `port.extensions.copy_starts`.

`sal`'s `anneal`, `tempering` and `hmc` starts sample a Gaussian-mixture
surrogate over raw count rows and snap each component mean to the nearest
observed row. These sample the objective the start seeds instead:
`port.extensions.jax_hmm.marginal_negative_log_likelihood` of `emission`, on
the clone-stacked rows, with no snapping -- the draw's parameters are the
start's states.

- **Parameters.** Per state `log_mu` and `logit(p_binom)`, unconstrained.
  The dispersions are fixed at `known_copy.hmm.ALPHA` and `TAU`, the values
  `known_copy.decode` scores a start at; the stickiness is `known_copy.hmm.T`
  and the start probabilities uniform. No per-clone shift (#276): the
  sampler reads each row's exposure as given.
- **Initial point.** Uninformed and seeded: `log_mu` uniform over the 5-95%
  range of the observed log depth ratios, `p_binom` uniform on (0.05, 0.95).
- **Moves.** Hamiltonian trajectories of `LEAPFROG` steps, unit mass, the
  energy `NLL / temperature`; one jit-compiled trajectory serves all three.
  The step size adapts multiplicatively on acceptance (`GROW`, `SHRINK`),
  scaled by `sqrt(temperature)` so one step size holds across a ladder.
- `hmc`: `HMC_ADAPT` adapting trajectories at temperature 1, then
  `HMC_DRAWS` at a frozen step; the lowest-NLL point visited.
- `anneal`: `ANNEAL_STEPS` trajectories under a temperature falling
  exponentially from `T_START` to 1, adapting throughout; the best point.
- `tempering`: `RUNGS` replicas on a geometric ladder from 1 to `T_TOP`,
  `ROUNDS` rounds of one trajectory each and a neighbour swap; the best point
  at any temperature.

"Best point" counts every point whose NLL was evaluated at a trajectory's end,
accepted or not, and the initial point: none is worse than the start.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import numpy as np

__all__ = ["SAMPLERS", "Sampled", "initial_point", "negative_log_likelihood", "sample"]

LEAPFROG = 8
"""Leapfrog steps per trajectory."""

GROW, SHRINK = 1.25, 0.6
"""The step size's factor after an accepted and a rejected trajectory."""

STEP0 = 1e-2
"""The initial step size at temperature 1."""

HMC_ADAPT, HMC_DRAWS = 8, 16
ANNEAL_STEPS = 24
T_START = 1e3
RUNGS, ROUNDS = 4, 6
T_TOP = 1e3

SAMPLERS = ("anneal-hmm", "tempering-hmm", "hmc-hmm")


class Sampled(NamedTuple):
    """A sampler's best states, their NLL, the initial point's NLL, and the NLL evaluations it spent."""

    log_mu: np.ndarray
    p_binom: np.ndarray
    nll: float
    initial_nll: float
    evaluations: int


@functools.cache
def _trajectory(n_states: int, lengths: tuple[int, ...]) -> Any:
    """`(theta, momentum, step, temperature, data) -> (theta, nll, kinetic)` jitted for one call's shape."""
    import jax
    import jax.numpy as jnp

    from port.extensions import jax_setup  # noqa: F401
    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood
    from port.sandbox.known_copy.hmm import ALPHA, TAU, T

    k = n_states
    stay = np.log(T)
    move = np.log((1.0 - T) / max(k - 1, 1))
    log_transmat = jnp.asarray(np.where(np.eye(k, dtype=bool), stay, move))
    log_startprob = jnp.full(k, -np.log(k))
    alphas, taus = jnp.full(k, ALPHA), jnp.full(k, TAU)
    lengths_array = np.asarray(lengths)

    def nll(theta: Any, data: tuple[Any, ...]) -> Any:
        total, b, exposure, trials = data
        log_emission = emission(
            theta[:k],
            alphas,
            jax.nn.sigmoid(theta[k:]),
            taus,
            total,
            exposure,
            b,
            trials,
        )
        return marginal_negative_log_likelihood(
            log_emission, log_startprob, log_transmat, lengths_array
        )

    value_and_grad = jax.value_and_grad(nll)

    def run(theta: Any, momentum: Any, step: Any, temperature: Any, data: Any) -> Any:
        _, grad = value_and_grad(theta, data)
        momentum = momentum - 0.5 * step * grad / temperature

        def leap(_: int, carry: Any) -> Any:
            theta, momentum = carry
            theta = theta + step * momentum
            _, grad = value_and_grad(theta, data)
            return theta, momentum - step * grad / temperature

        theta, momentum = jax.lax.fori_loop(0, LEAPFROG - 1, leap, (theta, momentum))
        theta = theta + step * momentum
        value, grad = value_and_grad(theta, data)
        momentum = momentum - 0.5 * step * grad / temperature
        return theta, value, 0.5 * jnp.sum(momentum**2)

    return jax.jit(run), jax.jit(nll)


def negative_log_likelihood(
    log_mu: Any,
    p_binom: Any,
    total: np.ndarray,
    b: np.ndarray,
    exposure: np.ndarray,
    trials: np.ndarray,
    lengths: Any,
) -> float:
    """The objective the samplers draw on, at given states: `p_binom` clipped to `[1e-4, 1 - 1e-4]` as `decode` clips it."""
    log_mu = np.asarray(log_mu, dtype=np.float64).ravel()
    p = np.clip(np.asarray(p_binom, dtype=np.float64).ravel(), 1e-4, 1 - 1e-4)
    _, nll = _trajectory(log_mu.size, tuple(int(v) for v in np.ravel(lengths)))
    data = tuple(np.asarray(v, dtype=np.float64) for v in (total, b, exposure, trials))
    return float(nll(np.concatenate([log_mu, np.log(p / (1.0 - p))]), data))


def initial_point(
    total: np.ndarray, exposure: np.ndarray, n_states: int, rng: np.random.Generator
) -> np.ndarray:
    """`[log_mu, logit p]`: `log_mu` uniform over the observed log depth ratios' 5-95% range, `p` on (0.05, 0.95)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.asarray(total, float) / np.asarray(exposure, float))
    finite = ratio[np.isfinite(ratio)]
    # NB a call with no exposure (the BAF-only stage) has no depth to read.
    low, high = np.quantile(finite, [0.05, 0.95]) if finite.size else (0.0, 0.0)
    log_mu = rng.uniform(low, high, n_states)
    p = rng.uniform(0.05, 0.95, n_states)
    return np.concatenate([log_mu, np.log(p / (1.0 - p))])


class _Chain:
    """One replica: its point, NLL and step size; tracks the best point any replica reached."""

    def __init__(self, sampler: _Sampler, theta: np.ndarray, value: float) -> None:
        self.sampler, self.theta, self.value, self.step = sampler, theta, value, STEP0

    def move(self, temperature: float, adapt: bool) -> None:
        s = self.sampler
        momentum = s.rng.standard_normal(self.theta.size)
        theta, value, kinetic = s.run(
            self.theta, momentum, self.step * np.sqrt(temperature), temperature, s.data
        )
        theta, value = np.asarray(theta), float(value)
        s.evaluations += LEAPFROG + 1
        s.offer(theta, value)
        log_accept = (
            (self.value - value) / temperature
            + 0.5 * float(momentum @ momentum)
            - float(kinetic)
        )
        accepted = np.isfinite(log_accept) and np.log(s.rng.uniform()) < log_accept
        if accepted:
            self.theta, self.value = theta, value
        if adapt:
            self.step *= GROW if accepted else SHRINK


class _Sampler:
    def __init__(
        self,
        data: tuple[np.ndarray, ...],
        lengths: Any,
        n_states: int,
        rng: np.random.Generator,
    ) -> None:
        self.run, self.nll = _trajectory(
            n_states, tuple(int(v) for v in np.ravel(lengths))
        )
        self.data, self.rng, self.k = data, rng, n_states
        self.best: tuple[float, np.ndarray] = (np.inf, np.empty(0))
        self.evaluations = 0

    def offer(self, theta: np.ndarray, value: float) -> None:
        if np.isfinite(value) and value < self.best[0]:
            self.best = (value, theta)

    def chain(self, theta: np.ndarray) -> _Chain:
        value = float(self.nll(theta, self.data))
        self.evaluations += 1
        self.offer(theta, value)
        return _Chain(self, theta, value)


def sample(
    name: str,
    total: np.ndarray,
    b: np.ndarray,
    exposure: np.ndarray,
    trials: np.ndarray,
    lengths: Any,
    n_states: int,
    rng: np.random.Generator,
) -> Sampled:
    """`name`'s best states `(log_mu, p_binom)` on the rows, one of `SAMPLERS`."""
    data = tuple(np.asarray(v, dtype=np.float64) for v in (total, b, exposure, trials))
    sampler = _Sampler(data, lengths, n_states, rng)
    theta0 = initial_point(total, exposure, n_states, rng)
    first = sampler.chain(theta0)
    initial = first.value
    if name == "hmc-hmm":
        for i in range(HMC_ADAPT + HMC_DRAWS):
            first.move(1.0, adapt=i < HMC_ADAPT)
    elif name == "anneal-hmm":
        for temperature in np.geomspace(T_START, 1.0, ANNEAL_STEPS):
            first.move(float(temperature), adapt=True)
    elif name == "tempering-hmm":
        ladder = np.geomspace(1.0, T_TOP, RUNGS)
        chains = [first] + [
            sampler.chain(initial_point(total, exposure, n_states, rng))
            for _ in range(RUNGS - 1)
        ]
        for r in range(ROUNDS):
            for chain, temperature in zip(chains, ladder, strict=True):
                chain.move(float(temperature), adapt=True)
            i = r % (RUNGS - 1)
            j = i + 1
            log_swap = (chains[i].value - chains[j].value) * (
                1.0 / ladder[i] - 1.0 / ladder[j]
            )
            if np.log(rng.uniform()) < log_swap:
                chains[i].theta, chains[j].theta = chains[j].theta, chains[i].theta
                chains[i].value, chains[j].value = chains[j].value, chains[i].value
    else:
        msg = f"{name!r} is not one of {SAMPLERS}"
        raise ValueError(msg)
    value, theta = sampler.best
    k = n_states
    return Sampled(
        theta[:k].copy(),
        1.0 / (1.0 + np.exp(-theta[k:])),
        value,
        initial,
        sampler.evaluations,
    )
