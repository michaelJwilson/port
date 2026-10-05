"""Copy-state starts sampled on the HMM's own NLL by `sal`'s samplers, through a `sal` `Objective` (#540, #634).

Ticket: #634 -- `sal.sample.hmc`'s `sample`, `anneal` and
  `parallel_tempering` replaced port's own samplers on this objective
  (`hmm_samplers`, deleted) after matching them at equal evaluations.
Measurement: `tests.studies.copy_state_stream` on
  `sim/manifests/baseline/dev_tree_1s_hard.toml` r3-r12, 10 seeds (PR #642):
  median rows missed after Baum-Welch 1.14 / 1.12 / 1.14% against port's
  1.19 / 1.17 / 1.10% (anneal / tempering / hmc), n = 100 each.
Exit: retire with the #540 study; or graduate when `sal`'s count-pair HMM
  objective costs no more per evaluation. At `sal` 253c84f it is this
  module's `oracle` (`EmissionHmmObjective` of a `CountPairEmission`, #634
  gaps 1-3) and costs 1.6x per value and gradient, with no JAX twin for the
  pair, and `Restricted` cannot hold `tau` with `p` free (T- #671).

**The objective.** `HmmObjective` is `jax_hmm.marginal_negative_log_likelihood`
of `jax_hmm.emission` on the clone-stacked rows, per segment (`lengths`),
over `theta = (log_mu, logit p_binom)`, one of each per state. The
dispersions, the stickiness and the uniform start probabilities are held,
at `known_copy.hmm`'s `ALPHA`, `TAU` and `T` (`objective_for`), the values
`known_copy.decode` scores a start at. No per-clone shift (#276). `theta`
crosses from torch to JAX as a `float64` array; `__call__` is
differentiable by a `torch.autograd.Function` carrying JAX's gradient, and
`value_and_gradient` and `energy` are declared (`sal.opt.objective`), so
`sal` reads the gradient from JAX rather than from a graph.

**Evaluations are points, not calls.** A `sal` transition asks for the
value at its start and end point beside the trajectory's gradients, and
`anneal` and `parallel_tempering` ask again for the point they kept. Every
one of those points is also a point a trajectory took a gradient at, so
the objective keeps the last `MEMO` points' value and gradient and
`evaluations` counts the forward-backward passes actually run.

**The starts.** `sample(name, ...)`: the initial point is `initial_point`,
the first draw from `rng`; trajectories are `LEAPFROG` leapfrog steps; the
start is the best point `sal` reports.

- `hmc-hmm`: `sal.sample.hmc.sample` at `temperature`, `draws` draws; with
  `adapt`, after `sal`'s `Adaptation` warm-up of `warmup` proposals (dual
  averaging toward `target` from `step`, a diagonal mass), else at the fixed
  `step` after `warmup` burn-in proposals. The best of the draws.
- `anneal-hmm`: `sal.sample.hmc.anneal` on an `ExponentialTempSchedule`
  from `t_start` to 1 over `steps` proposals, at the fixed `step`.
- `tempering-hmm`: `sal.sample.hmc.parallel_tempering` on `RUNGS` rungs
  geometric from 1 to `t_top`, `rounds` rounds, at the fixed `step`; every
  replica starts at the initial point, as `sal` starts them.

`sal`'s step needs no rescaling with temperature: its momentum is drawn with
variance `T`. `sal` has no step adaptation in `anneal` or
`parallel_tempering`, so there `step` is a tuned constant.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULTS",
    "LEAPFROG",
    "MEMO",
    "SAMPLERS",
    "HmmObjective",
    "Sampled",
    "initial_point",
    "negative_log_likelihood",
    "objective_for",
    "sample",
]

LEAPFROG = 8
"""Leapfrog steps per proposal: 8 passes each, the first gradient being the last proposal's end."""

MEMO = 64
"""Points whose value and gradient are kept: a tempering round's 4 replicas of 9 points each fit."""

HMC_ADAPT = 8
"""`hmc-hmm`'s warm-up where a setting names none."""

RUNGS = 4
"""`tempering-hmm`: replicas on the ladder."""

DEFAULTS: dict[str, dict[str, float]] = {
    "anneal-hmm": {"t_start": 1e2, "steps": 54, "step": 3e-3},
    "tempering-hmm": {"t_top": 1e4, "rounds": 13, "step": 1e-3},
    "hmc-hmm": {"temperature": 1.0, "warmup": 12, "draws": 13, "step": 1e-3, "adapt": 1.0, "target": 0.65},
}  # fmt: skip
"""Each start's knobs where no `setting` is given: the values `tests.studies.copy_state_stream --tune` chose
on `dev_tree_1s_hard`'s held-out realizations 0-2 (`tests/studies/copy_sampler_settings.json`), 5 seeds
per setting, over grids shaped as port's samplers' were (9 / 9 / 7 settings).

Budgets are the passes port's deleted samplers spent at their tuned schedules, not exceeded: 54
annealing steps (433 passes against 433), 13 tempering rounds (417 against 436), 12 + 13 hmc
proposals (204-207 against 217), counted on realization 0, 10 seeds.

`hmc`'s warm-up is 12 rather than port's 8, from a step of 1e-3: `sal`'s `Adaptation` raised (zero
warm-up variance) when the chain did not move in the 2 proposals it records at a warm-up of 8, on all
5 seeds of realization 0 at steps of 1e-3, 3e-3 and 1e-2 (#634, gap 5)."""


SAMPLERS = tuple(DEFAULTS)


class Sampled(NamedTuple):
    """An arm's best states, their NLL, the initial point's NLL, and the forward-backward passes it ran."""

    log_mu: np.ndarray
    p_binom: np.ndarray
    nll: float
    initial_nll: float
    evaluations: int


@functools.cache
def _compiled(n_states: int, lengths: tuple[int, ...]) -> Any:
    """`(theta, data, log_startprob, log_transmat, alphas, taus) -> (nll, grad)`, jitted for one shape."""
    import jax

    from port.extensions import jax_setup  # noqa: F401  (float64, before any array)
    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood

    k = n_states
    lengths_array = np.asarray(lengths)

    def nll(theta: Any, data: tuple[Any, ...], held: tuple[Any, ...]) -> Any:
        total, b, exposure, trials = data
        log_startprob, log_transmat, alphas, taus = held
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

    return jax.jit(jax.value_and_grad(nll))


class _Carried(torch.autograd.Function):
    """The objective's value as a torch scalar whose backward is JAX's gradient."""

    @staticmethod
    def forward(
        ctx: Any, _theta: torch.Tensor, value: torch.Tensor, grad: torch.Tensor
    ) -> torch.Tensor:
        # NB `_theta` is an input only so autograd records the dependency.
        ctx.save_for_backward(grad)
        return value.clone()

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (grad,) = ctx.saved_tensors
        return upstream * grad, None, None


class HmmObjective:
    """The copy-state HMM's forward NLL over `theta = (log_mu, logit p_binom)`, as a `sal.opt.objective.Objective`.

    Parameters
    ----------
    total, b, exposure, trials : np.ndarray
        Per row: read depth, B-allele reads, the depth's exposure, and allele
        trials, the clones stacked along the genome.
    lengths : Any
        Rows per segment; the forward recursion restarts at each.
    n_states : int
        Copy states.
    alpha, tau : float
        The negative binomial and beta-binomial dispersions, held.
    stay : float
        The probability of keeping the state from one row to the next, held;
        the rest spread evenly over the other states.
    start : np.ndarray
        `initial()`: the `theta` a sampler starts from.
    """

    def __init__(
        self,
        total: np.ndarray,
        b: np.ndarray,
        exposure: np.ndarray,
        trials: np.ndarray,
        lengths: Any,
        n_states: int,
        *,
        alpha: float,
        tau: float,
        stay: float,
        start: np.ndarray,
    ) -> None:
        k = int(n_states)
        self.n_states = k
        self.lengths = tuple(int(v) for v in np.ravel(lengths))
        self.data = tuple(
            np.asarray(v, dtype=np.float64) for v in (total, b, exposure, trials)
        )
        move = np.log((1.0 - stay) / max(k - 1, 1))
        self.held = (
            np.full(k, -np.log(k)),
            np.where(np.eye(k, dtype=bool), np.log(stay), move),
            np.full(k, float(alpha)),
            np.full(k, float(tau)),
        )
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.evaluations = 0
        self._memo: OrderedDict[bytes, tuple[float, np.ndarray]] = OrderedDict()
        self._run: Callable[..., Any] = _compiled(k, self.lengths)

    def _evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """`(U(x), dU/dx)`, from the memo where `x` was evaluated among the last `MEMO` points."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        key = x.tobytes()
        kept = self._memo.get(key)
        if kept is not None:
            self._memo.move_to_end(key)
            return kept
        value, grad = self._run(x, self.data, self.held)
        found = (float(value), np.asarray(grad, dtype=np.float64))
        self.evaluations += 1
        self._memo[key] = found
        if len(self._memo) > MEMO:
            self._memo.popitem(last=False)
        return found

    def initial(self) -> torch.Tensor:
        """The start `theta`, as given."""
        return torch.from_numpy(self.start.copy())

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """`log_mu` and `p_binom`, one per state."""
        k = self.n_states
        return {"log_mu": theta[:k], "p_binom": torch.sigmoid(theta[k:])}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """`(log_mu, logit p_binom)`."""
        p = torch.as_tensor(named["p_binom"], dtype=torch.float64)
        return torch.cat(
            [torch.as_tensor(named["log_mu"], dtype=torch.float64), torch.logit(p)]
        )

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """`U(theta)`, differentiable in `theta` through JAX's gradient."""
        value, grad = self._evaluate(theta.detach().numpy())
        return _Carried.apply(  # type: ignore[no-any-return]
            theta,
            torch.tensor(value, dtype=torch.float64),
            torch.from_numpy(grad.copy()),
        )

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(U(theta), dU/dtheta)`, detached."""
        value, grad = self._evaluate(theta.detach().numpy())
        return torch.tensor(value, dtype=torch.float64), torch.from_numpy(grad.copy())

    def energy(self, x: np.ndarray) -> float:
        """`U(x)` at a `float64` array."""
        return self._evaluate(x)[0]


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


def negative_log_likelihood(
    log_mu: Any,
    p_binom: Any,
    total: np.ndarray,
    b: np.ndarray,
    exposure: np.ndarray,
    trials: np.ndarray,
    lengths: Any,
) -> float:
    """The objective the starts sample, at given states: `p_binom` clipped to `[1e-4, 1 - 1e-4]` as `decode` clips it."""
    log_mu = np.asarray(log_mu, dtype=np.float64).ravel()
    p = np.clip(np.asarray(p_binom, dtype=np.float64).ravel(), 1e-4, 1 - 1e-4)
    theta = np.concatenate([log_mu, np.log(p / (1.0 - p))])
    objective = objective_for(total, b, exposure, trials, lengths, log_mu.size, theta)
    return objective.energy(theta)


def objective_for(
    total: np.ndarray,
    b: np.ndarray,
    exposure: np.ndarray,
    trials: np.ndarray,
    lengths: Any,
    n_states: int,
    start: np.ndarray,
) -> HmmObjective:
    """`HmmObjective` at `known_copy.hmm`'s `ALPHA`, `TAU` and `T`, the values `decode` scores a start at."""
    from port.sandbox.known_copy.hmm import ALPHA, TAU, T

    return HmmObjective(
        total, b, exposure, trials, lengths, n_states,
        alpha=ALPHA, tau=TAU, stay=T, start=start,
    )  # fmt: skip


def sample(
    name: str,
    total: np.ndarray,
    b: np.ndarray,
    exposure: np.ndarray,
    trials: np.ndarray,
    lengths: Any,
    n_states: int,
    rng: np.random.Generator,
    setting: dict[str, float] | None = None,
) -> Sampled:
    """`name`'s best states `(log_mu, p_binom)` by `sal`'s sampler, one of `SAMPLERS`; `setting` in place of `DEFAULTS[name]`.

    `nll` is the lowest value the sampler reports; `initial_nll` the initial
    point's. `anneal` and `tempering` count the initial point, so for them
    `nll <= initial_nll`; `hmc` keeps the best of its draws alone.
    """
    from sal.sample.chain import Adaptation
    from sal.sample.hmc import anneal, parallel_tempering
    from sal.sample.hmc import sample as hmc_sample
    from sal.sample.schedule import ExponentialTempSchedule

    if name not in DEFAULTS:
        msg = f"{name!r} is not one of {SAMPLERS}"
        raise ValueError(msg)
    knobs = {**DEFAULTS[name], **(setting or {})}
    theta0 = initial_point(total, exposure, n_states, rng)
    objective = objective_for(total, b, exposure, trials, lengths, n_states, theta0)
    initial = objective.energy(theta0)
    step = float(knobs["step"])
    if name == "hmc-hmm":
        adapt, warmup = bool(knobs["adapt"]), int(knobs.get("warmup", HMC_ADAPT))
        chain = hmc_sample(
            objective, rng, int(knobs["draws"]), step_size=step, n_steps=LEAPFROG,
            burn_in=0 if adapt else warmup, temperature=float(knobs["temperature"]),
            adaptation=Adaptation(warmup, float(knobs["target"]), 0.1) if adapt else None,
        )  # fmt: skip
        draws = chain.draws.numpy()
        values = np.array([objective.energy(d) for d in draws])
        finite = np.where(np.isfinite(values), values, np.inf)
        best_theta, best = draws[int(np.argmin(finite))], float(np.min(finite))
    elif name == "anneal-hmm":
        schedule = ExponentialTempSchedule(
            float(knobs["t_start"]), 1.0, int(knobs["steps"])
        )
        found = anneal(objective, schedule, rng, step_size=step, n_steps=LEAPFROG)
        best_theta, best = found.best.numpy(), float(found.value)
    else:
        ladder = np.geomspace(1.0, float(knobs["t_top"]), RUNGS).tolist()
        tempered = parallel_tempering(
            objective,
            ladder,
            rng,
            int(knobs["rounds"]),
            step_size=step,
            n_steps=LEAPFROG,
        )
        best_theta, best = tempered.best.numpy(), float(tempered.value)
    k = n_states
    return Sampled(
        best_theta[:k].copy(),
        1.0 / (1.0 + np.exp(-best_theta[k:])),
        best,
        initial,
        objective.evaluations,
    )
