"""Copy-state starts sampled on the HMM's own NLL by `sal`'s samplers, through a `sal` `Objective` (#540, #634).

Ticket: #634 -- `sal.sample.hmc`'s `sample`, `anneal` and
  `parallel_tempering` replaced port's own samplers on this objective
  (`hmm_samplers`, deleted) after matching them at equal evaluations.
Measurement: `port.studies.copy_state_stream` on
  `sim/manifests/baseline/dev_tree_1s_hard.toml` r3-r12, 10 seeds (PR #642):
  median rows missed after Baum-Welch 1.14 / 1.12 / 1.14% against port's
  1.19 / 1.17 / 1.10% (anneal / tempering / hmc), n = 100 each.
Exit: retire with the #540 study. Built on `sal`'s count-pair HMM objective
  since T- #707 (sal 006e49d): `Backend.JAX` costs 0.42-0.50x port's former
  jitted `jax_hmm` forward at 4 threads and 0.38-0.51x at 1 core, per value
  and gradient on PR- #672's instance, so that forward was deleted.
  `Backend.RUST` since sal #1265 (sal 9730280): its compiled `count_hmm`
  kernel takes this pair with a covariate per channel, 24.0 ms against
  `Backend.JAX`'s 119.0 ms per value and gradient at 200 segments of
  100-3,000 (328,785 positions), four states, min of 3; 9.4e-14 relative on
  the value and 4.5e-12 of the largest gradient component apart at 5 points.

**The objective.** `HmmObjective` is `sal`'s `EmissionHmmObjective`
(`backend=Backend.RUST`, sal #1248, #1265) of a
`RateConcentrationCountPairEmission` (sal #1205), independent form, on the
clone-stacked rows as a `Ragged` of segments (`lengths`), exposure and
trials as its covariate. `Restricted` varies its `mean` and `rate` blocks,
so `theta = (log_mu, logit p_binom)`, one of each per state; the
dispersions, the stickiness and the uniform start probabilities are held at
`known_copy.hmm`'s `ALPHA`, `TAU` and `T` (`objective_for`), the values
`known_copy.decode` scores a start at. No per-clone shift (#276).
`value_and_gradient` and `energy` are declared (`sal.opt.objective`), and
`__call__` returns the same value, differentiable by a
`torch.autograd.Function` carrying the kernel's gradient.

**Departure from `sal`.** `sal`'s own `__call__` is autograd through its
torch forward, its oracle route: `sal`'s samplers read potentials through
`__call__`, so it would run a second, slower implementation beside the compiled
gradient and differ from it at round-off. `_Carried` keeps one.

**Evaluations are passes.** `evaluations` counts the value-and-gradient
passes run. `sal` carries `U` and `grad U` along a chain (sal #1217, #1222),
so the former 64-point memo is gone; the last point's pass is kept, because
`sample` scores the initial point and `anneal` then asks for its value and
gradient again (435 passes against a budget of 433 without it, T- #707).

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
variance `T`. `anneal` and `parallel_tempering` run at the tuned constant
`step`; sal #1208's `adaptation` for them is not adopted (T- #707).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import torch

__all__ = [
    "DEFAULTS",
    "LEAPFROG",
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

HMC_ADAPT = 8
"""`hmc-hmm`'s warm-up where a setting names none."""

RUNGS = 4
"""`tempering-hmm`: replicas on the ladder."""

DEFAULTS: dict[str, dict[str, float]] = {
    "anneal-hmm": {"t_start": 1e2, "steps": 54, "step": 3e-3},
    "tempering-hmm": {"t_top": 1e4, "rounds": 13, "step": 1e-3},
    "hmc-hmm": {"temperature": 1.0, "warmup": HMC_ADAPT, "draws": 13, "step": 1e-3, "adapt": 1.0, "target": 0.65},
}  # fmt: skip
"""Each start's knobs where no `setting` is given: the values `port.studies.copy_state_stream --tune` chose
on `dev_tree_1s_hard`'s held-out realizations 0-2 (`python/port/studies/copy_sampler_settings.json`), 5 seeds
per setting, over grids shaped as port's samplers' were (9 / 9 / 7 settings).

Budgets are the passes port's deleted samplers spent at their tuned schedules, not exceeded: 54
annealing steps (433 passes against 433), 13 tempering rounds (417 against 436), 12 + 13 hmc
proposals (204-207 against 217), counted on realization 0, 10 seeds; 8 + 13 hmc proposals since
T- #707.

`hmc`'s warm-up is port's 8 (`HMC_ADAPT`). It was 12 at `sal` b61dfba-253c84f, whose `Adaptation`
raised on zero warm-up variance when the chain did not move in the 2 proposals it records at 8 (#634,
gap 5); `sal` #1207 regularizes that variance and reports the coordinate on `Adapted.flat` (T- #707).
At 8 on `sal` 006e49d, realization 0 (`d2938975`), seeds 0-3: best NLL 79,704-79,802 (median 79,724)
against 79,677-79,770 (median 79,725) at 12, at 188 passes against 220; none refused. Not re-tuned."""


SAMPLERS = tuple(DEFAULTS)


class Sampled(NamedTuple):
    """An arm's best states, their NLL, the initial point's NLL, and the forward-backward passes it ran."""

    log_mu: np.ndarray
    p_binom: np.ndarray
    nll: float
    initial_nll: float
    evaluations: int


class _Carried(torch.autograd.Function):
    """The objective's value as a torch scalar whose backward is the kernel's gradient."""

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
        from sal.backend import Backend
        from sal.emissions import RateConcentrationCountPairEmission
        from sal.opt.hmm import EmissionHmmObjective
        from sal.opt.objective import Restricted, coordinates
        from sal.ragged import Ragged

        k = int(n_states)
        self.n_states = k
        self.lengths = tuple(int(v) for v in np.ravel(lengths))
        data = [np.asarray(v, dtype=np.float64) for v in (total, b, exposure, trials)]
        family = RateConcentrationCountPairEmission(
            np.full(k, 1.0 / float(alpha)), np.ones(k), np.full(k, 0.5),
            np.full(k, float(tau)), np.ones(k), joint=False,
        )  # fmt: skip
        full = EmissionHmmObjective(
            Ragged(np.stack(data[:2], axis=1), self.lengths),
            family,
            covariate=np.stack(data[2:], axis=1),
            backend=Backend.RUST,
        )
        chain = np.full((k, k), (1.0 - stay) / max(k - 1, 1))
        np.fill_diagonal(chain, stay)
        at = full.theta_from_truth(
            np.full(k, 1.0 / k), chain, **family.named_parameters()
        )
        self._sal = Restricted(full, at, coordinates(full, ["mean", "rate"]))
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.evaluations = 0
        self._last: tuple[bytes, float, np.ndarray] | None = None

    def _evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """`(U(x), dU/dx)`: one pass of `sal`'s compiled kernel, or the last pass's where `x` is its point."""
        x = np.array(x, dtype=np.float64)
        key = x.tobytes()
        if self._last is not None and self._last[0] == key:
            return self._last[1], self._last[2]
        value, grad = self._sal.value_and_gradient(torch.from_numpy(x))
        self.evaluations += 1
        self._last = (key, float(value), grad.numpy())
        return self._last[1], self._last[2]

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
        """`U(theta)`, differentiable in `theta` through the kernel's gradient."""
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
