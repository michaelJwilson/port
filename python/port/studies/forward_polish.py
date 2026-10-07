"""The run's Baum-Welch fit with its EM swapped for L-BFGS on the forward log-likelihood (#748).

`run_cnaster_port --sal` fits the copy states by `cnaster`'s EM: one
`scipy` BFGS over the expected complete-data cost, its posteriors refreshed by
the callback every second iteration, the gradient #433's closed form
(`port.patch.hmm_nophasing.gradient.analytic_bfgs`). `polished` swaps that
`method` alone for `forward_method`, inside the run's own call, so the
emission, the per-clone shift, the free blocks, the transitions, the start
probabilities and the rescoring after the fit are the run's:

- **Objective.** `-log p(x | theta)`, the forward recursion's total over
  segments, at the emission `cnaster`'s `cost_fn` computes at `theta`.
- **Gradient.** #433's closed form (`EmGradient`) read at the posteriors at
  `theta` itself. By Fisher's identity that is the forward log-likelihood's
  gradient, exactly: the EM cost's gradient at its own posteriors. Nothing
  is differenced, and nothing is differentiated through the recursion.
- **Free blocks.** The run's: `log_mu`, `logit p_binom`, and the shared NB
  and beta-binomial dispersions (`shared_*_dispersion`), as
  `cnaster`'s `pack_params` lays them out.
- **Shift.** `S_c` is taken at a hard decode (#362), so the likelihood is
  piecewise in `theta`. The decode is held through a round of L-BFGS, as the
  EM holds its posteriors between refreshes, and taken again at the round's
  end; the fit stops when a round ends at a decode it has held before
  (unchanged, or a cycle), or after `ROUNDS`. `S_c` still moves with `log_mu` inside a round, as the run's
  gradient has it.
- **Optimizer.** `sal.opt.fit.fit`: L-BFGS, strong-Wolfe line search, its
  relative gradient test set to the run's BFGS `gtol` in absolute terms.
  `iterations` caps L-BFGS iterations over all rounds at the run's `maxiter`.

`Polished` counts what each arm spent: value-and-gradient passes and
forward-backward passes, for the run's EM through `analytic_bfgs` as well.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize
import scipy.special
import torch
from sal.opt.termination import Termination

_INNER = 20
"""`sal.opt.fit`'s L-BFGS iterations per step (`sal.opt.fit._INNER_ITERATIONS`)."""

__all__ = ["ROUNDS", "Counts", "forward_method", "lattice_for", "polished"]

ROUNDS = 20
"""Most decodes a forward fit takes the shift at; the EM refreshes its posteriors at most `maxiter / 2` times."""


@dataclass
class Counts:
    """What one fit spent: value-and-gradient passes, forward-backward passes, iterations, rounds.

    `iterations` is BFGS's for the EM; for the forward fit, `sal.opt.fit`'s
    steps, each up to `_INNER` L-BFGS iterations. `passes` compares the two.
    """

    passes: int = 0
    lattices: int = 0
    iterations: int = 0
    rounds: int = 0
    termination: Termination | None = None
    """The last fit's: BFGS's for the EM, the last round's `sal.opt.fit` for the forward fit."""
    trace: list[float] = field(default_factory=list)


class _Forward:
    """`-log p(x | x_packed)` at a held decode, as a `sal.opt.objective.Objective` over `cnaster`'s packed `x`."""

    def __init__(
        self,
        cost: Callable[[np.ndarray], float],
        gradient: Any,
        lattice: Callable[[], tuple[float, np.ndarray]],
        start: np.ndarray,
        counts: Counts,
    ) -> None:
        self._cost, self._gradient, self._lattice = cost, gradient, lattice
        self._start = np.asarray(start, dtype=np.float64)
        self._counts = counts
        self._last: tuple[bytes, float, np.ndarray] | None = None

    def evaluate(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """`(value, gradient)`: the emission at `x`, its posteriors, then #433's gradient at them."""
        x = np.array(x, dtype=np.float64)
        key = x.tobytes()
        if self._last is not None and self._last[0] == key:
            return self._last[1], self._last[2]
        self._cost(x)
        llf, gamma = self._lattice()
        model = self._gradient.model
        model.state_posteriors = gamma
        self._counts.passes += 1
        self._last = (key, -llf, np.asarray(self._gradient(x), dtype=np.float64))
        return self._last[1], self._last[2]

    def initial(self) -> torch.Tensor:
        return torch.from_numpy(self._start.copy())

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.as_tensor(named["x"], dtype=torch.float64)

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, grad = self.evaluate(theta.detach().numpy())
        return torch.tensor(value, dtype=torch.float64), torch.from_numpy(grad.copy())

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        value, grad = self.evaluate(theta.detach().numpy())
        from port.sandbox.extensions.hmm_objective import _Carried

        return _Carried.apply(  # type: ignore[no-any-return]
            theta, torch.tensor(value, dtype=torch.float64), torch.from_numpy(grad)
        )


def lattice_for(
    gradient: Any, lengths: np.ndarray, log_sitewise_transmat: Any, counts: Counts
) -> Callable[[], tuple[float, np.ndarray]]:
    """`(log p(x), posteriors)` at the emission the fit's model holds: one forward and one backward pass.

    The transitions are the fit's own (`get_initial_params` at the model's
    `t`), the start probabilities the model's, which the EM never moves.
    """
    model = gradient.model
    _, _, _, _, _, log_transmat = model.get_initial_params(gradient.n_states, 1)
    ends = np.cumsum(np.asarray(lengths, dtype=np.int64)) - 1

    def lattice() -> tuple[float, np.ndarray]:
        emission, start = model.log_emissions, model.log_startprob
        log_alpha = model.forward_lattice(
            lengths, log_transmat, start, emission, log_sitewise_transmat
        )
        log_beta = model.backward_lattice(
            lengths, log_transmat, start, emission, log_sitewise_transmat
        )
        counts.lattices += 1
        llf = float(np.sum(scipy.special.logsumexp(log_alpha[:, ends], axis=0)))
        log_gamma = log_alpha + log_beta
        log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)
        return llf, np.exp(log_gamma)

    return lattice


def forward_method(
    gradient: Any,
    lengths: np.ndarray,
    log_sitewise_transmat: Any,
    counts: Counts,
) -> Callable[..., scipy.optimize.OptimizeResult]:
    """A `scipy.optimize.minimize` `method` maximizing the forward log-likelihood, in `cnaster`'s EM call.

    `gradient` is the fit's `EmGradient`, its model the fit's; `lengths` and
    `log_sitewise_transmat` the call's, which the method protocol does not
    pass. `fun` is `cnaster`'s `cost_fn`: called at `x` it leaves the
    emission at `x` on the model, shifted at the decode `_decode` returns.
    """
    from sal.opt.fit import fit

    model = gradient.model
    lattice = lattice_for(gradient, lengths, log_sitewise_transmat, counts)

    def method(
        fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **options: Any
    ) -> scipy.optimize.OptimizeResult:
        options.pop("callback", None)
        maxiter = int(options.get("maxiter", 100))
        gtol = float(options.get("gtol", 1e-5))

        def cost(x: np.ndarray) -> float:
            return float(fun(x, *args))

        x = np.asarray(x0, dtype=np.float64)
        # NB the start's posteriors and decode: what the EM's first cost call computes
        cost(x)
        _, gamma = lattice()
        model.state_posteriors = gamma
        value, success = math.nan, False
        steps = max(1, math.ceil(maxiter / _INNER))
        seen: set[bytes] = set()
        while counts.rounds < ROUNDS:
            held = np.asarray(model._decode(), dtype=np.int64)
            seen.add(held.tobytes())
            model._decode = lambda held=held: held
            try:
                objective = _Forward(cost, gradient, lattice, x, counts)
                value0, _ = objective.evaluate(x)
                result = fit(
                    objective,
                    max_iterations=steps,
                    tolerance=gtol / max(1.0, abs(value0)),
                )
                x = result.theta.numpy().copy()
                value, _ = objective.evaluate(x)
            finally:
                del model._decode
            counts.rounds += 1
            counts.iterations += result.iterations
            counts.trace.append(-value)
            success = bool(result.converged)
            counts.termination = result.termination
            # NB a decode seen before ends the fit: unchanged, or a cycle (period 3 on r3 from the run's start)
            if np.asarray(model._decode(), dtype=np.int64).tobytes() in seen:
                break
        return scipy.optimize.OptimizeResult(
            x=x, fun=value, nit=counts.iterations, nfev=counts.passes,
            success=success, message="forward L-BFGS",
        )  # fmt: skip

    return method


def _counted_em(counts: Counts, analytic_bfgs: Any) -> Callable[[Any], Any]:
    """`analytic_bfgs`, counting passes and iterations into `counts`: the EM arm's spend."""

    def wrap(gradient: Any) -> Any:
        inner = analytic_bfgs(gradient)

        def method(
            fun: Any, x0: np.ndarray, args: tuple[Any, ...] = (), **kwargs: Any
        ) -> Any:
            def counted(x: np.ndarray, *a: Any) -> Any:
                counts.passes += 1
                return fun(x, *a)

            callback = kwargs.get("callback")
            if callback is not None:
                model = gradient.model

                def refresh(*a: Any, **k: Any) -> Any:
                    before = model.state_posteriors
                    out = callback(*a, **k)
                    counts.lattices += model.state_posteriors is not before
                    return out

                kwargs["callback"] = refresh
            result = inner(counted, x0, args, **kwargs)
            counts.iterations = int(result.nit)
            counts.termination = Termination.after(
                int(result.nit), converged=bool(result.success)
            )
            return result

        return method

    return wrap


@contextlib.contextmanager
def polished(stage: Any, method: str) -> Iterator[Counts]:
    """Inside, `stage.run` fits by `method`: `"em"`, the run's, or `"forward"`; yields the fit's `Counts`."""
    from port.patch.hmm_nophasing import shifted_emission

    # NB the module's name for the fit's `method`, which `optimize` reads at each call
    emission: Any = shifted_emission

    if method not in ("em", "forward"):
        msg = f"method is 'em' or 'forward', got {method!r}"
        raise ValueError(msg)
    counts = Counts()
    installed = emission.analytic_bfgs
    if method == "em":
        swap = _counted_em(counts, installed)
    else:

        def swap(gradient: Any) -> Any:
            return forward_method(gradient, stage.lengths, stage.args[6], counts)

    emission.analytic_bfgs = swap
    try:
        yield counts
    finally:
        emission.analytic_bfgs = installed
