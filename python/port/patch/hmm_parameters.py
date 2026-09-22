"""The HMM's parameters, named once instead of threaded as a five-tuple.

**#259 stage 3.** `cnaster` passes `(log_startprob, log_mu, p_binom, alphas,
taus)` positionally through `pack_params`, `unpack_params`, `get_bounds`,
both cost functions and the return. Eleven call sites, five elements, no
names: `_, this_log_mu, this_p_binom, this_alphas, this_taus = ...` is what
reading the M step looks like, and a transposition of two `(n_states, 1)`
arrays is a bug no type checker can see.

`Parameters` is that tuple with its names attached. It is deliberately not
more than that: no packing logic, no bounds, no optimizer coupling. Those
live where they already live, and this is what they hand each other.

## On `jax` and `torch`, which is the obvious question

Both have a name for this. `torch.nn.Module.named_parameters` and
`ParameterDict` hold exactly such a bundle; `jax` treats any `NamedTuple` or
dataclass as a pytree, and `jax.flatten_util.ravel_pytree` is `pack_params`
and `unpack_params` in one function, with `jax.grad` giving the M step's
gradient for nothing.

**They are not reached for here, and the reason is the emission, not the
dependency.** `_nb_logpmf_1d` and `_bb_logpmf_1d` are `numba` kernels;
`jax.grad` cannot differentiate through them, so "the gradient for nothing"
costs a reimplementation of the emission in `jax` first -- at which point the
`numba` kernels this repository has measured are no longer what runs. That is
a different project from naming a five-tuple, and it is the one #259 stage 5
derives the gradient by hand to avoid.

A stdlib dataclass gets the naming, the immutability and the type checking
with no dependency and no change to what executes. `CLAUDE.md` asks for
permission before a dependency and evidence that it is needed; this is the
evidence that it is not, for this particular want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Parameters"]


@dataclass(frozen=True)
class Parameters:
    """One HMM parameterization, in `cnaster`'s own order.

    The order is upstream's, so `as_tuple()` is what every `cnaster` call site
    already expects and `of()` takes what they already return. Frozen, because
    the optimizer rebuilds these per iteration and mutating one in place is
    how an iteration's parameters end up in the next one's report.
    """

    log_startprob: np.ndarray
    """`(n_states,)`, normalized; a softmax of the raw vector on unpack."""

    log_mu: np.ndarray
    """`(n_states, 1)`, the log RDR rates. `theta` in the shift's algebra."""

    p_binom: np.ndarray
    """`(n_states, 1)`, the BAF success probabilities."""

    alphas: np.ndarray
    """`(n_states, 1)`, the negative-binomial dispersions."""

    taus: np.ndarray
    """`(n_states, 1)`, the beta-binomial dispersions."""

    @classmethod
    def of(cls, values: tuple[Any, ...]) -> Parameters:
        """From the five-tuple `unpack_params` returns, in its order."""
        if len(values) != _FIELDS:
            msg = f"expected {_FIELDS} parameters, got {len(values)}"
            raise ValueError(msg)

        return cls(*values)

    def as_tuple(self) -> tuple[np.ndarray, ...]:
        """Back to the five-tuple, for a `cnaster` call site that wants one."""
        return (
            self.log_startprob,
            self.log_mu,
            self.p_binom,
            self.alphas,
            self.taus,
        )

    @property
    def n_states(self) -> int:
        """Read off `log_mu`, which carries it in every parameterization."""
        return int(self.log_mu.shape[0])


_FIELDS = 5
"""How many `cnaster` packs, which is what `of` refuses to guess about."""
