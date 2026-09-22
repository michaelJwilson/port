"""The two shapes the fit produces, and nothing else (#278).

**`cnaster` supports layouts it cannot produce, and the support is the
defect.** Five sites branch on how `pred_cnv` is laid out and two more guard
a second column of the state parameters; every branch resolves the same way
on every run, because `hmrf_utils.clone_stack_obs` reshapes observations to
`(-1, n_comp, 1)` and `hmm_nophasing.get_initial_params` refuses
`n_spots != 1`.

A patch that carried those branches would carry the ambiguity with them.
These take the shapes the fit produces -- a state parameter is `(n_states,)`
or `(n_states, 1)`, a path is the concatenated genome -- and refuse the rest
once, at the edge, rather than deciding what to do about it at five sites.

## What is deliberately not supported

`deconcatenate_clones` splits `pred_cnv` back out to `(n_obs, n_clones)`, and
upstream's `else: path = array[:, clone]` reads that. It is off by default and
`run_cnaster` never turns it on. Supporting it here would mean every consumer
tests the layout again, which is the thing being removed -- so a
`(n_obs, n_clones)` path is refused rather than quietly flattened row-major
into nonsense.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["clone_path", "clone_paths", "parameter_by_path", "state_vector"]


def state_vector(parameters: Any, name: str = "a state parameter") -> np.ndarray:
    """A fitted state parameter as `(n_states,)`.

    Takes `(n_states,)` or `(n_states, 1)` -- the two shapes the fit
    produces -- and refuses anything else. `#267` established that nothing in
    `cnaster` agrees on what a second column would mean, so the one reading
    that cannot be silently wrong is to stop.

    `name` goes into the message where the caller knows which of the four it
    is holding, so a refusal says what to look at.
    """
    array = np.asarray(parameters)

    if array.ndim == 1:
        return array

    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]

    msg = (
        f"{name} has shape {array.shape}, expected (n_states,) or "
        f"(n_states, 1): the fit produces no other (#278)"
    )
    raise ValueError(msg)


def clone_path(
    pred_cnv: Any, clone: int, n_obs: int, n_states: int | None = None
) -> np.ndarray:
    """One clone's states, sliced out of the concatenated genome.

    `n_states` takes the modulus where the state space is phased, which two
    of upstream's five decodes remember to do.
    """
    array = np.asarray(pred_cnv)

    if array.ndim > 1 and array.shape[1] != 1:
        msg = (
            f"expected a concatenated path, got {array.shape}: "
            f"`deconcatenate_clones` is off on every run `run_cnaster` makes"
        )
        raise ValueError(msg)

    path = array.reshape(-1)[clone * n_obs : (clone + 1) * n_obs]

    return path if n_states is None else path % n_states


def clone_paths(
    pred_cnv: Any, n_clones: int, n_obs: int, n_states: int | None = None
) -> list[np.ndarray]:
    """Every clone's path, in order."""
    return [clone_path(pred_cnv, c, n_obs, n_states) for c in range(n_clones)]


def parameter_by_path(parameters: Any, path: np.ndarray) -> np.ndarray:
    """A fitted parameter read along a state path."""
    read: np.ndarray = state_vector(parameters)[np.asarray(path)]

    return read
