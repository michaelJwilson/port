"""The two shapes the fit produces, and nothing else (#278).

`cnaster` branches on `pred_cnv` layouts it never produces; these accept a
state parameter as `(n_states,)` or `(n_states, 1)` and a path as the
concatenated genome, and refuse the rest (incl. `deconcatenate_clones`'
`(n_obs, n_clones)`, off on every run).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["clone_path", "clone_paths", "parameter_by_path", "state_vector"]


def state_vector(parameters: Any, name: str = "a state parameter") -> np.ndarray:
    """A fitted state parameter as `(n_states,)`; raises ValueError on other shapes.

    `name` labels the parameter in the error message (#267).
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
    """One clone's states from the concatenated genome; `n_states` folds a phased path."""
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
