"""How `pred_cnv` is laid out, decided once (#278).

**Four live sites re-derive this, and they do not agree on what to call it.**
`plot_genomic.py:277, 355, 542, 650` and `plot_loh_density.py:218` each test
`pred_cnv.ndim == 1` (two of them also `shape[1] == 1`) and then slice either
`[c * n_obs : (c + 1) * n_obs]` or `[:, c]`. Two of the five take the modulus
by `n_states` and three do not, which is the kind of divergence a repeated
derivation produces and a shared one cannot.

`cnaster` writes `pred_cnv` concatenated along the genomic axis when clones
are stacked -- which `hmrf_utils.clone_stack_obs` always does -- and as
`(n_obs, n_clones)` when `deconcatenate_clones` splits it back out. Both
shapes are live, so this reads the layout rather than assuming one.

## The state parameters have one column, and that is not a special case

`plot_genomic.py:646` and `plot_loh_density.py:221` guard the clone index
with `0 if shape[1] == 1 else c`. The fit cannot return more than one column
-- `clone_stack_obs` reshapes observations to `(-1, n_comp, 1)` and
`get_initial_params` refuses `n_spots != 1` -- so the guard resolves to `0`
on every run `cnaster` performs. These functions take the contract instead of
testing for it, and say so when it is broken rather than silently indexing
column zero of an array that has five.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["clone_column", "clone_path", "clone_paths"]


def clone_column(parameters: np.ndarray) -> int:
    """The only column a fitted state parameter has, checked rather than assumed.

    Returns `0`. It takes an argument so that the assumption is *tested* at
    every call site that used to branch, and a `cnaster` that starts
    producing per-clone parameters is a loud failure here rather than a
    silent read of the first column.
    """
    array = np.asarray(parameters)

    if array.ndim != 2 or array.shape[1] != 1:
        msg = (
            f"expected a fitted state parameter of shape (n_states, 1), got "
            f"{array.shape}. `clone_stack_obs` reshapes observations to "
            f"(-1, n_comp, 1) and `get_initial_params` refuses n_spots != 1, "
            f"so a second column means the fit changed and every consumer "
            f"that indexes column 0 is now wrong (#278)."
        )
        raise ValueError(msg)

    return 0


def clone_path(
    pred_cnv: np.ndarray, clone: int, n_obs: int, n_states: int | None = None
) -> np.ndarray:
    """One clone's state path, whichever layout `pred_cnv` arrived in.

    `n_states` takes the modulus when given, which is what the phased state
    space needs and what two of the five call sites remembered to do.
    """
    array = np.asarray(pred_cnv)

    if array.ndim == 1 or array.shape[1] == 1:
        path = array.reshape(-1)[clone * n_obs : (clone + 1) * n_obs]
    else:
        path = array[:, clone]

    return path if n_states is None else path % n_states


def clone_paths(
    pred_cnv: np.ndarray, n_clones: int, n_obs: int, n_states: int | None = None
) -> list[np.ndarray]:
    """Every clone's path, in order."""
    return [clone_path(pred_cnv, c, n_obs, n_states) for c in range(n_clones)]


def parameter_by_path(
    parameters: np.ndarray, path: np.ndarray, column: int | None = None
) -> Any:
    """A fitted parameter read along a state path.

    `p_binom[c_pred, c if p_binom.shape[1] > 1 else 0]` written as what it
    is. The column defaults to the one `clone_column` guarantees.
    """
    array = np.asarray(parameters)
    index = clone_column(array) if column is None else column

    return array[np.asarray(path), index]
