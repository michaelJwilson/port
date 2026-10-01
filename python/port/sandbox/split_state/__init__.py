"""One unbalanced copy state split by depth and refitted on the same clones (#471, #481): sandbox.

**Set aside, not installed by `run_cnaster_port`.** Run it with
`python -m port.sandbox.split_state [run_cnaster_port arguments]`, which
enters :func:`split_state` around the entry point, or enter the context
around `tests.sim_audit.run_arm`.

The read-depth + BAF HMM fits a one-copy loss and a copy-neutral LOH, which
share a BAF, as one state at the LOH's depth, so both decode to one `(A, B)`.
After that fit, :func:`port.sandbox.split_state.split.split_init` splits the
unbalanced state whose bins fall at two depths more than 0.3 apart in log
ratio, frees the closer of the two nearest states for the lower depth, and
the fit is run once more from those rates with `max_iter_outer = 0` on the
clones already found. Clone labels do not change.

The refit is port's own `run_core_inference`, so the neutral pin and the
per-clone shifts (#276, #293) apply to it as to any fit. It runs only where
the first fit was shifted, the call has read depth (`single_base_nb_mean`
nonzero), and the M step fits `m`.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = ["contiguous", "installed", "refit", "split_state", "splits"]

logger = logging.getLogger(__name__)

_INSTALLED = [False]
_SPLITS: list[dict[str, Any]] = []


def installed() -> bool:
    """Whether :func:`split_state` is active."""
    return _INSTALLED[0]


def splits() -> list[dict[str, Any]]:
    """Each split the open block made: the state split, the one freed, and their depths."""
    return list(_SPLITS)


@contextlib.contextmanager
def split_state() -> Iterator[None]:
    """Refit after the read-depth + BAF fit with one state split by depth, for the block.

    Binds its own wrapper around `port.patch.hmrf.run_core_inference`, so
    nothing outside the sandbox names it. Entered before `run_cnaster_port`
    installs its swaps, the wrapped name is the one those swaps resolve.
    """
    from port.patch import hmrf

    previous = _INSTALLED[0]
    fit = hmrf.run_core_inference

    def run_core_inference(*args: Any, **kwargs: Any) -> Any:
        return refit(fit, fit(*args, **kwargs), args, kwargs)

    _INSTALLED[0] = True
    _SPLITS.clear()
    hmrf.run_core_inference = run_core_inference

    try:
        yield
    finally:
        _INSTALLED[0] = previous
        hmrf.run_core_inference = fit


def contiguous(labels: np.ndarray) -> np.ndarray:
    """`labels` renumbered `0..k-1` in their order: the refit's columns, one per clone present.

    A clone the floor emptied leaves a gap, `{0, 1, 2, 4}`; the refit fits one
    column per clone present, so the kept labels have to name those columns
    (#570's `IndexError: index 4 is out of bounds for axis 1 with size 4`).
    """
    _, ranks = np.unique(labels, return_inverse=True)
    return np.asarray(ranks.reshape(labels.shape), dtype=np.asarray(labels).dtype)


def refit(fit: Any, result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """`result`, or the refit from one state split by depth where the call qualifies."""
    from port.patch.hmm_nophasing.shifted_emission import shifted
    from port.patch.hmrf.core_inference import UPSTREAM as upstream
    from port.sandbox.split_state.split import split_init

    # NB `cnaster`'s own signature, held at import: by the call the swaps have
    #    rebound `cnaster.hmrf.run_core_inference` to port's wrapper, whose
    #    own options (`hmm_start`, `distinct_init`) pass through unbound.
    signature = inspect.signature(upstream)
    options = {k: v for k, v in kwargs.items() if k not in signature.parameters}
    bound = signature.bind(*args, **{k: v for k, v in kwargs.items() if k in signature.parameters})
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    base = np.asarray(arguments["single_base_nb_mean"])

    # NB the BAF-only call has no read depth to split by, and an unshifted
    #    fit has no per-clone scale to compare depths across clones.
    if (
        not np.any(base > 0)
        or "m" not in str(arguments.get("params", ""))
        or not shifted(arguments.get("hmmclass"))
    ):
        return result

    init = split_init(result, arguments["single_X"], arguments["lengths"], base)

    if init is None:
        logger.info("split state: no unbalanced state splits by depth")
        return result

    log_mu, p_binom, state, freed = init
    logger.info(
        f"split state: state {state} at log mu {log_mu[state, 0]:.3f}, "
        f"state {freed} freed for {log_mu[freed, 0]:.3f}"
    )
    _SPLITS.append(
        {
            "state": int(state),
            "freed": int(freed),
            "log_mu": (float(log_mu[state, 0]), float(log_mu[freed, 0])),
            "p_binom": float(p_binom[state, 0]),
        }
    )

    kept = contiguous(np.asarray(result["new_assignment"]))
    clones = [np.flatnonzero(kept == label) for label in range(int(kept.max()) + 1)]
    refitted = fit(
        **{
            **arguments,
            **options,
            "initial_clone_index": clones,
            "init_log_mu": log_mu,
            "init_p_binom": p_binom,
            "max_iter_outer": 0,
        }
    )

    locked = bool(getattr(refitted, "_locked", False))

    if locked:
        refitted.unlock()

    try:
        refitted["new_assignment"] = kept
    finally:
        if locked:
            refitted.lock()

    return refitted
