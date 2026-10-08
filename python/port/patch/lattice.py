"""Replaces `cnaster`'s four `forward_lattice`/`backward_lattice` with one pair (#205).

`n_states` and `phased` select the unphased (`K`-state) or phased (`2K`,
per-site transition via `cnaster`'s `update_combined_transmat`) chain.
Bitwise `cnaster`'s; a simplification, not a speedup. Also `rust_lattices`,
the `port.oxiport` versions (#318).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from cnaster.hmm_nophasing import numba_logsumexp
from cnaster.hmm_phased import PEANLIZE_PHASE_ONLY_ON_SAME_CNV, update_combined_transmat
from numba import njit

MIRRORS: tuple[str, ...] = (
    "cnaster.hmm_nophasing",
    "cnaster.hmm_phased",
)
"""The `cnaster` modules this stands in for (#250); both define the lattices."""

__all__ = [
    "RUST_LATTICES",
    "backward_lattice",
    "backward_lattice_phased_rust",
    "backward_lattice_rust",
    "forward_lattice",
    "forward_lattice_phased_rust",
    "forward_lattice_rust",
    "is_phased",
    "rust_lattices",
    "spot_sums_agree",
]


@njit(nogil=True, cache=True, error_model="numpy")
def _axis_spot_sum(block):
    """`hmm_nophasing`'s initialization: the whole state block at once."""
    return np.sum(block, axis=1)


@njit(nogil=True, cache=True, error_model="numpy")
def _row_spot_sum(block):
    """`hmm_phased`'s initialization: one state's row at a time."""
    out = np.zeros(block.shape[0])

    for j in range(block.shape[0]):
        out[j] = np.sum(block[j, :])

    return out


def spot_sums_agree(block: np.ndarray) -> bool:
    """Whether `cnaster`'s two spot sums give the same floats on `block` (licenses one init)."""
    return bool(np.array_equal(_axis_spot_sum(block), _row_spot_sum(block)))


def is_phased(log_emission: np.ndarray, n_states: int) -> bool:
    """Whether `log_emission` (`n_states` or `2 * n_states` rows) is the phased chain.

    `n_states` is the copy-state count, passed rather than inferred; raises ValueError otherwise.
    """
    rows = int(log_emission.shape[0])

    if rows == n_states:
        return False

    if rows == 2 * n_states:
        return True

    msg = (
        f"log_emission has {rows} rows, which is neither {n_states} nor {2 * n_states}"
    )
    raise ValueError(msg)


@njit(nogil=True, cache=True, error_model="numpy")
def forward_lattice(
    lengths,
    log_transmat,
    log_startprob,
    log_emission,
    log_sitewise_transmat,
    n_states,
    phased,
    penalize_phase_only_on_same_cnv=PEANLIZE_PHASE_ONLY_ON_SAME_CNV,
):
    """`log alpha`, `(n_paired_states, n_obs)`, for either chain.

    `lengths` are segment lengths (the recursion restarts at each);
    `log_transmat` is `(n_states, n_states)`, the base of the paired one when
    `phased`; `log_startprob` is `(n_states,)`, halved across phases;
    `log_emission` is `(n_paired_states, n_obs, n_spots)`, summed over spots;
    `log_sitewise_transmat` is read only when `phased`. One spot-sum form
    serves both chains (see `spot_sums_agree`).
    """
    n_paired_states, n_obs, _ = log_emission.shape

    log_alpha = np.zeros((n_paired_states, n_obs))
    buf = np.zeros(n_paired_states)
    transition = np.empty((n_paired_states, n_paired_states))

    log_half = np.log(0.5)

    if phased:
        combined_start = log_half + np.append(log_startprob, log_startprob)
        log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))
    else:
        combined_start = log_startprob
        log_sitewise_self_transmat = log_sitewise_transmat

        for i in range(n_paired_states):
            for j in range(n_paired_states):
                transition[i, j] = log_transmat[i, j]

    cumlen = 0

    for le in lengths:
        log_alpha[:, cumlen] = combined_start + np.sum(
            log_emission[:, cumlen, :], axis=1
        )

        for t in range(1, le):
            idx = cumlen + t - 1

            if phased:
                update_combined_transmat(
                    transition,
                    n_states,
                    log_transmat,
                    log_sitewise_self_transmat[idx],
                    log_sitewise_transmat[idx],
                    penalize_phase_only_on_same_cnv,
                    log_half,
                )

            for j in range(n_paired_states):
                for i in range(n_paired_states):
                    buf[i] = log_alpha[i, idx] + transition[i, j]

                log_alpha[j, cumlen + t] = numba_logsumexp(buf) + np.sum(
                    log_emission[j, cumlen + t, :]
                )

        cumlen += le

    return log_alpha


@njit(nogil=True, cache=True, error_model="numpy")
def backward_lattice(
    lengths,
    log_transmat,
    # NB unread, as in `cnaster`'s signature.
    log_startprob,  # noqa: ARG001
    log_emission,
    log_sitewise_transmat,
    n_states,
    phased,
    penalize_phase_only_on_same_cnv=PEANLIZE_PHASE_ONLY_ON_SAME_CNV,
):
    """`log beta`, for either chain; mirror of :func:`forward_lattice`."""
    n_paired_states, n_obs, _ = log_emission.shape

    log_beta = np.zeros((n_paired_states, n_obs))
    buf = np.zeros(n_paired_states)
    transition = np.empty((n_paired_states, n_paired_states))

    log_half = np.log(0.5)

    if phased:
        log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))
    else:
        log_sitewise_self_transmat = log_sitewise_transmat

        for i in range(n_paired_states):
            for j in range(n_paired_states):
                transition[i, j] = log_transmat[i, j]

    cumlen = 0

    for le in lengths:
        log_beta[:, cumlen + le - 1] = 0.0

        for t in range(le - 2, -1, -1):
            idx = cumlen + t

            if phased:
                update_combined_transmat(
                    transition,
                    n_states,
                    log_transmat,
                    log_sitewise_self_transmat[idx],
                    log_sitewise_transmat[idx],
                    penalize_phase_only_on_same_cnv,
                    log_half,
                )

            for i in range(n_paired_states):
                for j in range(n_paired_states):
                    buf[j] = (
                        log_beta[j, cumlen + t + 1]
                        + transition[i, j]
                        + np.sum(log_emission[j, cumlen + t + 1, :])
                    )

                log_beta[i, cumlen + t] = numba_logsumexp(buf)

        cumlen += le

    return log_beta


RUST_LATTICES: tuple[tuple[str, str], ...] = (
    ("cnaster.hmm_nophasing", "hmm_nophasing"),
    ("cnaster.hmm_phased", "hmm_phased"),
)
"""The classes whose `@staticmethod` lattices :func:`rust_lattices` replaces (#318).

A class attribute, not a `SWAPS` row: `cnaster` calls them via the class.
"""


def _lengths(lengths: Any) -> np.ndarray:
    return np.ascontiguousarray(lengths, dtype=np.int64)


def _f64(array: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def forward_lattice_rust(
    lengths: Any,
    log_transmat: Any,
    log_startprob: Any,
    log_emission: Any,
    log_sitewise_transmat: Any,  # noqa: ARG001 -- cnaster's signature
) -> np.ndarray:
    """`cnaster.hmm_nophasing.hmm_nophasing.forward_lattice`, from Rust."""
    from port import oxiport

    return oxiport.forward_lattice(
        _lengths(lengths), _f64(log_transmat), _f64(log_startprob), _f64(log_emission)
    )


def backward_lattice_rust(
    lengths: Any,
    log_transmat: Any,
    log_startprob: Any,  # noqa: ARG001 -- cnaster's signature, unread there too
    log_emission: Any,
    log_sitewise_transmat: Any,  # noqa: ARG001 -- cnaster's signature
) -> np.ndarray:
    """`cnaster.hmm_nophasing.hmm_nophasing.backward_lattice`, from Rust."""
    from port import oxiport

    return oxiport.backward_lattice(
        _lengths(lengths), _f64(log_transmat), _f64(log_emission)
    )


def forward_lattice_phased_rust(
    lengths: Any,
    log_transmat: Any,
    log_startprob: Any,
    log_emission: Any,
    log_sitewise_transmat: Any,
    penalize_phase_only_on_same_cnv: bool = PEANLIZE_PHASE_ONLY_ON_SAME_CNV,
) -> np.ndarray:
    """`cnaster.hmm_phased.hmm_phased.forward_lattice`, from Rust."""
    from port import oxiport

    return oxiport.forward_lattice_phased(
        _lengths(lengths),
        _f64(log_transmat),
        _f64(log_startprob),
        _f64(log_emission),
        _f64(log_sitewise_transmat),
        bool(penalize_phase_only_on_same_cnv),
    )


def backward_lattice_phased_rust(
    lengths: Any,
    log_transmat: Any,
    log_startprob: Any,  # noqa: ARG001 -- cnaster's signature, unread there too
    log_emission: Any,
    log_sitewise_transmat: Any,
    penalize_phase_only_on_same_cnv: bool = PEANLIZE_PHASE_ONLY_ON_SAME_CNV,
) -> np.ndarray:
    """`cnaster.hmm_phased.hmm_phased.backward_lattice`, from Rust."""
    from port import oxiport

    return oxiport.backward_lattice_phased(
        _lengths(lengths),
        _f64(log_transmat),
        _f64(log_emission),
        _f64(log_sitewise_transmat),
        bool(penalize_phase_only_on_same_cnv),
    )


_RUST = {
    "hmm_nophasing": (forward_lattice_rust, backward_lattice_rust),
    "hmm_phased": (forward_lattice_phased_rust, backward_lattice_phased_rust),
}


@contextmanager
def rust_lattices() -> Iterator[None]:
    """Replace `cnaster`'s four lattices with `port.oxiport`'s for the block.

    Bitwise `cnaster`'s, without the per-process `@njit` compile (#312);
    restored on exit.
    """
    import importlib

    undo: list[tuple[type, str, Any]] = []

    try:
        for module_name, class_name in RUST_LATTICES:
            cls = getattr(importlib.import_module(module_name), class_name)
            forward, backward = _RUST[class_name]

            for name, replacement in (
                ("forward_lattice", forward),
                ("backward_lattice", backward),
            ):
                undo.append((cls, name, cls.__dict__[name]))
                setattr(cls, name, staticmethod(replacement))

        yield
    finally:
        for cls, name, original in reversed(undo):
            setattr(cls, name, original)
