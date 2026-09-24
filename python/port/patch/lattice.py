"""One forward/backward recursion, with the phased state space as an argument.

**#205's second step.** `cnaster` carries the recursion twice.
`hmm_nophasing.forward_lattice` runs a `K`-state chain under a transition
that does not move along it; `hmm_phased.forward_lattice` overrides it to run
a `2K`-state chain under a transition rebuilt per site from the same `K x K`
base and a two-element phase kernel. The same is true of `backward_lattice`.
Between them that is four recursions where the difference is **two
arguments**: how wide the state space is, and whether the transition depends
on the site.

This is the pair they collapse to. `n_states` says which chain is being run
-- `log_emission.shape[0] == n_states` is the unphased one, `== 2 * n_states`
the phased one -- and the site-dependent transition is built into a buffer
the loop reuses rather than into a second function.

**Offered as a simplification, and held to the bar `CLAUDE.md` sets for one:
evidence of equivalence.** All four of `cnaster`'s recursions are reproduced
**bitwise**, which is available here because nothing is reassociated: the
same `logsumexp` runs over the same `buf` in the same order, and the
transition each step reads is the same matrix `cnaster` would have read.
`tests/test_unified_lattice.py` is where that is asserted.

It is not offered as a speedup and does not measure as one. At `K = 7`,
`G = 3,000`, `S = 50`, minimum over the rounds `pytest-benchmark` took:

| chain | pass | `cnaster` | this | ratio |
| --- | --- | ---: | ---: | ---: |
| unphased | forward | 5.098 ms | 4.339 ms | 1.18 |
| unphased | backward | 9.454 ms | 8.350 ms | 1.13 |
| phased | forward | 14.174 ms | 14.345 ms | 0.99 |
| phased | backward | 31.535 ms | 32.743 ms | 0.96 |

**0.96x to 1.18x, and none of it is the claim.** `CLAUDE.md` puts a speedup
at 2x measured at a stress size and this is nowhere near it in either
direction; what the table says is that one implementation for four costs
nothing. The unphased rows gain what the phased rows lose: the transition is
copied into a buffer once instead of being indexed out of `log_transmat` per
step, which helps where the transition is constant and is dead weight where
it is rebuilt per site anyway.

**Who would maintain it.** `snakes_and_ladders` carries this job already, and
carries it further: `oxi_snakes_and_ladders.ragged_posteriors` runs the
ragged recursion in Rust and writes posteriors, transition counts and the
evidence in place. What upstream does **not** carry is a transition that
varies along the chain, which is exactly the phased half here -- `#32`'s
covariate gap in its transition form. So this is `port` code today and
upstream functionality once that lands, and the phased lattice is the
concrete thing that would use it.

**What is deliberately not changed.** `PEANLIZE_PHASE_ONLY_ON_SAME_CNV` is
read from `cnaster` rather than restated, and `update_combined_transmat` is
`cnaster`'s own kernel, imported. A patch that reimplemented the phase kernel
would be comparing two implementations of the kernel as well as two of the
recursion, and the bitwise claim would then be about the wrong thing.
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
"""**Two**, and that is the finding: `forward_lattice` and `backward_lattice` are defined in both, so one recursion here replaces a duplicate pair rather than a module.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

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
    """Whether `cnaster`'s two spot sums are the same floats, on this block.

    The licence for :func:`forward_lattice` initializing both chains one
    way. `numba` compiles the two reductions separately and nothing
    guarantees they associate identically, so this is measured on the shapes
    the recursion sees rather than assumed -- and
    `tests/test_unified_lattice.py` is where it is asserted, so a `numba`
    release that changed it fails there rather than moving a likelihood.
    """
    return bool(np.array_equal(_axis_spot_sum(block), _row_spot_sum(block)))


def is_phased(log_emission: np.ndarray, n_states: int) -> bool:
    """Which chain `log_emission` describes, read off its state axis.

    `n_states` is the copy-state count either way. The phased chain pairs
    each copy state with a phase, so its emission carries `2 * n_states`
    rows, and `cnaster` recovers `n_states` from that by halving -- here it
    is passed, because a recursion that infers its own state space cannot be
    asked to run the unphased chain on an even number of states.
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
    """`log alpha`, for either chain.

    Parameters
    ----------
    lengths : np.ndarray
        Segment lengths, summing to `log_emission.shape[1]`. The recursion
        restarts at each one, which is what makes the genome ragged rather
        than one chain.
    log_transmat : np.ndarray
        The `(n_states, n_states)` copy-state transition. Read directly on
        the unphased chain and as the base of the paired one.
    log_startprob : np.ndarray
        `(n_states,)`. Halved across the phases when `phased`.
    log_emission : np.ndarray
        `(n_paired_states, n_obs, n_spots)`, summed over spots as `cnaster`
        sums it: the spot axis is treated as independent.
    log_sitewise_transmat : np.ndarray
        The per-site switch probability. Read only when `phased`.
    n_states : int
        Copy states, not paired states. See :func:`is_phased`.
    phased : bool
        Whether the transition moves along the chain.

    Returns
    -------
    np.ndarray
        `(n_paired_states, n_obs)`.

    Notes
    -----
    **A hypothesis died here and is recorded rather than carried forward.**
    `cnaster` initializes the two chains differently -- the unphased one sums
    the spot axis with `np.sum(..., axis=1)` over the whole state block, the
    phased one sums one state's row at a time -- and floating-point addition
    is not associative, so an earlier draft kept both forms to protect the
    bitwise claim. It did not need to: under `numba` the two reductions agree
    **bitwise**, contiguous or strided, which
    `tests/test_unified_lattice.py::test_the_two_spot_sums_agree_bitwise`
    measures. One form serves both chains, and the branch that would have
    made this two recursions wearing one name is gone.
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
    # NB accepted and unread, which is `cnaster`'s signature: the two passes
    #    are interchangeable at a call site only if they take the same
    #    arguments. `tests/test_unified_lattice.py` pins that it is unread.
    log_startprob,  # noqa: ARG001
    log_emission,
    log_sitewise_transmat,
    n_states,
    phased,
    penalize_phase_only_on_same_cnv=PEANLIZE_PHASE_ONLY_ON_SAME_CNV,
):
    """`log beta`, for either chain.

    The mirror of :func:`forward_lattice`, and the same two arguments decide
    it. `log_startprob` is accepted and unread on both chains -- `cnaster`'s
    signature, kept so the two are interchangeable at a call site, and
    `tests/test_unified_lattice.py` pins that it is unread rather than
    leaving a reader to infer it from the body.
    """
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
"""The two classes whose `forward_lattice` and `backward_lattice`
:func:`rust_lattices` replaces with `port.oxiport`'s (#318).

**Why a class attribute rather than a `SWAPS` row.** `cnaster` defines the
four recursions as `@staticmethod`s and calls them as `hmmclass.forward_lattice`
or `self.forward_lattice`, so there is no module-level name to rebind. `port`'s
own `hmm_nophasing` (in `SHIFT_SWAPS`) subclasses `cnaster`'s and inherits
whatever the class carries, so installing on these two reaches every caller.
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

    Bitwise `cnaster`'s (`tests/test_rust_lattice.py`), with no compile on
    first call: `hmm_nophasing`'s two are `@njit` without `cache=True`, so
    every process compiled them, 9.75 s of a 40 s dev run (#312, O1).
    Restored on the way out, so a test comparing the two sees both.
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
