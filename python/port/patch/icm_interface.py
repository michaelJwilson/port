"""The label solver's interface, reduced to what it reads.

Issue #59 item 5. `cnaster.icm.icm_sweep_deque` takes fifteen parameters.
Seven of them are not information the solver needs:

| Parameter | Why it goes |
| --- | --- |
| `posterior` | Dead. Its only use in the live body is commented out, and the call site passes `None` -- to a **required positional**. |
| `log_persample_weights`, `sample_ids` | A per-`(spot, clone)` constant added inside the inner loop. Folds into the field. |
| `onehot_allowed_clones` | Sets a cost to `-inf`. Folds into the field, as `-inf`. |
| `temp` | Used once, as `spatial_weight / temp`. Folds into the coupling. |
| `adj_indptr`, `adj_indices`, `adj_weights` | One graph in three arrays, passed and re-passed as three. |

What is left is the problem: a unary field, a weighted graph, a coupling, a
starting labelling, and the solver's own stopping and exploration knobs.

    icm_sweep(field, graph, assignment, beta, *, tol, epsilon, min_clone_spots)

Eight parameters, of which four are the problem and four have defaults.

**This is offered as a simplification, not a speedup.** `CLAUDE.md` splits
the two, and the evidence here is the bitwise equivalence in
`tests/test_icm_interface.py`. The fold does turn one indexed add per clone
per spot *visit* into one vectorized add per spot *sweep*, and that measures
1.19x at 400 spots and 1.38x at 20,000 -- below the 2x bar, so it is
reported rather than claimed.

**Which `icm_sweep_deque`.** `cnaster.icm` defines that name four times, at
lines 363, 513, 658 and 807; only the last survives the module body. The
first takes a COO triple, the last takes CSR, and the commented-out block at
`hmrf.py:294` still calls the COO form. This module wraps the live one, and
`tests/test_icm_interface.py` pins which that is -- so a reordering of
`icm.py` that changed the winner would fail here rather than silently change
the solver.

`#8`'s upstream correspondence reaches the same reduction from the other
side: `log_persample_weights`, `onehot_allowed_clones` and `temp` fold into
the field and the couplings there too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIRRORS: tuple[str, ...] = ("cnaster.icm",)
"""`icm.icm_sweep_deque`, reduced to the problem it solves.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

__all__ = ["CsrGraph", "IcmResult", "fold_unary", "icm_sweep"]


@dataclass(frozen=True)
class CsrGraph:
    """One spatial graph, passed as one argument.

    `cnaster` carries the same graph as three arrays through every call in
    the boundary, and as a COO triple beside them (issue #59 item 3). The
    three are meaningless apart -- an `indices` without its `indptr` cannot
    be read -- so they travel together.
    """

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_matrix(cls, adjacency_mat: object) -> CsrGraph:
        """From the `scipy.sparse` CSR matrix the boundary already holds.

        No conversion and no copy: `hmrf.py:309-311` passes these same three
        attributes, so this is the call site's own expression named once.
        """
        return cls(
            indptr=adjacency_mat.indptr,  # type: ignore[attr-defined]
            indices=adjacency_mat.indices,  # type: ignore[attr-defined]
            weights=adjacency_mat.data,  # type: ignore[attr-defined]
        )

    @property
    def n_spots(self) -> int:
        """Nodes, which is `len(indptr) - 1`."""
        return int(self.indptr.shape[0] - 1)


@dataclass(frozen=True)
class IcmResult:
    """What the sweep returns, named.

    `cnaster` returns a bare `(niter, cost)` tuple, and the call site unpacks
    it positionally into `niter, new_cost`.
    """

    niter: int
    cost: float


def fold_unary(
    single_llf: np.ndarray,
    log_persample_weights: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    onehot_allowed_clones: np.ndarray | None = None,
) -> np.ndarray:
    """Fold the solver's per-`(spot, clone)` constants into the field.

    Both terms are added to `single_llf[i, c]` inside the inner loop, once
    per visit to spot `i`. They depend on `(i, c)` and nothing the sweep
    changes, so adding them once to the field gives the solver the same
    numbers with two fewer arguments and two fewer branches per visit.

    Parameters
    ----------
    single_llf : np.ndarray
        The field, shape `(n_spots, n_clones)`. Not modified.
    log_persample_weights : np.ndarray or None
        Shape `(n_clones, n_samples)`, indexed `[c, sample_ids[i]]`.
        Requires `sample_ids`.
    onehot_allowed_clones : np.ndarray or None
        Boolean, shape `(n_spots, n_clones)`. Disallowed entries become
        `-inf`.

    Returns
    -------
    np.ndarray
        A new `(n_spots, n_clones)` field, `float64`.

    Notes
    -----
    **Bitwise, not approximately.** `cnaster` computes
    `single_llf[i, c] + log_persample_weights[c, s] + w_edge[c] * beta`, left
    to right. Folding the first two moves no arithmetic: the same two floats
    are added first, and the edge term is added to the same intermediate. The
    mask is exact for the same reason -- `-inf + finite` is `-inf`, so
    masking before the edge term and after it agree.

    Raises
    ------
    ValueError
        If `log_persample_weights` is given without `sample_ids`, where the
        rows could not be selected, or if a shape cannot index the field.
    """
    field = np.array(single_llf, dtype=np.float64, copy=True)
    n_spots, n_clones = field.shape

    if log_persample_weights is not None:
        if sample_ids is None:
            msg = "log_persample_weights requires sample_ids"
            raise ValueError(msg)
        if log_persample_weights.shape[0] != n_clones:
            msg = (
                f"log_persample_weights has {log_persample_weights.shape[0]} clones, "
                f"the field has {n_clones}"
            )
            raise ValueError(msg)

        field += log_persample_weights[:, np.asarray(sample_ids)[:n_spots]].T

    if onehot_allowed_clones is not None:
        if onehot_allowed_clones.shape != field.shape:
            msg = (
                f"onehot_allowed_clones is {onehot_allowed_clones.shape}, "
                f"the field is {field.shape}"
            )
            raise ValueError(msg)

        field[~np.asarray(onehot_allowed_clones, dtype=bool)] = -np.inf

    return field


def icm_sweep(
    field: np.ndarray,
    graph: CsrGraph,
    assignment: np.ndarray,
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
) -> IcmResult:
    """Run `cnaster`'s live sweep through the reduced interface.

    A delegation, not a reimplementation: the solver is `cnaster`'s, so the
    comparison this patch has to survive is bitwise against calling it
    directly. Rewriting the sweep would put a second implementation in the
    way of that.

    Parameters
    ----------
    field : np.ndarray
        The unary cost per `(spot, clone)`, with the per-sample weights and
        the allowed-clone mask already folded in by :func:`fold_unary`.
    beta : float
        The spatial coupling, `spatial_weight / temp` in `cnaster`'s terms.
        One number because the solver uses one number.
    assignment : np.ndarray
        The starting labelling, **updated in place** -- `cnaster`'s
        behaviour, kept rather than hidden, because the call site reads the
        array afterwards rather than a return value.

    Notes
    -----
    The sweep draws from the legacy global `numpy` RNG (`np.random.shuffle`
    for the queue order, and `np.random.rand`/`choice` under `epsilon` and
    `min_clone_spots`). That is `cnaster`'s, and it is neither seeded nor
    threaded through a `Generator` -- so two runs of the same problem differ.
    Stated rather than fixed: seeding it here would be a behaviour change
    hiding inside an interface change.
    """
    from cnaster.icm import icm_sweep_deque

    niter, cost = icm_sweep_deque(
        single_llf=field,
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.weights,
        new_assignment=assignment,
        spatial_weight=beta,
        posterior=None,
        onehot_allowed_clones=None,
        tol=tol,
        log_persample_weights=None,
        sample_ids=None,
        cost_zeropoint=cost_zeropoint,
        temp=1.0,
        min_clone_spots=min_clone_spots,
        epsilon=epsilon,
    )

    return IcmResult(niter=int(niter), cost=float(cost))
