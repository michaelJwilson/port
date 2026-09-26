"""Integer copies decoded by the likelihood the HMM maximised (#327).

`cnaster` decodes a clone's integer copies with an L1 cost on its fitted
`(mu, p)`. This decodes them with the pseudobulk NB/BB log-likelihood the EM
fitted, on the same counts, the same normal baseline and, when the shift is
on, the same per-clone normalizer `Z_c = logsumexp(log mu[path] + log lambda)`.
The fitted dispersions are held; so is the decoded path -- the previous best
fit's -- so there is no E-step.

A state `k` takes `(A, B)` as `mu_k = (A + B) / 2` and `p_k = A / (A + B)`,
`cnaster`'s convention (`integer_copy.get_acn_baf_rdr` returns the major
allele over the total). The neutral state is pinned at `(1, 1)`, which fixes
the ploidy scale the shifted likelihood cannot. The other states are chosen
by coordinate ascent from a start -- the MILP's answer -- one state at a time
with the rest held, because `Z_c` couples them; it stops when a pass changes
nothing.

Beside each choice it records the likelihood-ratio set: every candidate within
`CHI2_HALF` of the best with the other states held, which is how wide the data
leave a state's copies.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = [
    "CHI2_HALF",
    "Decoded",
    "Pseudobulk",
    "candidates",
    "capture",
    "decode",
    "log_likelihood",
]

CHI2_HALF = 5.991464547107979 / 2
"""Half the 95 per cent point of chi-squared on two degrees of freedom."""


@dataclass
class Pseudobulk:
    """One clone's summed counts and what the fit held fixed."""

    counts_nb: np.ndarray
    base_nb_mean: np.ndarray
    counts_bb: np.ndarray
    total_bb_rd: np.ndarray
    log_lambda: np.ndarray
    alpha: float
    tau: float


@dataclass
class Decoded:
    """Per state `(A, B)`, the log-likelihood reached and each state's set."""

    copies: np.ndarray
    log_likelihood: float
    passes: int
    sets: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


def candidates(max_total_copy: int) -> np.ndarray:
    """Every `(A, B)` with `0 < A + B <= max_total_copy`, `(n, 2)`."""
    return np.array(
        [
            (a, b)
            for a in range(max_total_copy + 1)
            for b in range(max_total_copy + 1)
            if 0 < a + b <= max_total_copy
        ],
        dtype=np.int64,
    )


def _emission(
    log_rate: np.ndarray, p: np.ndarray, bulk: Pseudobulk, bins: np.ndarray
) -> np.ndarray:
    """NB + BB log pmf per bin, in `port.extensions.jax_hmm.emission`'s terms."""
    x = bulk.counts_nb[bins]
    exposure = bulk.base_nb_mean[bins]
    mean = exposure * np.exp(log_rate)
    size = 1.0 / max(bulk.alpha, 1e-10)
    success = 1.0 / (1.0 + bulk.alpha * mean)

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(
            mean <= 0.0,
            0.0,
            gammaln(x + size)
            - gammaln(size)
            - gammaln(x + 1.0)
            + size * np.log(success)
            + x * np.log1p(-success),
        )

    k = bulk.counts_bb[bins]
    n = bulk.total_bb_rd[bins]
    a = np.maximum(p * bulk.tau, 1e-10)
    b = np.maximum((1.0 - p) * bulk.tau, 1e-10)
    allele = (
        gammaln(n + 1.0)
        - gammaln(k + 1.0)
        - gammaln(n - k + 1.0)
        + gammaln(k + a)
        + gammaln(n - k + b)
        - gammaln(n + a + b)
        - (gammaln(a) + gammaln(b) - gammaln(a + b))
    )

    return np.asarray(depth + allele)


def _parameters(copies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = copies.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(total / 2.0), np.where(total > 0, copies[:, 0] / total, 0.5)


def log_likelihood(
    copies: np.ndarray, path: np.ndarray, bulk: Pseudobulk, *, shift: bool
) -> float:
    """The clone's pseudobulk log-likelihood at `copies`, path held."""
    log_mu, p = _parameters(copies)
    rates = log_mu[path]
    offset = float(logsumexp(rates + bulk.log_lambda)) if shift else 0.0
    bins = np.arange(path.size)

    return float(np.sum(_emission(rates - offset, p[path], bulk, bins)))


def decode(
    start: np.ndarray,
    path: np.ndarray,
    bulk: Pseudobulk,
    *,
    max_total_copy: int,
    neutral: int | None,
    shift: bool,
    max_passes: int = 20,
) -> Decoded:
    """Coordinate ascent over the states on `path`, from `start`, neutral pinned."""
    copies = np.array(start, dtype=np.int64).copy()
    lattice = candidates(max_total_copy)
    present = [int(k) for k in np.unique(path)]

    if neutral is not None:
        copies[neutral] = (1, 1)

    free = [k for k in present if k != neutral]
    best = log_likelihood(copies, path, bulk, shift=shift)
    passes = 0

    for passes in range(1, max_passes + 1):  # noqa: B007 -- read after the loop
        changed = False

        for k in free:
            scores = np.empty(len(lattice))

            for i, pair in enumerate(lattice):
                trial = copies.copy()
                trial[k] = pair
                scores[i] = log_likelihood(trial, path, bulk, shift=shift)

            choice = int(np.argmax(scores))

            if scores[choice] > best + 1e-9:
                copies[k] = lattice[choice]
                best = float(scores[choice])
                changed = True

        if not changed:
            break

    sets: dict[int, list[tuple[int, int]]] = {}

    for k in free:
        within = []

        for pair in lattice:
            trial = copies.copy()
            trial[k] = pair
            if log_likelihood(trial, path, bulk, shift=shift) >= best - CHI2_HALF:
                within.append((int(pair[0]), int(pair[1])))

        sets[k] = within

    return Decoded(copies, best, passes, sets)


_CAPTURED: list[tuple[Any, Any, Any, Any]] = []
"""`(single_X, single_base_nb_mean, single_total_bb_RD, result)` of the last fit."""


@contextlib.contextmanager
def capture() -> Iterator[None]:
    """Keep the RDR+BAF fit's inputs and result, for the decoder that follows.

    Wraps `port`'s `run_core_inference`, the one the shift installs, as
    `tests/realizations.py` does, and keeps only the `params="smp"` call.
    """
    import port.patch.hmrf as patch

    original = patch.run_core_inference

    def keep(
        single_x: Any, lengths: Any, base: Any, total: Any, *rest: Any, **kw: Any
    ) -> Any:
        result = original(single_x, lengths, base, total, *rest, **kw)

        if kw.get("params") == "smp":
            _CAPTURED[:] = [
                (np.array(single_x), np.array(base), np.array(total), result)
            ]

        return result

    patch.run_core_inference = keep

    try:
        yield
    finally:
        patch.run_core_inference = original
        _CAPTURED.clear()


def pseudobulk_for(base_column: np.ndarray) -> Pseudobulk | None:
    """The captured clone whose summed baseline is `base_column`, if one is.

    `run_cnaster` hands each clone's decoder its pseudobulk baseline and no
    counts; the clone is identified by that column rather than by an index
    whose order two code paths would have to agree on.
    """
    if not _CAPTURED:
        return None

    single_x, base, total, result = _CAPTURED[0]
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    column = np.asarray(base_column, dtype=np.float64).reshape(-1)

    for clone in np.unique(assignment):
        spots = assignment == clone
        summed = base[:, spots].sum(axis=1)

        if summed.shape == column.shape and np.allclose(summed, column, rtol=1e-9):
            profile = base.sum(axis=1)
            return Pseudobulk(
                counts_nb=single_x[:, 0, spots].sum(axis=1),
                base_nb_mean=summed,
                counts_bb=single_x[:, 1, spots].sum(axis=1),
                total_bb_rd=total[:, spots].sum(axis=1),
                log_lambda=np.log(profile / profile.sum()),
                alpha=float(np.asarray(result["new_alphas"]).reshape(-1)[0]),
                tau=float(np.asarray(result["new_taus"]).reshape(-1)[0]),
            )

    return None
