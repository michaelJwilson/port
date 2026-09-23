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
    "decode_fixed",
    "decode_shared",
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


def decode_fixed(
    path: np.ndarray,
    bulk: Pseudobulk,
    *,
    n_states: int,
    max_total_copy: int,
    normal: int,
    log_shift: float,
) -> Decoded:
    """Each state's `(A, B)` by the HMM's likelihood, everything else fixed (#362).

    Held: the pinned `mu` and the clone's `log_shift` (so `Z_c` does not
    move with the candidate), the clone's spots, the decoded `path` and the
    fitted dispersions. The likelihood is then a sum over states of terms
    each depending on one state's copies, so the one-candidate-per-state
    MILP separates and its exact solution is each state's argmax. `normal`
    -- the pinned state, shared by every clone -- is `(1, 1)`; a state not
    on `path` is `(1, 1)` too, having no bins to decide it. Each decoded
    state's set is every candidate within `CHI2_HALF` of its best.
    """
    lattice = candidates(max_total_copy)
    log_mu, p = _parameters(lattice)
    copies = np.ones((n_states, 2), dtype=np.int64)
    total = 0.0
    sets: dict[int, list[tuple[int, int]]] = {}

    for state in np.unique(path):
        bins = np.flatnonzero(path == state)
        k = int(state)

        if k == normal:
            pair = np.array([[1, 1]])
            one_mu, one_p = _parameters(pair)
            total += float(
                np.sum(_emission(one_mu[0] - log_shift, one_p[0], bulk, bins))
            )
            continue

        scores = np.array(
            [
                float(np.sum(_emission(log_mu[i] - log_shift, p[i], bulk, bins)))
                for i in range(len(lattice))
            ]
        )
        best = int(np.argmax(scores))
        copies[k] = lattice[best]
        total += float(scores[best])
        sets[k] = [
            (int(a), int(b))
            for (a, b), score in zip(lattice, scores, strict=True)
            if score >= scores[best] - CHI2_HALF
        ]

    return Decoded(copies, total, 1, sets)


def decode_shared(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    *,
    n_states: int,
    max_total_copy: int,
    normal: int,
) -> Decoded:
    """Each state's `(A, B)`, one pair shared by every clone (#362).

    `clones` holds each clone's decoded path, pseudobulk and `log_shift`.
    State `k`'s pair maximizes the likelihood summed over every clone's bins
    in `k`, each clone's rate `log((A + B) / 2) - log_shift`; `normal` is
    `(1, 1)` by definition and a state no clone visits is `(1, 1)` too.
    Everything but the copies is held, so the states separate and the
    one-candidate-per-state MILP is solved exactly, state by state.
    """
    lattice = candidates(max_total_copy)
    log_mu, p = _parameters(lattice)
    copies = np.ones((n_states, 2), dtype=np.int64)
    total = 0.0
    sets: dict[int, list[tuple[int, int]]] = {}
    visited = np.unique(np.concatenate([path for path, _, _ in clones]))

    for state in visited:
        k = int(state)
        members = [
            (np.flatnonzero(path == k), bulk, shift)
            for path, bulk, shift in clones
            if np.any(path == k)
        ]

        def score(i: int, members: list[Any] = members) -> float:
            return sum(
                float(np.sum(_emission(log_mu[i] - shift, p[i], bulk, bins)))
                for bins, bulk, shift in members
            )

        if k == normal:
            one = int(np.flatnonzero((lattice[:, 0] == 1) & (lattice[:, 1] == 1))[0])
            total += score(one)
            continue

        scores = np.array([score(i) for i in range(len(lattice))])
        best = int(np.argmax(scores))
        copies[k] = lattice[best]
        total += float(scores[best])
        sets[k] = [
            (int(a), int(b))
            for (a, b), value in zip(lattice, scores, strict=True)
            if value >= scores[best] - CHI2_HALF
        ]

    return Decoded(copies, total, 1, sets)


def captured_clones() -> list[tuple[np.ndarray, Pseudobulk, float]] | None:
    """Every captured clone's path, pseudobulk and shift, in the fit's order."""
    if not _CAPTURED:
        return None

    single_x, base, total, result = _CAPTURED[0]
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    log_mu = np.asarray(result["new_log_mu"], dtype=np.float64).reshape(-1)
    path = np.asarray(result["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1) % log_mu.size

    try:
        shifts = np.asarray(result["new_log_mu_shift"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError):
        shifts = np.zeros(path.shape[1])

    if shifts.size != path.shape[1]:
        shifts = np.zeros(path.shape[1])

    profile = base.sum(axis=1)
    alpha = float(np.asarray(result["new_alphas"]).reshape(-1)[0])
    tau = float(np.asarray(result["new_taus"]).reshape(-1)[0])
    rows = []

    for clone in range(path.shape[1]):
        spots = assignment == clone
        bulk = Pseudobulk(
            counts_nb=single_x[:, 0, spots].sum(axis=1),
            base_nb_mean=base[:, spots].sum(axis=1),
            counts_bb=single_x[:, 1, spots].sum(axis=1),
            total_bb_rd=total[:, spots].sum(axis=1),
            log_lambda=np.log(profile / profile.sum()),
            alpha=alpha,
            tau=tau,
        )
        rows.append((path[:, clone], bulk, float(shifts[clone])))

    return rows


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
