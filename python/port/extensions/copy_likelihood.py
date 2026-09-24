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
    "IntegerFit",
    "Pseudobulk",
    "TemperedFit",
    "candidates",
    "capture",
    "decode",
    "decode_fixed",
    "decode_shared",
    "integer_em",
    "log_likelihood",
    "rounded_start",
    "tempered_em",
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


def _parameters(
    copies: np.ndarray, purity: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """`(log mu, p)` of each pair, in a spot `purity` tumour and the rest normal.

    Depth `purity (A + B) / 2 + (1 - purity)`; allele share
    `(purity A + 1 - purity) / (purity (A + B) + 2 (1 - purity))`, 0.5 where
    there are no copies at all.
    """
    total = copies.sum(axis=1).astype(np.float64)
    depth = purity * total / 2.0 + (1.0 - purity)
    alleles = purity * total + 2.0 * (1.0 - purity)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (purity * copies[:, 0] + (1.0 - purity)) / alleles
        return np.log(depth), np.where(alleles > 0, share, 0.5)


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
    distinct: bool = False,
    purity: list[float] | None = None,
) -> Decoded:
    """Each state's `(A, B)`, one pair shared by every clone (#362).

    `clones` holds each clone's decoded path, pseudobulk and `log_shift`.
    State `k`'s pair maximizes the likelihood summed over every clone's bins
    in `k`, each clone's rate `log((A + B) / 2) - log_shift`; `normal` is
    `(1, 1)` by definition and a state no clone visits is `(1, 1)` too.
    Everything but the copies is held, so the states separate and the
    one-candidate-per-state MILP is solved exactly, state by state. With
    `distinct`, no two states share a pair: the assignment of pairs to states
    maximizing the summed likelihood, `(1, 1)` the normal state's alone.
    `purity`, one per clone, scores each clone's pair at its tumour
    fraction (:func:`_parameters`); pure where not given.
    """
    lattice = candidates(max_total_copy)
    fractions = [1.0] * len(clones) if purity is None else list(purity)
    rates = [_parameters(lattice, f) for f in fractions]
    copies = np.ones((n_states, 2), dtype=np.int64)
    total = 0.0
    sets: dict[int, list[tuple[int, int]]] = {}
    table: dict[int, np.ndarray] = {}
    visited = np.unique(np.concatenate([path for path, _, _ in clones]))

    for state in visited:
        k = int(state)
        members = [
            (np.flatnonzero(path == k), bulk, shift, rate)
            for (path, bulk, shift), rate in zip(clones, rates, strict=True)
            if np.any(path == k)
        ]

        def score(i: int, members: list[Any] = members) -> float:
            return sum(
                float(np.sum(_emission(mu[i] - shift, p[i], bulk, bins)))
                for bins, bulk, shift, (mu, p) in members
            )

        if k == normal:
            one = int(np.flatnonzero((lattice[:, 0] == 1) & (lattice[:, 1] == 1))[0])
            total += score(one)
            continue

        scores = np.array([score(i) for i in range(len(lattice))])
        table[k] = scores
        best = int(np.argmax(scores))
        copies[k] = lattice[best]
        sets[k] = [
            (int(a), int(b))
            for (a, b), value in zip(lattice, scores, strict=True)
            if value >= scores[best] - CHI2_HALF
        ]

    if distinct and table:
        from scipy.optimize import linear_sum_assignment

        states = sorted(table)
        one = (lattice[:, 0] == 1) & (lattice[:, 1] == 1)
        matrix = np.array([np.where(one, -np.inf, table[k]) for k in states])
        rows, columns = linear_sum_assignment(
            np.where(np.isfinite(matrix), -matrix, 1e300)
        )

        for row, column in zip(rows, columns, strict=True):
            copies[states[row]] = lattice[column]

    total += sum(
        float(table[k][int(np.flatnonzero((lattice == copies[k]).all(axis=1))[0])])
        for k in table
    )

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


@dataclass
class IntegerFit:
    """The integer EM's fixed point: shared copies, per-clone paths and shifts."""

    copies: np.ndarray
    paths: list[np.ndarray]
    shifts: np.ndarray
    alpha: float
    tau: float
    log_likelihood: float
    iterations: int
    trace: list[float] = field(default_factory=list)
    purity: np.ndarray = field(default_factory=lambda: np.ones(0))


def _with(bulk: Pseudobulk, alpha: float, tau: float) -> Pseudobulk:
    return Pseudobulk(
        bulk.counts_nb,
        bulk.base_nb_mean,
        bulk.counts_bb,
        bulk.total_bb_rd,
        bulk.log_lambda,
        alpha,
        tau,
    )


def rounded_start(
    log_mu: np.ndarray, p_binom: np.ndarray, normal: int, max_total_copy: int
) -> np.ndarray:
    """Each state's `(A, B)` from the continuous fit, `normal` at `(1, 1)`.

    The total is `2 exp(mu_k - mu_normal)` rounded, clipped to
    `[1, max_total_copy]`; `A` is `p_k` of it, rounded.
    """
    total = np.clip(
        np.rint(2.0 * np.exp(log_mu - log_mu[normal])), 1, max_total_copy
    ).astype(np.int64)
    major = np.clip(np.rint(p_binom * total), 0, total).astype(np.int64)
    copies = np.stack([major, total - major], axis=1)
    copies[normal] = (1, 1)
    return copies


def _viterbi(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """`(n_states, n_obs)` emissions; the best path, restarted at each length."""
    path = np.empty(log_emission.shape[1], dtype=np.int64)
    total = 0.0
    start = 0

    for length in np.asarray(lengths, dtype=np.int64):
        stop = start + int(length)
        delta = log_startprob + log_emission[:, start]
        back = np.empty((stop - start, log_transmat.shape[0]), dtype=np.int64)

        for t in range(start + 1, stop):
            scores = delta[:, None] + log_transmat
            back[t - start] = np.argmax(scores, axis=0)
            delta = scores[back[t - start], np.arange(scores.shape[1])]
            delta = delta + log_emission[:, t]

        path[stop - 1] = int(np.argmax(delta))
        total += float(np.max(delta))

        for t in range(stop - 1, start, -1):
            path[t - 1] = back[t - start, path[t]]

        start = stop

    return path, total


def _negative_shifted(
    shift: float, log_rate: np.ndarray, p: np.ndarray, bulk: Pseudobulk
) -> float:
    return -float(np.sum(_emission(log_rate - shift, p, bulk, np.arange(p.size))))


def _negative_joint(
    log_value: float,
    which: str,
    other: float,
    paths: list[np.ndarray],
    bulks: list[Pseudobulk],
    shifts: np.ndarray,
    copies: np.ndarray,
    purity: np.ndarray,
) -> float:
    """Minus the joint log-likelihood at `exp(log_value)` for `which` dispersion."""
    value = float(np.exp(log_value))
    alpha, tau = (value, other) if which == "alpha" else (other, value)
    total = 0.0

    for z, b, s, f in zip(paths, bulks, shifts, purity, strict=True):
        log_mu, p = _parameters(copies, float(f))
        bins = np.arange(z.size)
        total += float(
            np.sum(_emission(log_mu[z] - s, p[z], _with(b, alpha, tau), bins))
        )

    return -total


def _negative_purity(
    purity: float, copies: np.ndarray, z: np.ndarray, shift: float, bulk: Pseudobulk
) -> float:
    log_mu, p = _parameters(copies, purity)
    return -float(np.sum(_emission(log_mu[z] - shift, p[z], bulk, np.arange(z.size))))


def integer_em(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    start: np.ndarray,
    *,
    normal_clone: int,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
    max_total_copy: int,
    normal: int,
    zero_normal: bool = True,
    max_iter: int = 20,
    max_inner: int = 10,
    fit_purity: bool = False,
) -> IntegerFit:
    """EM over integer states: the copies are the M-step's parameters (#362).

    M-step, each clone's path held: each clone's `logmu_shift` (the normal
    clone's held at 0 under `zero_normal`), then the shared dispersions
    `alpha` and `tau`, then each state's `(A, B)`, shared by every clone
    (:func:`decode_shared`), `normal` at `(1, 1)`. Each state's `mu` and
    `p` are those of its pair. E-step: each clone's Viterbi path under them,
    with the fit's transitions. It starts from `start` and the continuous
    fit's paths, shifts and dispersions, and stops when an iteration changes
    no path and no pair.

    With `fit_purity` each tumour clone also carries a tumour fraction,
    fitted beside its shift in `[0.05, 1]` from 1: its spots are that
    fraction tumour and the rest normal (:func:`_parameters`). The normal
    clone's is 1, having no tumour to dilute.
    """
    from scipy.optimize import minimize_scalar

    n_states = start.shape[0]
    copies = np.array(start, dtype=np.int64)
    copies[normal] = (1, 1)
    paths = [np.asarray(path, dtype=np.int64) for path, _, _ in clones]
    bulks = [bulk for _, bulk, _ in clones]
    shifts = np.array([shift for _, _, shift in clones], dtype=np.float64)
    alpha, tau = bulks[0].alpha, bulks[0].tau
    purity = np.ones(len(clones))
    trace: list[float] = []
    total = -np.inf
    iteration = 0

    def fitted(a: float, t: float) -> list[Pseudobulk]:
        return [_with(bulk, a, t) for bulk in bulks]

    for iteration in range(1, max_iter + 1):  # noqa: B007 -- read after the loop
        updated = copies.copy()

        for _ in range(max_inner):  # the M-step to its own fixed point
            previous = (updated.copy(), shifts.copy(), purity.copy())
            copies_m = updated

            for i, (z, bulk) in enumerate(zip(paths, fitted(alpha, tau), strict=True)):
                if fit_purity and i != normal_clone:
                    purity[i] = float(
                        minimize_scalar(
                            _negative_purity,
                            bounds=(0.05, 1.0),
                            args=(copies_m, z, float(shifts[i]), bulk),
                            method="bounded",
                        ).x
                    )

                if zero_normal and i == normal_clone:
                    shifts[i] = 0.0
                    continue

                log_mu, p = _parameters(copies_m, float(purity[i]))
                shifts[i] = float(
                    minimize_scalar(
                        _negative_shifted,
                        bounds=(shifts[i] - 3.0, shifts[i] + 3.0),
                        args=(log_mu[z], p[z], bulk),
                        method="bounded",
                    ).x
                )

            alpha = float(
                np.exp(
                    minimize_scalar(
                        _negative_joint,
                        bounds=(np.log(alpha) - 5.0, np.log(alpha) + 5.0),
                        args=("alpha", tau, paths, bulks, shifts, copies_m, purity),
                        method="bounded",
                    ).x
                )
            )
            tau = float(
                np.exp(
                    minimize_scalar(
                        _negative_joint,
                        bounds=(np.log(tau) - 5.0, np.log(tau) + 5.0),
                        args=("tau", alpha, paths, bulks, shifts, copies_m, purity),
                        method="bounded",
                    ).x
                )
            )

            members = [
                (z, bulk, float(s))
                for z, bulk, s in zip(paths, fitted(alpha, tau), shifts, strict=True)
            ]
            shared = decode_shared(
                members,
                n_states=n_states,
                max_total_copy=max_total_copy,
                normal=normal,
                distinct=True,
                purity=purity.tolist(),
            )
            visited = np.unique(np.concatenate(paths))
            updated = copies_m.copy()
            updated[visited] = shared.copies[visited]

            if (
                np.array_equal(updated, previous[0])
                and np.allclose(shifts, previous[1], atol=1e-4)
                and np.allclose(purity, previous[2], atol=1e-4)
            ):
                break

        new_paths = []
        total = 0.0

        for (z, bulk, s), f in zip(members, purity, strict=True):
            log_mu, p = _parameters(updated, float(f))
            bins = np.arange(z.size)
            emission = np.stack(
                [_emission(log_mu[k] - s, p[k], bulk, bins) for k in range(n_states)]
            )
            path, score = _viterbi(emission, log_transmat, log_startprob, lengths)
            new_paths.append(path)
            total += score

        trace.append(total)
        unchanged = np.array_equal(updated, copies) and all(
            np.array_equal(a, b) for a, b in zip(paths, new_paths, strict=True)
        )
        copies, paths = updated, new_paths

        if unchanged:
            break

    return IntegerFit(
        copies, paths, shifts, alpha, tau, total, iteration, trace, purity
    )


def _forward_backward(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """`(n_states, n_obs)` responsibilities with everything scaled by `1 / T`.

    At `T = 1` the HMM's posteriors; as `T -> 0` they concentrate on the
    Viterbi path, one state per bin.
    """
    inverse = 1.0 / temperature
    emission = log_emission * inverse
    transmat = log_transmat * inverse
    start_prob = log_startprob * inverse
    gamma = np.empty_like(emission)
    begin = 0

    for length in np.asarray(lengths, dtype=np.int64):
        stop = begin + int(length)
        n = stop - begin
        forward = np.empty((n, emission.shape[0]))
        backward = np.zeros((n, emission.shape[0]))
        forward[0] = start_prob + emission[:, begin]

        for t in range(1, n):
            forward[t] = (
                logsumexp(forward[t - 1][:, None] + transmat, axis=0)
                + emission[:, begin + t]
            )

        for t in range(n - 2, -1, -1):
            backward[t] = logsumexp(
                transmat + (emission[:, begin + t + 1] + backward[t + 1])[None, :],
                axis=1,
            )

        joint = forward + backward
        gamma[:, begin:stop] = np.exp(joint - logsumexp(joint, axis=1)[:, None]).T
        begin = stop

    return gamma


@dataclass
class TemperedFit:
    """The tempered EM over every `(A, B)`: pairs per bin, and each clone's fit."""

    copies: list[np.ndarray]
    shifts: np.ndarray
    purity: np.ndarray
    alpha: float
    tau: float
    temperatures: list[float]


def _weighted(
    log_mu: np.ndarray,
    p: np.ndarray,
    shift: float,
    bulk: Pseudobulk,
    gamma: np.ndarray,
    states: np.ndarray,
) -> float:
    """`sum_g sum_k gamma_kg log f(x_g | k)` over the states carrying weight."""
    bins = np.arange(gamma.shape[1])
    return float(
        sum(
            np.sum(gamma[k] * _emission(log_mu[k] - shift, p[k], bulk, bins))
            for k in states
        )
    )


def _negative_marginal(
    purity: float,
    lattice: np.ndarray,
    shift: float,
    bulk: Pseudobulk,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
    best: bool = False,
) -> float:
    """Minus the clone's forward log-likelihood, every path summed, at `purity`.

    With `best`, the best path's alone (Viterbi's), for the hard scheme.

    The tumour fraction is shared by every bin, so it is fitted to the
    marginal rather than to responsibilities computed at the old fraction,
    which pin it: a bin placed at `(1, 5)` under a pure fit only loses
    likelihood as the fraction falls, though `(0, 1)` would then fit it.
    """
    log_mu, p = _parameters(lattice, purity)
    bins = np.arange(bulk.counts_nb.size)
    emission = np.stack(
        [_emission(log_mu[k] - shift, p[k], bulk, bins) for k in range(len(lattice))]
    )
    emission = np.where(np.isfinite(emission), emission, -1e10)
    total = 0.0
    begin = 0

    for length in np.asarray(lengths, dtype=np.int64):
        stop = begin + int(length)
        forward = log_startprob + emission[:, begin]

        for t in range(begin + 1, stop):
            step = forward[:, None] + log_transmat
            reduced = step.max(axis=0) if best else logsumexp(step, axis=0)
            forward = reduced + emission[:, t]

        total += float(forward.max() if best else logsumexp(forward))
        begin = stop

    return -total


def _negative_shift_weighted(
    shift: float,
    log_mu: np.ndarray,
    p: np.ndarray,
    bulk: Pseudobulk,
    gamma: np.ndarray,
    states: np.ndarray,
) -> float:
    return -_weighted(log_mu, p, shift, bulk, gamma, states)


def _negative_dispersion_weighted(
    log_value: float,
    which: str,
    other: float,
    bulks: list[Pseudobulk],
    lattice: np.ndarray,
    purity: np.ndarray,
    shifts: np.ndarray,
    gammas: list[np.ndarray],
    floor: float,
) -> float:
    value = float(np.exp(log_value))
    alpha, tau = (value, other) if which == "alpha" else (other, value)
    total = 0.0

    for i, bulk in enumerate(bulks):
        log_mu, p = _parameters(lattice, float(purity[i]))
        states = np.flatnonzero(gammas[i].max(axis=1) > floor)
        total += _weighted(
            log_mu, p, float(shifts[i]), _with(bulk, alpha, tau), gammas[i], states
        )

    return -total


def tempered_em(
    bulks: list[Pseudobulk],
    shifts: np.ndarray,
    *,
    normal_clone: int,
    lengths: np.ndarray,
    max_total_copy: int,
    stay: float,
    temperatures: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05),
    fit_purity: bool = True,
    zero_normal: bool = True,
    floor: float = 1e-6,
) -> TemperedFit:
    """EM with one state per `(A, B)`, responsibilities tempered to one state (#362).

    States are every pair with `A + B <= max_total_copy`, each at its own
    `mu` and `p` (:func:`_parameters`) under each clone's tumour fraction.
    Transitions: `stay` on the diagonal, the rest spread evenly. At each
    temperature `T` the E-step is forward-backward with log emissions and
    transitions scaled by `1 / T`; the M-step then fits each clone's shift
    (the normal clone's held at 0 under `zero_normal`) and, with
    `fit_purity`, its tumour fraction (the normal clone's 1) to the
    marginal likelihood (:func:`_negative_marginal`), and then the
    shared `alpha` and `tau`, all weighted by the responsibilities. The
    last temperature's responsibilities are hardened to each bin's argmax.
    A temperature of 0 is the Viterbi scheme: each bin's responsibility is
    one-hot on its state in the best path, and the tumour fraction is fitted
    to that path's likelihood.
    """
    from scipy.optimize import minimize_scalar

    lattice = candidates(max_total_copy)
    n = len(lattice)
    log_transmat = np.log(
        np.full((n, n), (1.0 - stay) / (n - 1))
        + np.eye(n) * (stay - (1.0 - stay) / (n - 1))
    )
    log_startprob = np.full(n, -np.log(n))
    shifts = np.asarray(shifts, dtype=np.float64).copy()
    purity = np.ones(len(bulks))
    alpha, tau = bulks[0].alpha, bulks[0].tau
    gammas: list[np.ndarray] = []

    for temperature in temperatures:
        gammas = []

        for i, bulk in enumerate(bulks):
            fitted = _with(bulk, alpha, tau)
            log_mu, p = _parameters(lattice, float(purity[i]))
            bins = np.arange(bulk.counts_nb.size)
            emission = np.stack(
                [_emission(log_mu[k] - shifts[i], p[k], fitted, bins) for k in range(n)]
            )
            emission = np.where(np.isfinite(emission), emission, -1e10)
            if temperature == 0.0:
                path, _ = _viterbi(emission, log_transmat, log_startprob, lengths)
                hard = np.zeros_like(emission)
                hard[path, np.arange(path.size)] = 1.0
                gammas.append(hard)
            else:
                gammas.append(
                    _forward_backward(
                        emission, log_transmat, log_startprob, lengths, temperature
                    )
                )

        for i, bulk in enumerate(bulks):
            fitted = _with(bulk, alpha, tau)
            states = np.flatnonzero(gammas[i].max(axis=1) > floor)

            if fit_purity and i != normal_clone:
                purity[i] = float(
                    minimize_scalar(
                        _negative_marginal,
                        bounds=(0.05, 1.0),
                        args=(
                            lattice,
                            float(shifts[i]),
                            fitted,
                            log_transmat,
                            log_startprob,
                            lengths,
                            temperature == 0.0,
                        ),
                        method="bounded",
                    ).x
                )

            if zero_normal and i == normal_clone:
                shifts[i] = 0.0
                continue

            log_mu, p = _parameters(lattice, float(purity[i]))
            shifts[i] = float(
                minimize_scalar(
                    _negative_shift_weighted,
                    bounds=(shifts[i] - 3.0, shifts[i] + 3.0),
                    args=(log_mu, p, fitted, gammas[i], states),
                    method="bounded",
                ).x
            )

        common = (bulks, lattice, purity, shifts, gammas, floor)
        alpha = float(
            np.exp(
                minimize_scalar(
                    _negative_dispersion_weighted,
                    bounds=(np.log(alpha) - 5.0, np.log(alpha) + 5.0),
                    args=("alpha", tau, *common),
                    method="bounded",
                ).x
            )
        )
        tau = float(
            np.exp(
                minimize_scalar(
                    _negative_dispersion_weighted,
                    bounds=(np.log(tau) - 5.0, np.log(tau) + 5.0),
                    args=("tau", alpha, *common),
                    method="bounded",
                ).x
            )
        )

    copies = [lattice[np.argmax(gamma, axis=0)] for gamma in gammas]
    return TemperedFit(copies, shifts, purity, alpha, tau, list(temperatures))
