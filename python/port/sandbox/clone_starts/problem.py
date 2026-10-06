"""Set aside (#541): a clone-assignment problem built from a labelling, as `--sal` builds it.

Ticket: #541 -- which clone-label start and which Potts solver place the
  spots in their clones, with copy states from #540's starts.
Measurement: `docs/nb/clone_label_study.ipynb`, on dev_tree 60 x 50 r0 (`3381575a`).
Exit: a start or solver graduates to `extensions/` or `patch/` if it is at
  least as accurate as `--sal`'s on dev_tree, easy, hard and
  `dev_shared_unique` end to end and no slower; else this stays the study's.

A labelling fixes everything else, as one round of `pipeline_clone_assignment`
does. From the spots' counts (`Capture`) and a label per spot:

- **pseudobulk**: each clone's summed counts, exposure and trials per bin;
- **copy states**: a start of `port.extensions.copy_starts` on the clones'
  stacked pseudobulk (`kmeans++x5+em`, `--sal`'s), polished by `sal`'s EM;
- **profiles**: each clone's Viterbi path over those states, bin by bin,
  restarting at each contig, with the NB size and BB concentration fitted
  along it (`profiles`);
- **field**: each spot's log-likelihood under each clone's profile, by the
  kernel the pipeline uses (`port.patch.hmrf.tabulated_field`);
- **graph**: the run's adjacency at its coupling (`potts_graph_from`).

No arm reads the planted labels; `Capture.planted` is the score's alone.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = ["Capture", "Problem", "build", "load", "pseudobulk"]


class Capture(NamedTuple):
    """One run's spot-level inputs to its first BAF + RDR inference (`port.studies.clone_labels capture`)."""

    counts: np.ndarray
    """`(bins, spots)` total UMIs."""
    b: np.ndarray
    """`(bins, spots)` B-allele reads."""
    base: np.ndarray
    """`(bins, spots)` exposure, `base_nb_mean`."""
    trials: np.ndarray
    """`(bins, spots)` SNP-covering reads, `total_bb_RD`."""
    lengths: np.ndarray
    """Bins per contig."""
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    spatial_weight: float
    t: float
    n_states: int
    planted: np.ndarray
    normal_candidates: np.ndarray
    coords: np.ndarray
    sample_ids: np.ndarray
    floor: int

    @property
    def n_spots(self) -> int:
        return int(self.counts.shape[1])

    @property
    def umis(self) -> np.ndarray:
        """Each spot's total UMIs."""
        umis: np.ndarray = self.counts.sum(axis=0)
        return umis


def load(path: Path) -> Capture:
    """A capture written by `port.studies.clone_labels capture`."""
    held = np.load(path)
    single_X = held["single_X"]
    return Capture(
        counts=np.asarray(single_X[:, 0, :], dtype=np.float64),
        b=np.asarray(single_X[:, 1, :], dtype=np.float64),
        base=held["base"],
        trials=held["trials"],
        lengths=held["lengths"],
        indptr=held["indptr"],
        indices=held["indices"],
        weights=held["weights"],
        spatial_weight=float(held["spatial_weight"]),
        t=float(held["t"]),
        n_states=int(held["n_states"]),
        planted=held["planted"],
        normal_candidates=held["normal_candidates"],
        coords=held["coords"],
        sample_ids=held["sample_ids"],
        floor=int(held["min_spots_per_clone"]),
    )


class Problem(NamedTuple):
    """The field a labelling gives, and what it cost."""

    labels: np.ndarray
    """`(spots,)`, relabelled `0 .. q - 1` over the clones it uses."""
    field: np.ndarray
    """`(spots, q)` log-likelihood of each spot under each clone's profile."""
    log_mu: np.ndarray
    p_binom: np.ndarray
    pred: np.ndarray
    """`(bins, q)` each clone's state per bin."""
    seconds: float
    """The copy states, profiles and field, from the labels."""
    states_by: str
    """What gave the states: the start asked for, or the fallback `sal`'s refusal took."""


def _compact(labels: np.ndarray) -> np.ndarray:
    compact: np.ndarray = np.unique(np.asarray(labels), return_inverse=True)[1]
    return compact.astype(np.int64)


def pseudobulk(
    capture: Capture, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each clone's `(counts, b, base, trials)`, `(bins, clones)`: sums over its spots."""
    import scipy.sparse as sp

    labels = _compact(labels)
    member = sp.csr_matrix(
        (np.ones(labels.size), (np.arange(labels.size), labels)),
        shape=(labels.size, int(labels.max()) + 1),
    )
    return (
        np.asarray(member.T.dot(capture.counts.T).T),
        np.asarray(member.T.dot(capture.b.T).T),
        np.asarray(member.T.dot(capture.base.T).T),
        np.asarray(member.T.dot(capture.trials.T).T),
    )


def _call(capture: Capture, labels: np.ndarray) -> Any:
    """The clones' pseudobulk, stacked clone after clone, as #540's `CopyCall`."""
    from port.extensions.copy_starts import CopyCall

    counts, b, base, trials = pseudobulk(capture, labels)
    n_bins, n_clones = counts.shape
    unplaced = np.zeros(n_bins * n_clones)
    return CopyCall(
        stage="rdrbaf",
        n_states=capture.n_states,
        total=counts.T.ravel(),
        b=b.T.ravel(),
        exposure=base.T.ravel(),
        trials=trials.T.ravel(),
        clone=np.repeat(np.arange(n_clones), n_bins),
        contig=unplaced.astype(str),
        start=unplaced,
        length=unplaced,
        planted=np.full((n_bins * n_clones, 2), -1),
        raw={},
    )


def _viterbi(emission: np.ndarray, lengths: np.ndarray, t: float) -> np.ndarray:
    """The most likely state path, `emission` `(bins, states)`, restarting at each contig; stay `t`, else uniform."""
    n_bins, k = emission.shape
    stay = np.log(t)
    move = np.log((1.0 - t) / max(k - 1, 1))
    transition = np.full((k, k), move)
    np.fill_diagonal(transition, stay)
    path = np.zeros(n_bins, dtype=np.int64)
    edges = np.concatenate([[0], np.cumsum(lengths)])

    for first, last in itertools.pairwise(edges):
        score = emission[first].copy()
        back = np.zeros((last - first, k), dtype=np.int64)
        for i in range(first + 1, last):
            candidate = score[:, None] + transition
            back[i - first] = np.argmax(candidate, axis=0)
            score = candidate[back[i - first], np.arange(k)] + emission[i]
        state = int(np.argmax(score))
        for i in range(last - 1, first - 1, -1):
            path[i] = state
            state = int(back[i - first, state])
    return path


def profiles(
    capture: Capture,
    labels: np.ndarray,
    log_mu: np.ndarray,
    p_binom: np.ndarray,
    rounds: int = 2,
) -> tuple[np.ndarray, float, float]:
    """Each clone's Viterbi path over the states, `(bins, clones)`, and the NB size and BB concentration fitted along it.

    The emission is `sal`'s count pair on each clone's pseudobulk
    (`copy_starts._channels`); the shapes start at the seam's (size 10,
    concentration 1,000) and are refitted along the paths `rounds` times.
    """
    from port.extensions.copy_starts import _channels, _fit_shapes
    from port.patch.hmm_initialize.sal_mixture import EXPOSURE_SCALE

    counts, b, base, trials = pseudobulk(capture, labels)
    n_bins, n_clones = counts.shape
    observations = np.column_stack([counts.T.ravel(), b.T.ravel()])
    covariate = np.column_stack([base.T.ravel() / EXPOSURE_SCALE, trials.T.ravel()])
    channels = _channels(observations, covariate)
    depth, allele = channels
    rate = np.exp(np.asarray(log_mu, dtype=np.float64)) * EXPOSURE_SCALE
    share = np.clip(np.asarray(p_binom, dtype=np.float64), 1e-4, 1 - 1e-4)
    size, concentration = 10.0, 1_000.0
    pred = np.zeros((n_bins, n_clones), dtype=np.int64)

    for _ in range(rounds):
        emission = depth(rate, size) + allele(share, concentration)
        for clone in range(n_clones):
            rows = slice(clone * n_bins, (clone + 1) * n_bins)
            pred[:, clone] = _viterbi(emission[rows], capture.lengths, capture.t)
        state = pred.T.ravel()
        onehot = np.zeros((state.size, share.size))
        onehot[np.arange(state.size), state] = 1.0
        size, concentration, _ = _fit_shapes(
            channels, rate, share, onehot, concentration, 0.0
        )
    return pred, size, concentration


def field(
    capture: Capture,
    log_mu: np.ndarray,
    p_binom: np.ndarray,
    size: float,
    concentration: float,
    pred: np.ndarray,
) -> np.ndarray:
    """`(spots, clones)`: each spot's log-likelihood under each clone's profile, by the pipeline's kernel."""
    import scipy.sparse as sp

    from port.patch.hmrf.clone_assignment import boundary
    from port.patch.hmrf.tabulated_field import spot_clone_field

    n_states = np.asarray(log_mu).size
    weight = boundary(capture.base, capture.trials, sp.identity(capture.n_spots)).weight
    out: np.ndarray = spot_clone_field(
        capture.counts,
        capture.base,
        capture.b,
        capture.trials,
        np.asarray(log_mu, dtype=np.float64),
        np.full(n_states, 1.0 / size),
        np.clip(np.asarray(p_binom, dtype=np.float64), 1e-4, 1 - 1e-4),
        np.full(n_states, concentration),
        np.ascontiguousarray(pred),
        weight,
        np.empty((capture.n_spots, pred.shape[1])),
    )
    return out


def build(
    capture: Capture,
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    start: str = "kmeans++x5+em",
    seconds: float = 60.0,
    states: tuple[np.ndarray, np.ndarray] | None = None,
) -> Problem:
    """The problem `labels` give: copy states by `start` (or `states`, as given), profiles, field."""
    from port.extensions.copy_starts import (
        CopyStart,
        lattice_start,
        polish_states,
        run_start,
    )

    opened = time.perf_counter()
    labels = _compact(labels)
    call = _call(capture, labels)

    # NB `sal`'s EM refuses some pseudobulks -- a beta-binomial fit that
    #    degenerates, an M step that does not settle -- which the pipeline
    #    would raise on. The study records the fallback instead: the lattice,
    #    polished, then its states unpolished.
    try:
        fitted = (
            run_start(start, call, rng, seconds=seconds)
            if states is None
            else polish_states(start, call, *states, seconds=seconds)
        )
        states_by = start if states is None else "warm"
    except ValueError:
        try:
            fitted = run_start("lattice", call, rng, seconds=seconds)
            states_by = "lattice (fallback)"
        except ValueError:
            log_mu, p = lattice_start(call)
            fitted = CopyStart("lattice", "rdrbaf", log_mu, p, float("nan"), 0.0, 0.0)
            states_by = "lattice unpolished (fallback)"

    pred, size, concentration = profiles(capture, labels, fitted.log_mu, fitted.p_binom)
    return Problem(
        labels=labels,
        field=field(capture, fitted.log_mu, fitted.p_binom, size, concentration, pred),
        log_mu=fitted.log_mu,
        p_binom=fitted.p_binom,
        pred=pred,
        seconds=time.perf_counter() - opened,
        states_by=states_by,
    )
