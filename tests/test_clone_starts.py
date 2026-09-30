"""#541: a clone-assignment problem built from a labelling, and the starts that read it, on spots drawn here.

The study's numbers are in `docs/nb/clone_label_study.ipynb`; these pin what
its problem builder must do for them to mean anything: the pseudobulk is the
spots' sum, the profile is the most likely path, and the field a labelling
gives separates the clones that drew the counts.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.sandbox.clone_starts import problem as pr
from port.sandbox.clone_starts import starts as st

STATES = ((0.0, 0.5), (np.log(0.5), 0.02), (np.log(1.5), 1.0 / 3.0))
"""`(log mu, p)`: neutral, a one-copy loss, a gain."""


def _capture(n_side: int = 20, n_bins: int = 120, seed: int = 0) -> pr.Capture:
    """Two clones side by side on an `n_side` square lattice; clone 1 carries a loss then a gain over the second half of the bins."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n = n_side * n_side
    x, y = np.divmod(np.arange(n), n_side)
    planted = (x >= n_side // 2).astype(np.int64)
    state = np.zeros((n_bins, 2), dtype=np.int64)
    state[n_bins // 2 : 3 * n_bins // 4, 1] = 1
    state[3 * n_bins // 4 :, 1] = 2
    log_mu = np.array([s[0] for s in STATES])[state[:, planted]]
    p = np.array([s[1] for s in STATES])[state[:, planted]]
    base = np.tile(rng.uniform(2.0, 6.0, (n_bins, 1)), (1, n))
    trials = rng.integers(2, 12, (n_bins, n)).astype(np.float64)
    counts = rng.poisson(base * np.exp(log_mu)).astype(np.float64)
    b = rng.binomial(trials.astype(int), p).astype(np.float64)
    right = sp.coo_matrix(
        (np.ones(n - n_side), (np.arange(n - n_side), np.arange(n_side, n))),
        shape=(n, n),
    )
    down_rows = np.flatnonzero(y < n_side - 1)
    down = sp.coo_matrix(
        (np.ones(down_rows.size), (down_rows, down_rows + 1)), shape=(n, n)
    )
    adjacency = (right + down + right.T + down.T).tocsr()
    return pr.Capture(
        counts=counts,
        b=b,
        base=base,
        trials=trials,
        lengths=np.array([n_bins // 2, n_bins - n_bins // 2]),
        indptr=adjacency.indptr,
        indices=adjacency.indices,
        weights=adjacency.data.astype(np.float64),
        spatial_weight=1.0,
        t=1 - 1e-4,
        n_states=3,
        planted=planted,
        normal_candidates=planted == 0,
        coords=np.column_stack([x, y]).astype(np.float64),
        sample_ids=np.zeros(n, dtype=np.int64),
        floor=10,
    )


@pytest.mark.oracle
def test_the_pseudobulk_is_each_clones_sum_over_its_spots() -> None:
    """Against a loop over clones summing the spots' columns."""
    capture = _capture()
    labels = np.random.default_rng(1).integers(0, 3, capture.n_spots)
    counts, b, base, trials = pr.pseudobulk(capture, labels)

    for clone in range(3):
        spots = labels == clone
        np.testing.assert_allclose(
            counts[:, clone], capture.counts[:, spots].sum(axis=1)
        )
        np.testing.assert_allclose(b[:, clone], capture.b[:, spots].sum(axis=1))
        np.testing.assert_allclose(base[:, clone], capture.base[:, spots].sum(axis=1))
        np.testing.assert_allclose(
            trials[:, clone], capture.trials[:, spots].sum(axis=1)
        )


@pytest.mark.oracle
def test_the_profile_path_is_the_most_likely_one_and_restarts_at_each_contig() -> None:
    """`_viterbi` on 12 bins, 3 states, two contigs, against every path scored by brute force."""
    import itertools

    rng = np.random.default_rng(2)
    emission = rng.normal(size=(6, 3)) * 3.0
    lengths = np.array([3, 3])
    t = 0.9
    path = pr._viterbi(emission, lengths, t)

    for first in (0, 3):
        best, chosen = -np.inf, None
        for candidate in itertools.product(range(3), repeat=3):
            score = sum(emission[first + i, s] for i, s in enumerate(candidate))
            score += sum(
                np.log(t) if a == c else np.log((1 - t) / 2)
                for a, c in itertools.pairwise(candidate)
            )
            if score > best:
                best, chosen = score, candidate
        assert tuple(path[first : first + 3]) == chosen


@pytest.mark.end2end
def test_the_field_the_planted_labels_give_places_each_spot_in_the_clone_that_drew_it() -> (
    None
):
    """From the drawing states, polished: the field's argmax agrees with the planted clone on 95% of spots."""
    capture = _capture()
    states = (
        np.array([s[0] for s in STATES]),
        np.array([s[1] for s in STATES]),
    )
    built = pr.build(
        capture, capture.planted, np.random.default_rng(0), states=states, seconds=10.0
    )

    assert (np.argmax(built.field, axis=1) == capture.planted).mean() >= 0.95


@pytest.mark.analytic
def test_mean_field_corrects_isolated_spots_the_argmax_misplaces() -> None:
    """A field favouring the planted halves, with 10% of spots flipped alone: mean field at coupling 1 restores every flipped spot."""
    capture = _capture()
    rng = np.random.default_rng(3)
    field = np.zeros((capture.n_spots, 2))
    field[np.arange(capture.n_spots), capture.planted] = 1.0
    flipped = rng.random(capture.n_spots) < 0.1
    field[flipped] = field[flipped][:, ::-1]

    labels = st.mean_field(capture, rng, field=field, k=2)

    assert (st.field_argmax(capture, rng, field=field) != capture.planted).sum() > 0
    assert np.array_equal(labels, capture.planted)
