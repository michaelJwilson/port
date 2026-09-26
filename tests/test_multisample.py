"""Several samples of one genome with shared clones, through `run_cnaster_port` (#328).

`tests.multisample` concatenates realizations of one planted genome, so the
clones are shared by construction and the referee is the planted labelling:

- the entry point runs as is on three samples, and recovers the planted
  clones in every sample and pooled (`end2end`);
- the fixture: one genome, redrawn counts, each spot's `sample_label`, and
  the samples side by side on the grid with a gap (`analytic`);
- the cross-sample adjacency placeholder has no edge inside a sample, and
  `sample_layout` draws one panel per sample in the run's colours (`infra`).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

N_SAMPLES = 3


INSTANCE: dict[str, Any] = {
    "n_clones": 2,
    "n_states": 3,
    "lattice": (25, 40),
    "n_obs": 40,
    "n_segments": 3,
    "seed": 11,
}
"""`tests/test_run_cnaster_port_end_to_end.py`'s instance, which one sample
recovers at ARI 1.000, so what three change is what is measured."""


def _base() -> Any:
    from tests.fixtures import core_inference_truth

    return core_inference_truth(**INSTANCE)


def _multi() -> Any:
    from tests.multisample import multi_sample_truth

    return multi_sample_truth(_base(), N_SAMPLES)


@pytest.mark.end2end
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_the_entry_point_recovers_the_shared_clones_in_every_sample(
    tmp_path: Path,
) -> None:
    """One clone labelling across three samples: every clone in every sample,
    at most 2 spots per sample misplaced."""
    import matplotlib as mpl
    from port.scripts.run_cnaster import main

    from tests.run_config import isolated_run, write_run_cnaster_config
    from tests.test_core_inference_end_to_end import _adjusted_rand_index
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    mpl.use("Agg")
    multi = _multi()
    truth = multi.truth
    written = write_tmp_inputs(
        truth,
        unsegment(truth, flip_every=0, unassigned_genes=0),
        tmp_path,
        sample_label=multi.sample_label,
        positions=multi.positions,
    )
    config = write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3)

    with isolated_run(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([str(config), "--sample-layout", "3,1"]) == 0

    output = written.root / "output"
    labels = pd.read_csv(next(output.rglob("clone_labels.tsv")), sep="\t", comment="#")
    spots = labels["barcode"].str.slice(2, 7).astype(int).to_numpy()
    fitted = np.full(truth.n_spots, -1, dtype=np.int64)
    fitted[spots] = labels["clone_label"].to_numpy()

    assert (fitted >= 0).all(), "every spot of every sample is labelled"

    # NB each fitted clone read as the planted clone most of its spots carry,
    #    pooled, so a clone is shared only if it maps to one planted clone in
    #    every sample. Measured here: 0 of 3,000 misplaced, ARI 1.000 pooled
    #    and per sample. Two per sample is the stated tolerance, as in
    #    `tests/test_run_cnaster_port_end_to_end.py`: the ICM's ties fall on
    #    platform floating-point order.
    majority = {
        clone: np.bincount(truth.labels[fitted == clone]).argmax()
        for clone in np.unique(fitted)
    }
    placed = np.vectorize(majority.get)(fitted)

    assert sorted(majority.values()) == list(range(truth.n_clones))
    assert _adjusted_rand_index(truth.labels, fitted) >= 0.99

    for sample in range(N_SAMPLES):
        here = multi.spots(sample)
        wrong = int(np.sum(placed[here] != truth.labels[here]))

        assert wrong <= 2, f"sample {sample}: {wrong} spots in the wrong clone"
        assert set(fitted[here]) == set(fitted), "each clone is in every sample"


@pytest.mark.analytic
def test_the_samples_are_one_genome_redrawn_side_by_side() -> None:
    """Shared states and labels; distinct counts; sample `k` offset by `k (40 + 1)`."""
    base = _base()
    multi = _multi()
    n = base.n_spots
    rows, columns = base.lattice

    np.testing.assert_array_equal(multi.truth.states, base.states)
    np.testing.assert_array_equal(multi.truth.labels, np.tile(base.labels, N_SAMPLES))
    np.testing.assert_array_equal(multi.truth.counts_nb[:, :n], base.counts_nb)
    assert not np.array_equal(multi.truth.counts_nb[:, n : 2 * n], base.counts_nb)
    np.testing.assert_array_equal(
        multi.sample_label, np.repeat(np.arange(N_SAMPLES), n)
    )

    for sample in range(N_SAMPLES):
        spots = multi.spots(sample)
        assert multi.positions[spots, 1].min() == sample * (columns + multi.gap)
        assert multi.positions[spots, 0].max() == rows - 1


@pytest.mark.analytic
def test_the_grid_labels_each_sample_and_leaves_the_gaps_empty() -> None:
    """`grid()` is `sample_label` on the grid: a 25 x 122 slice, three blocks, two gap columns."""
    multi = _multi()
    grid = multi.grid()
    rows, columns = INSTANCE["lattice"]

    assert grid.shape == (rows, N_SAMPLES * columns + (N_SAMPLES - 1) * multi.gap)

    for sample in range(N_SAMPLES):
        start = sample * (columns + multi.gap)
        assert (grid[:, start : start + columns] == sample).all()

    assert (grid[:, [columns, 2 * columns + 1]] == -1).all()
    assert (grid >= 0).sum() == multi.truth.n_spots


@pytest.mark.infra
def test_the_cross_sample_placeholder_has_no_edge_inside_a_sample() -> None:
    """Empty today; the invariant an implementation keeps is no within-sample entry."""
    from port.extensions.multisample import cross_sample_adjacency

    multi = _multi()
    adjacency = cross_sample_adjacency(multi.sample_label)
    same = multi.sample_label[:, None] == multi.sample_label[None, :]

    assert adjacency.shape == (multi.truth.n_spots,) * 2
    assert (abs(adjacency - adjacency.T)).nnz == 0
    assert adjacency.toarray()[same].sum() == 0.0


@pytest.mark.infra
def test_a_layout_places_the_samples_row_by_row_and_refuses_too_few_panels() -> None:
    from port.extensions.multisample import sample_panels

    labels = np.repeat([0, 1, 2], 4)
    panels = sample_panels(labels, (3, 1))

    assert [(row, column) for row, column, _ in panels] == [(0, 0), (1, 0), (2, 0)]
    np.testing.assert_array_equal(panels[1][2], np.arange(4, 8))

    with pytest.raises(ValueError, match="holds 2 of 3"):
        sample_panels(labels, (1, 2))


@pytest.mark.infra
def test_sample_layout_draws_a_panel_per_sample_in_the_runs_colours() -> None:
    """Three panels, each in its own coordinates; a clone absent from one sample
    keeps its colour in the others; unset, one axis as upstream."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.patch.plotting.spatial import plot_clones_spatial

    rng = np.random.default_rng(0)
    coords = np.array([(r, c) for r in range(6) for c in range(6)] * 3, dtype=float)
    sample_ids = np.repeat(np.arange(3), 36)
    clones = np.asarray(rng.integers(0, 3, coords.shape[0]), dtype=np.int64)
    # NB sample 1 carries no clone 2, which must keep its colour elsewhere.
    clones[(sample_ids == 1) & (clones == 2)] = 1
    assignment = pd.Series([f"clone {c}" for c in clones])

    figure = plot_clones_spatial(
        coords,
        assignment,
        sample_ids=sample_ids,
        sample_list=["S1", "S2", "S3"],
        sample_layout=(3, 1),
    )
    faces = [np.asarray(ax.collections[0].get_facecolor()) for ax in figure.axes]

    assert len(figure.axes) == 3
    assert all(ax.get_xlim()[1] < 6 for ax in figure.axes), "own coordinates"
    for sample, colours in enumerate(faces):
        here = clones[sample_ids == sample]
        for clone in np.unique(here):
            np.testing.assert_array_equal(
                np.unique(colours[here == clone], axis=0),
                np.unique(faces[0][clones[sample_ids == 0] == clone], axis=0),
            )
    plt.close(figure)

    single = plot_clones_spatial(
        coords, assignment, sample_ids=sample_ids, sample_list=["S1", "S2", "S3"]
    )
    assert len(single.axes) == 1
    plt.close(single)
