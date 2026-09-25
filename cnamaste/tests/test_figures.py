"""The figures (#392 stage 2): that what is drawn is the data given, and that a
written figure is a function of its content alone."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from sim.truth import Truth, planted

from cnamaste.plot_genomic import clone_path, fitted_levels
from cnamaste.plotting import draw_clones_spatial, pitch, spot_colours
from cnamaste.utils import write_fig

mpl.use("Agg")


@pytest.fixture(scope="module")
def truth() -> Truth:
    return planted(n_clones=3, n_states=4, lattice=(12, 10), n_obs=60, n_segments=3)


def _coords(truth: Truth, spacing: float = 1.0) -> np.ndarray:
    rows, columns = np.unravel_index(np.arange(truth.n_spots), truth.lattice)
    return spacing * np.stack([columns, rows], axis=1).astype(np.float64)


def _named(labels: np.ndarray) -> pd.Series:
    """Clone labels as the pipeline writes them, `"clone <k>"`."""
    return pd.Series([f"clone {label}" for label in labels])


def _res(truth: Truth, layout: str) -> dict[str, Any]:
    """The fit's result as the plots read it, `pred_cnv` in either layout."""
    pred = truth.states.T if layout == "columns" else truth.states.reshape(-1)
    return {
        "new_log_mu": truth.log_mu[:, None],
        "new_p_binom": truth.p_binom[:, None],
        "pred_cnv": pred,
        "new_assignment": truth.labels,
    }


@pytest.mark.analytic
def test_each_spot_takes_its_clones_colour_and_clones_are_distinguished(
    truth: Truth,
) -> None:
    """One colour per clone, the normal clone light grey, an unassigned spot transparent."""
    assignment = _named(truth.labels)
    assignment[:3] = None
    rgba, clones, colours = spot_colours(assignment)

    assert list(clones) == [f"clone {k}" for k in range(truth.n_clones)]
    assert len({to_rgba(c) for c in colours}) == truth.n_clones
    assert to_rgba(colours[0]) == to_rgba("lightgrey")
    np.testing.assert_array_equal(rgba[:3, 3], 0.0)

    for clone, colour in zip(clones, colours, strict=True):
        spots = (assignment == clone).to_numpy()
        np.testing.assert_allclose(
            rgba[spots], np.tile(to_rgba(colour), (spots.sum(), 1))
        )


@pytest.mark.analytic
def test_tumour_proportion_is_the_spots_opacity(truth: Truth) -> None:
    proportion = np.linspace(0.0, 1.0, truth.n_spots)
    proportion[::7] = np.nan
    rgba, _, _ = spot_colours(_named(truth.labels), single_tumor_prop=proportion)
    np.testing.assert_allclose(
        rgba[:, 3], np.where(np.isnan(proportion), 0.5, proportion)
    )


@pytest.mark.analytic
@pytest.mark.parametrize("spacing", [1.0, 2.5])
def test_the_pitch_is_the_lattice_spacing(truth: Truth, spacing: float) -> None:
    assert pitch(_coords(truth, spacing)) == pytest.approx(spacing, rel=1e-12)


@pytest.mark.analytic
def test_every_spot_is_drawn_once_at_its_place_in_its_colour(truth: Truth) -> None:
    """One tile per spot, centred on `(x, -y)`, filled with the spot's colour."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    coords = _coords(truth)
    figure, ax = plt.subplots()
    draw_clones_spatial(ax, coords, _named(truth.labels))
    (tiles,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    plt.close(figure)

    centres = np.array(
        [np.asarray(path.vertices)[:-1].mean(axis=0) for path in tiles.get_paths()]
    )
    np.testing.assert_allclose(centres, coords * [1, -1], atol=1e-12)

    rgba, _, _ = spot_colours(_named(truth.labels))
    np.testing.assert_allclose(
        np.asarray(tiles.get_facecolor(), dtype=np.float64), rgba
    )


@pytest.mark.analytic
def test_both_pred_layouts_give_each_clone_its_planted_path(truth: Truth) -> None:
    for clone in range(truth.n_clones):
        for layout in ("columns", "concatenated"):
            path = clone_path(_res(truth, layout), clone, truth.n_obs)
            np.testing.assert_array_equal(path, truth.states[clone], err_msg=layout)


@pytest.mark.analytic
def test_levels_are_the_maximal_runs_of_the_path_at_the_fitted_parameters(
    truth: Truth,
) -> None:
    """Runs `[start, end)` by brute-force scan, at `exp(log_mu)` and `p` of the run's state."""
    res = _res(truth, "columns")

    for clone in range(truth.n_clones):
        path = truth.states[clone]
        levels = fitted_levels(
            res, clone, truth.n_obs, truth.base_nb_mean, shifted=False
        )

        starts = np.flatnonzero(np.diff(path, prepend=-1) != 0)
        np.testing.assert_array_equal(levels.starts, starts)
        np.testing.assert_allclose(
            levels.rdr, np.exp(truth.log_mu[path[starts]]), rtol=1e-15
        )
        np.testing.assert_array_equal(levels.baf, truth.p_binom[path[starts]])
        # NB half-open: a run ends where the next begins, the last at `n_obs`.
        np.testing.assert_array_equal(levels.ends, np.append(starts[1:], truth.n_obs))


@pytest.mark.analytic
def test_shifted_levels_normalize_the_clones_library(truth: Truth) -> None:
    """`sum_g lambda_g exp(log_mu_g - log Z_c) = 1`, with `lambda` the baseline's shares."""
    res = _res(truth, "columns")
    profile = truth.base_nb_mean.sum(axis=1)
    share = profile / profile.sum()

    for clone in range(truth.n_clones):
        path = truth.states[clone]
        levels = fitted_levels(
            res, clone, truth.n_obs, truth.base_nb_mean, shifted=True
        )
        starts = np.flatnonzero(np.diff(path, prepend=-1) != 0)
        run = np.searchsorted(starts, np.arange(truth.n_obs), side="right") - 1
        assert np.sum(share * levels.rdr[run]) == pytest.approx(1.0, rel=1e-12)


def _figure(seed: int) -> Any:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    figure, ax = plt.subplots()
    ax.scatter(*rng.normal(size=(2, 500)), rasterized=True)
    ax.plot(rng.normal(size=20))
    return figure


@pytest.mark.analytic
@pytest.mark.parametrize("suffix", [".pdf", ".png"])
def test_a_written_figure_is_a_function_of_its_content(
    tmp_path: Path, suffix: str
) -> None:
    """The same figure twice writes the same bytes; a different one does not.

    `SOURCE_DATE_EPOCH` fixes the creation date a PDF otherwise stamps.
    """
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        paths = [tmp_path / f"{name}{suffix}" for name in ("a", "b", "c")]
        for path, seed in zip(paths, (0, 0, 1), strict=True):
            write_fig(str(path), _figure(seed))
    finally:
        if previous is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous

    first, second, other = (path.read_bytes() for path in paths)
    assert first == second
    assert first != other
