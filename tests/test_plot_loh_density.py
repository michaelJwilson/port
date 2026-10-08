"""The LOH density figure against the one `cnaster` draws (#281).

Compares drawn artists (each `Path3DCollection`'s offsets and RGBA), not pixels.
"""

from typing import Any

import numpy as np
import pytest


def _instance(
    n_bins: int = 12, n_spots: int = 9, n_clones: int = 3, n_states: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """A three-clone instance whose clones hold different spots, so a per-column decode
    would differ.
    """
    rng = np.random.default_rng(11)

    coords = rng.uniform(0.0, 10.0, size=(n_spots, 2))
    total_bb_RD = rng.integers(15, 60, size=(n_bins, n_spots)).astype(float)

    single_X = np.zeros((n_bins, 2, n_spots))
    single_X[:, 0, :] = rng.poisson(120, size=(n_bins, n_spots))
    single_X[:, 1, :] = rng.binomial(total_bb_RD.astype(int), 0.38)

    result = {
        "new_assignment": np.tile(np.arange(n_clones), n_spots // n_clones),
        "pred_cnv": rng.integers(0, n_states, size=n_bins * n_clones),
        "new_p_binom": rng.uniform(0.1, 0.9, size=(n_states, 1)),
    }

    return coords, single_X, total_bb_RD, result


def _cloud(figure: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every point a panel placed and its RGBA colour."""
    drawn = []

    for axis in figure.axes:
        for collection in axis.collections:
            offsets = np.asarray(collection._offsets3d)
            drawn.append((offsets, np.asarray(collection.get_facecolors())))

    return drawn


@pytest.mark.cnaster
@pytest.mark.patch
def test_the_replacement_draws_the_density_upstream_draws() -> None:
    """Both figures on one instance: every point and colour equal bitwise."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_loh_density import plot_loh_density as upstream
    from port.sandbox.patch.plotting.loh_density import plot_loh_density as replacement

    coords, single_X, total_bb_RD, result = _instance()

    theirs = _cloud(upstream(coords, single_X, total_bb_RD, res_combine=result))
    ours = _cloud(replacement(coords, single_X, total_bb_RD, res_combine=result))

    assert len(ours) == len(theirs), (
        f"drew {len(ours)} panels against upstream's {len(theirs)}"
    )

    for index, ((mine, my_colours), (drawn, colours)) in enumerate(
        zip(ours, theirs, strict=True)
    ):
        np.testing.assert_array_equal(mine, drawn, err_msg=f"panel {index} offsets")
        np.testing.assert_array_equal(
            my_colours, colours, err_msg=f"panel {index} colours"
        )


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.parametrize("plot_type", ["empirical", "model"])
def test_each_panel_alone_is_upstreams(plot_type: str) -> None:
    """Each channel alone equals upstream's, so a swap between them would fail."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_loh_density import plot_loh_density as upstream
    from port.sandbox.patch.plotting.loh_density import plot_loh_density as replacement

    coords, single_X, total_bb_RD, result = _instance()

    theirs = _cloud(
        upstream(coords, single_X, total_bb_RD, res_combine=result, plot_type=plot_type)
    )
    ours = _cloud(
        replacement(
            coords, single_X, total_bb_RD, res_combine=result, plot_type=plot_type
        )
    )

    assert len(ours) == 1, f"{plot_type} drew {len(ours)} panels"

    np.testing.assert_array_equal(ours[0][0], theirs[0][0])
    np.testing.assert_array_equal(ours[0][1], theirs[0][1])


@pytest.mark.cnaster
@pytest.mark.patch
def test_a_clone_holding_no_spots_leaves_its_column_alone() -> None:
    """An empty clone's column is skipped, as upstream skips it."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_loh_density import plot_loh_density as upstream
    from port.sandbox.patch.plotting.loh_density import plot_loh_density as replacement

    coords, single_X, total_bb_RD, result = _instance()

    # NB clone 2 keeps its states in `pred_cnv` but loses every spot.
    result["new_assignment"] = np.where(
        result["new_assignment"] == 2, 0, result["new_assignment"]
    )

    theirs = _cloud(upstream(coords, single_X, total_bb_RD, res_combine=result))
    ours = _cloud(replacement(coords, single_X, total_bb_RD, res_combine=result))

    for (mine, my_colours), (drawn, colours) in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(mine, drawn)
        np.testing.assert_array_equal(my_colours, colours)
