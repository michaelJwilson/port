"""The LOH density figure, against the one `cnaster` draws (#281).

`tests/test_clone_paths.py` refereed the **model** half -- the nine lines
that decode a clone's path and read its BAF. This referees the rest: the
empirical channel, the alpha ramp, the axis remapping and the two-panel
layout, which together are 41 of the module's 67 statements and had no
referee at all.

**Drawn artists, not pixels.** The figure is a 3-D scatter, so what it
asserts is the point cloud: the offsets each `Path3DCollection` holds and
the RGBA it holds them with. A pixel comparison would fail on a font or a
backend version and pass a rewiring that moved a clone.
"""

from typing import Any

import numpy as np
import pytest


def _instance(
    n_bins: int = 12, n_spots: int = 9, n_clones: int = 3, n_states: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """A three-clone instance whose clones hold different spots.

    Three rather than one, for the reason `test_the_loh_model_matches_
    upstreams_loop` gives: a single clone cannot tell the concatenated decode
    apart from a per-column one, which is the guard `clone_paths` removed.
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
    """Every point a panel put in space, and the colour it put there.

    Both halves matter and for different reasons. The offsets say which
    `(bin, spot)` survived the visibility cut; the RGBA says what the LOH
    signal was, which is the quantity the figure exists to show. Checking
    one alone would pass a patch that drew the right points in the wrong
    colour.
    """
    drawn = []

    for axis in figure.axes:
        for collection in axis.collections:
            offsets = np.asarray(collection._offsets3d)
            drawn.append((offsets, np.asarray(collection.get_facecolors())))

    return drawn


@pytest.mark.cnaster
@pytest.mark.patch
def test_the_replacement_draws_the_density_upstream_draws() -> None:
    """Both figures, one instance, every point and colour compared bitwise.

    Nothing is reassociated between the two, so the same arithmetic on the
    same inputs gives the same doubles and a tolerance would only hide a
    rewiring. The claim is that replacing the clone loop with `clone_paths`
    changed the loop and not the figure.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_loh_density import plot_loh_density as upstream
    from port.sandbox.patch.loh_density import plot_loh_density as replacement

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
    """One channel at a time, because `both` would hide a swap between them.

    The empirical channel comes from the smoothed BAF and the model channel
    from the decoded path, and the two-panel figure draws them in order. A
    patch that exchanged them would reproduce `both` as a set and fail here.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_loh_density import plot_loh_density as upstream
    from port.sandbox.patch.loh_density import plot_loh_density as replacement

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
    """Upstream skips an empty clone rather than writing from an empty mask.

    The replacement carries that, and it is worth its own test because the
    two ways of being wrong -- writing zeros, or raising on the empty
    reduction -- both look like a working figure until a clone empties.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_loh_density import plot_loh_density as upstream
    from port.sandbox.patch.loh_density import plot_loh_density as replacement

    coords, single_X, total_bb_RD, result = _instance()

    # NB clone 2 keeps its states in `pred_cnv` and loses every spot, which
    #    is the case `new_assignment` and `pred_cnv` disagreeing produces.
    result["new_assignment"] = np.where(
        result["new_assignment"] == 2, 0, result["new_assignment"]
    )

    theirs = _cloud(upstream(coords, single_X, total_bb_RD, res_combine=result))
    ours = _cloud(replacement(coords, single_X, total_bb_RD, res_combine=result))

    for (mine, my_colours), (drawn, colours) in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(mine, drawn)
        np.testing.assert_array_equal(my_colours, colours)
