"""`plot_clones_genomic`'s `colour_by` against the planted oversampled states.

Five fitted states over three integer pairs; unset is pinned against upstream.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from port.patch import plot_genomic
from port.patch.plot_genomic import bin_colours
from port.scripts.run_cnaster import main

from tests.fixtures import genomic_plot_instance

PAIRS = [(1, 1), (1, 1), (2, 1), (2, 1), (3, 0)]
"""State `k`'s decoded `(A, B)`: states 0/1 and 2/3 oversample one pair each."""


def _instance() -> dict[str, Any]:
    n_obs, n_spots = 40, 4
    path = np.repeat(np.arange(len(PAIRS)), n_obs // len(PAIRS))
    frame = {
        "CHR": np.ones(n_obs, dtype=int),
        "clone0 A": np.array([PAIRS[k][0] for k in path]),
        "clone0 B": np.array([PAIRS[k][1] for k in path]),
    }

    return {
        "arguments": genomic_plot_instance(5, n_obs=n_obs, n_spots=n_spots)[
            "arguments"
        ],
        "result": {
            "new_assignment": np.zeros(n_spots, dtype=np.int64),
            "pred_cnv": path[:, None],
            "new_log_mu": np.log([[1.0], [1.08], [1.45], [1.55], [1.5]]),
            "new_p_binom": np.array([[0.5], [0.52], [0.66], [0.69], [0.99]]),
        },
        "df_cnv": pd.DataFrame(frame),
        "path": path,
    }


def _colours(colour_by: str | None) -> tuple[np.ndarray, list[str]]:
    instance = _instance()
    colours, legend = bin_colours(
        df_cnv=instance["df_cnv"],
        res_combine=instance["result"],
        label="0",
        clone=0,
        n_obs=instance["path"].size,
        palette_name="chisel",
        phased_integer_copies=True,
        colour_by=colour_by,
    )
    return np.asarray(colours), [text for _, text in legend]


@pytest.mark.analytic
def test_integer_colours_deduplicate_the_oversampled_states() -> None:
    colours, names = _colours("integer")
    path = _instance()["path"]

    assert len(np.unique(colours, axis=0)) == 3
    np.testing.assert_array_equal(colours[path == 0], colours[path == 1])
    np.testing.assert_array_equal(colours[path == 2], colours[path == 3])
    assert [name.split("% ")[1] for name in names] == ["(1, 1)", "(2, 1)", "(3, 0)"]


@pytest.mark.analytic
def test_state_colours_keep_each_fitted_state_with_its_continuous_copies() -> None:
    colours, names = _colours("states")
    path = _instance()["path"]

    assert len(np.unique(colours, axis=0)) == len(PAIRS)
    assert not np.array_equal(colours[path == 0][0], colours[path == 1][0])
    assert [name.split("% ")[1] for name in names] == [
        "2mu=2.00 p=0.50",
        "2mu=2.16 p=0.52",
        "2mu=2.90 p=0.66",
        "2mu=3.10 p=0.69",
        "2mu=3.00 p=0.99",
    ]


@pytest.mark.patch
def test_unset_is_upstreams_choice() -> None:
    """With `df_cnv`, unset colours as `"integer"`; the default is unchanged."""
    unset, _ = _colours(None)
    integer, _ = _colours("integer")

    np.testing.assert_array_equal(unset, integer)


@pytest.mark.warning
def test_a_mode_without_its_input_is_refused() -> None:
    instance = _instance()
    common: dict[str, Any] = {
        "label": "0",
        "clone": 0,
        "n_obs": 40,
        "palette_name": "chisel",
        "phased_integer_copies": True,
    }

    with pytest.raises(ValueError, match="needs df_cnv"):
        bin_colours(
            df_cnv=None, res_combine=instance["result"], colour_by="integer", **common
        )
    with pytest.raises(ValueError, match="one of"):
        bin_colours(
            df_cnv=instance["df_cnv"],
            res_combine=instance["result"],
            colour_by="continuous",
            **common,
        )


@pytest.mark.smoke
def test_the_preference_reaches_a_figure_and_falls_back_where_it_cannot() -> None:
    """Preferring "states" recolours the figure; "integer" without `df_cnv` falls back to states."""

    mpl.use("Agg")

    instance = _instance()
    lengths, X, base, total = instance["arguments"]

    def drawn(**keywords: Any) -> np.ndarray:
        figure = plot_genomic.plot_clones_genomic(
            lengths,
            X,
            base,
            total,
            res_combine=instance["result"],
            phased_integer_copies=True,
            **keywords,
        )

        (points,) = (
            c for c in figure.axes[0].collections if isinstance(c, PathCollection)
        )
        colours = np.asarray(points.get_facecolor())
        plt.close(figure)
        return colours

    recoloured = drawn(df_cnv=instance["df_cnv"], preferred_colour_by="states")
    fallback = drawn(preferred_colour_by="integer")

    np.testing.assert_array_equal(recoloured, drawn(df_cnv=None))
    np.testing.assert_array_equal(fallback, drawn(df_cnv=None))
    assert len(np.unique(recoloured, axis=0)) == len(PAIRS)


@pytest.mark.smoke
def test_run_cnaster_port_refuses_the_colours_without_the_figure_swaps() -> None:
    with pytest.raises(SystemExit):
        main(["config.yaml", "--no-figure-swaps", "--genomic-colours", "states"])
