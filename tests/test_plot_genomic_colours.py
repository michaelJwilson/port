"""`plot_clones_genomic`'s `colour_by`: integer copies deduplicated, or states.

The instance oversamples: five fitted states over three integer pairs, two
of them at `(2, 1)` and two at `(1, 1)`. The referee is that construction
(`analytic`): `"integer"` draws one colour per distinct pair, three;
`"states"` one per fitted state, five, each labelled with its own `2 mu`
and `p`; and unset is upstream's choice, pinned against it (`patch`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

PAIRS = [(1, 1), (1, 1), (2, 1), (2, 1), (3, 0)]
"""State `k`'s decoded `(A, B)`: states 0/1 and 2/3 oversample one pair each."""


def _instance() -> dict[str, Any]:
    rng = np.random.default_rng(5)
    n_obs, n_spots = 40, 4
    path = np.repeat(np.arange(len(PAIRS)), n_obs // len(PAIRS))

    total = rng.integers(20, 80, size=(n_obs, n_spots)).astype(float)
    X = np.zeros((n_obs, 2, n_spots))
    X[:, 0, :] = rng.poisson(150, size=(n_obs, n_spots))
    X[:, 1, :] = rng.binomial(total.astype(int), 0.45)
    frame = {
        "CHR": np.ones(n_obs, dtype=int),
        "clone0 A": np.array([PAIRS[k][0] for k in path]),
        "clone0 B": np.array([PAIRS[k][1] for k in path]),
    }

    return {
        "arguments": (
            np.array([n_obs]),
            X,
            rng.uniform(100.0, 200.0, size=(n_obs, n_spots)),
            total,
        ),
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
    from port.patch.plot_genomic import bin_colours

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


@pytest.mark.infra
def test_a_mode_without_its_input_is_refused() -> None:
    from port.patch.plot_genomic import bin_colours

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


@pytest.mark.infra
def test_the_module_default_reaches_a_figure_and_falls_back_where_it_cannot() -> None:
    """`COLOUR_BY = "states"` recolours the df_cnv figure; "integer" leaves a
    figure without df_cnv coloured by state rather than refusing it."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.patch import plot_genomic

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
        from matplotlib.collections import PathCollection

        (points,) = (
            c for c in figure.axes[0].collections if isinstance(c, PathCollection)
        )
        colours = np.asarray(points.get_facecolor())
        plt.close(figure)
        return colours

    try:
        plot_genomic.COLOUR_BY = "states"
        recoloured = drawn(df_cnv=instance["df_cnv"])
        plot_genomic.COLOUR_BY = "integer"
        fallback = drawn()
    finally:
        plot_genomic.COLOUR_BY = None

    np.testing.assert_array_equal(recoloured, drawn(df_cnv=None))
    np.testing.assert_array_equal(fallback, drawn(df_cnv=None))
    assert len(np.unique(recoloured, axis=0)) == len(PAIRS)


@pytest.mark.infra
def test_run_cnaster_port_refuses_the_colours_without_the_figure_swaps() -> None:
    from port.scripts.run_cnaster import main

    with pytest.raises(SystemExit):
        main(["config.yaml", "--no-figure-swaps", "--genomic-colours", "states"])
