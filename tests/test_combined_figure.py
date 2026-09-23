"""`port.extensions.combined_figure`: one page from a run's figures (#309).

Two claims a reader relies on. **Drawing into a subfigure changes the layout
and nothing drawn**: every point, segment and colour of (a) is the one the
standalone `clones_genomic` page carries. **Recording does not touch the
run**: each wrapper calls through, returns what it wraps, and is removed on
exit. The page itself is checked for what #280 asks of it -- a text column
wide, no text above the cap -- which is `smoke`: it says the page composes,
not that anything on it is right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.test_plot_genomic_patch import _drawn, _instance, _integer_copies


def _genomic_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
    instance = _instance()
    keywords = {
        "res_combine": instance["result"],
        "df_cnv": _integer_copies(instance["rng"], 24, 3),
    }

    return instance["arguments"], keywords


@pytest.mark.patch
def test_a_subfigure_draws_what_the_standalone_page_draws(cnaster_config: None) -> None:
    """Every array of (a), bitwise, against the page `clones_genomic.pdf` is."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.patch.plot_genomic import plot_clones_genomic

    arguments, keywords = _genomic_arguments()
    page = plot_clones_genomic(*arguments, **keywords)

    host = plt.figure(figsize=(6.5, 4.0), layout="constrained")
    panels: Any = host.subfigures(2, 1)
    top = panels[0]
    plot_clones_genomic(*arguments, **{**keywords, "figure": top})

    ours, theirs = _drawn(host), _drawn(page)

    assert len(ours) == len(theirs)

    for index, (mine, reference) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(mine, reference, err_msg=f"array {index}")


@pytest.mark.infra
def test_recording_calls_through_and_restores() -> None:
    """A recorded call returns the wrapped function's figure; the names return."""
    import matplotlib as mpl

    mpl.use("Agg")
    import cnaster.scripts.run_cnaster as script
    import port.patch.plot_genomic as genomic
    import port.patch.plotting as spatial
    from port.extensions.combined_figure import recording

    before = (
        genomic.plot_clones_genomic,
        spatial.plot_clones_spatial,
        script.plot_copy_number_profile,
    )
    coords = np.column_stack([np.arange(4.0), np.zeros(4)])
    assignment = pd.Series(["clone 0", "clone 1"] * 2)

    with recording() as recorded:
        figure = spatial.plot_clones_spatial(coords, assignment)

    assert figure.axes[0].collections, "the wrapped function drew nothing"
    assert recorded.spatial is not None
    assert recorded.spatial.args[0] is coords
    assert recorded.calls == {"spatial": 1}
    assert (
        genomic.plot_clones_genomic,
        spatial.plot_clones_spatial,
        script.plot_copy_number_profile,
    ) == before


@pytest.mark.smoke
def test_the_page_is_a_column_wide_with_capped_text(
    cnaster_config: None, tmp_path: Path
) -> None:
    """6.5 in wide, four panels labelled (a) to (d), no text over 6 pt else."""
    import matplotlib as mpl

    mpl.use("Agg")
    from cnaster.he import get_he_image
    from matplotlib.text import Text
    from port.extensions.combined_figure import (
        FONT_SIZE,
        LABEL_SIZE,
        Call,
        Recorded,
        combined_figure,
    )

    from tests.fixtures import clone_bands
    from tests.he_slide import mock_he, write_he_slide

    arguments, keywords = _genomic_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, columns]).astype(float)
    assignment = pd.Series(
        [f"clone {c}" for c in keywords["res_combine"]["new_assignment"]]
    )

    write_he_slide(mock_he(clone_bands(3, 3, 3), (3, 3), seed=1), tmp_path)
    frame = get_he_image(str(tmp_path), pos=None)

    recorded = Recorded(
        genomic=Call(arguments, keywords),
        spatial=Call((coords, assignment), {}),
        profile=Call((keywords["df_cnv"].assign(START=0, END=1),), {}),
    )
    figure = combined_figure(recorded, frame)

    assert figure.get_size_inches()[0] == pytest.approx(6.5)

    titles = [panel._suptitle.get_text() for panel in figure.subfigs[:2]]
    titles += [panel._suptitle.get_text() for panel in figure.subfigs[2].subfigs]
    assert titles == ["(a)", "(b)", "(c)", "(d)"]

    sizes = [
        text.get_fontsize()
        for text in figure.findobj(Text)
        if text.get_text() not in titles
    ]
    assert max(sizes) <= FONT_SIZE < LABEL_SIZE
