"""`--no-plots` leaves pyplot as `cnaster`'s `write_fig` does, and writes nothing (#403)."""

from __future__ import annotations

from pathlib import Path

import cnaster.utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from cnaster.utils import write_fig as cnaster_write_fig
from port.patch.utils import discard_fig, write_fig
from port.pipeline import FIGURE_SWAPS, PLOT_OFF_SWAPS, patched


@pytest.mark.smoke
@pytest.mark.parametrize("given", [True, False])
def test_the_figure_is_closed_as_cnasters_closes_it_and_no_file_is_written(
    tmp_path: Path, given: bool
) -> None:
    """Both close the figure handed them; `discard_fig` writes no file."""

    mpl.use("Agg")

    for writer, name in ((cnaster_write_fig, "theirs.pdf"), (discard_fig, "ours.pdf")):
        figure = plt.figure()
        writer(str(tmp_path / name), figure if given else None)
        if given:
            assert not plt.fignum_exists(figure.number), (
                f"{writer.__module__} left it open"
            )
        plt.close("all")

    assert (tmp_path / "theirs.pdf").exists()
    assert not (tmp_path / "ours.pdf").exists()


@pytest.mark.infra
def test_no_plots_rebinds_whichever_write_fig_is_in_place() -> None:
    """After the figure swaps, `--no-plots` replaces port's `write_fig` and restores it."""

    with patched(FIGURE_SWAPS + PLOT_OFF_SWAPS):
        assert cnaster.utils.write_fig is discard_fig
    with patched(FIGURE_SWAPS):
        # NB the row binds its options into port's function (#517)
        assert getattr(cnaster.utils.write_fig, "func", None) is write_fig
    assert cnaster.utils.write_fig.__module__ == "cnaster.utils"
