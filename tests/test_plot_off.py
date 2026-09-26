"""`--no-plots` leaves pyplot as `cnaster`'s `write_fig` does, and writes nothing (#403)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.patch
@pytest.mark.parametrize("given", [True, False])
def test_the_figure_is_closed_as_cnasters_closes_it_and_no_file_is_written(
    tmp_path: Path, given: bool
) -> None:
    """Both close the figure they were handed; `discard_fig` writes no file.

    With no figure given, `cnaster` builds and writes an empty one; there is
    nothing of the run's to close, and `discard_fig` touches no figure at all.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from cnaster.utils import write_fig
    from port.patch.utils import discard_fig

    for writer, name in ((write_fig, "theirs.pdf"), (discard_fig, "ours.pdf")):
        figure = plt.figure()
        writer(str(tmp_path / name), figure if given else None)
        if given:
            assert not plt.fignum_exists(figure.number), (
                f"{writer.__module__} left it open"
            )
        plt.close("all")

    assert (tmp_path / "theirs.pdf").exists()
    assert not (tmp_path / "ours.pdf").exists()


@pytest.mark.patch
def test_no_plots_rebinds_whichever_write_fig_is_in_place() -> None:
    """Installed after the figure swaps, it replaces port's `write_fig` and
    puts it back on the way out."""
    import cnaster.utils
    from port.patch.utils import discard_fig, write_fig
    from port.pipeline import FIGURE_SWAPS, PLOT_OFF_SWAPS, patched

    with patched(FIGURE_SWAPS + PLOT_OFF_SWAPS):
        assert cnaster.utils.write_fig is discard_fig
    with patched(FIGURE_SWAPS):
        assert cnaster.utils.write_fig is write_fig
    assert cnaster.utils.write_fig.__module__ == "cnaster.utils"
