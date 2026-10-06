"""One figure font, palette and page width, the font stated in `pyproject.toml` (`[tool.port.figures]`).

`cnaster.plotting` sets `font.family` to DejaVu Serif when it is imported and
its plots set seaborn's theme when they run, and `port`'s combined figure
resets to matplotlib's defaults (`page_style`, #342), so a run's figures
carried two faces. :func:`figure_rc` is the one set of `rcParams` every
figure takes: the face, its family and the matching math fonts.

The values are read from `pyproject.toml` where the package runs from a
checkout, and are :data:`DEFAULT` otherwise, which a test holds equal to the
file. A face matplotlib cannot find is refused by name rather than left to
fall back silently to another.

The palette (`INK`, `MUTED`, `GRID`, `axes_style`) and the page width
(`PAPER_WIDTH`) are page geometry and style with no `cnaster` counterpart;
`port.sim.analysis`, `port.patch.plot_genomic` and the paper figures each
held a copy until T- #673 G7.
"""

from __future__ import annotations

import contextlib
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

__all__ = [
    "DEFAULT",
    "FIT_MARGIN",
    "GRID",
    "INK",
    "LLNCS_TEXT_WIDTH_MM",
    "MUTED",
    "PAPER_WIDTH",
    "STAMP_ROOM",
    "apply",
    "axes_style",
    "figure_font",
    "figure_rc",
    "fit_to_content",
    "stated",
]

DEFAULT: dict[str, str] = {"family": "serif", "font": "STIXGeneral", "mathtext": "stix"}
"""`[tool.port.figures]` as shipped; used where no `pyproject.toml` is found."""

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
"""Text and marks, axes and ticks, grid lines: the truth page's and the paper figures' one palette (T- #673 G7)."""

LLNCS_TEXT_WIDTH_MM = 122.0
"""`\\textwidth` of `\\documentclass[runningheads,11pt]{llncs}`, fixed by the
class whatever the paper (#339)."""

PAPER_WIDTH = LLNCS_TEXT_WIDTH_MM / 25.4
"""A text column, 4.80 in: the width `combined_figure` draws at (#280, #339).

Measured from the genomic figures `docs/plots/` then tracked: 20.03 in
wide, so `\\includegraphics[width=\\linewidth]` scales them by **0.240** and
a 10 pt tick label lands at **2.4 pt** on the page. At a text column the
figure is included at 1:1, so a declared size is the size on the page and
nothing has to be undone at the point of inclusion.
"""


FIT_MARGIN = 0.03
"""Inches of white `fit_to_content` leaves at a page's head and sides."""

STAMP_ROOM = 0.1
"""Inches `fit_to_content` leaves at a page's foot: a 6 pt stamp's row."""


def fit_to_content(figure: Any) -> None:
    """The page cut to what its axes draw, `FIT_MARGIN` at the head and
    sides and `STAMP_ROOM` at the foot; each axis keeps its size in inches.

    For a page whose axes hold their aspect (a spatial map, `set_aspect`
    "equal"), which otherwise leaves the white its aspect does not fill as
    bands round the axes. What an axis anchors -- its legend, title and
    texts -- moves with it; a figure-level text or legend does not, so a
    caller anchors its key to an axis (PR- #701 follow-up).
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    dpi = figure.dpi
    box = figure.get_tightbbox(renderer)
    kept = [(ax, ax.get_window_extent(renderer).frozen()) for ax in figure.get_axes()]
    width = box.width + 2 * FIT_MARGIN
    height = box.height + FIT_MARGIN + STAMP_ROOM
    figure.set_size_inches(width, height)

    for ax, at in kept:
        x0 = at.x0 / dpi - box.x0 + FIT_MARGIN
        y0 = at.y0 / dpi - box.y0 + STAMP_ROOM
        ax.set_position(
            (x0 / width, y0 / height, at.width / dpi / width, at.height / dpi / height)
        )


def axes_style(ax: Any, *, labelsize: float = 8, grid: bool = True) -> None:
    """Open axes in `MUTED`, labels in `INK`, and a `GRID` grid behind the marks unless `grid` is off."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=labelsize)
    if grid:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def stated() -> dict[str, str]:
    """`[tool.port.figures]` from the checkout's `pyproject.toml`, else :data:`DEFAULT`."""
    if not PYPROJECT.exists():
        return dict(DEFAULT)

    section = tomllib.loads(PYPROJECT.read_text()).get("tool", {}).get("port", {})
    figures = section.get("figures")

    return dict(DEFAULT) if figures is None else {k: str(v) for k, v in figures.items()}


def figure_rc() -> dict[RcKeyType, Any]:
    """The `rcParams` of the stated face: family, the face first in it, math fonts."""
    import matplotlib.font_manager as fm

    style = stated()
    family, font = style["family"], style["font"]

    try:
        fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
    except ValueError as error:
        msg = f"[tool.port.figures] font {font!r} is not installed for matplotlib"
        raise ValueError(msg) from error

    return {
        "font.family": family,
        cast("RcKeyType", f"font.{family}"): [font],
        "mathtext.fontset": style["mathtext"],
    }


def apply() -> None:
    """Set the stated face in `matplotlib.rcParams`."""
    import matplotlib as mpl

    mpl.rcParams.update(figure_rc())


@contextlib.contextmanager
def figure_font() -> Iterator[None]:
    """The stated face for the block, after `cnaster.plotting` has set its own."""
    import cnaster.plotting  # noqa: F401 -- sets `font.family` on import
    import matplotlib as mpl

    with mpl.rc_context(figure_rc()):
        yield
