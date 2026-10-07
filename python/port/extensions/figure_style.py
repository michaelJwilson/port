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

The palette (`INK`, `MUTED`, `GRID`, `axes_style`) and the page geometry
(`PAPER_WIDTH`, `TEXT_HEIGHT`, `CAPTION_ROOM`, `page_size`) are style with no
`cnaster` counterpart; `port.sim.analysis`, `port.patch.plot_genomic` and the
paper figures each held a copy of the palette until T- #673 G7, and
`combined_figure` the text height until T- #733.
"""

from __future__ import annotations

import contextlib
import tomllib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, cast

from port.extensions.repository import ROOT

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

__all__ = [
    "CAPTION_ROOM",
    "DEFAULT",
    "FIT_MARGIN",
    "GRID",
    "HEAD_AND_FOOT",
    "INK",
    "MIN_FONT_SIZE",
    "MUTED",
    "PAGE_FRACTIONS",
    "PAGE_MARGIN",
    "PAPER_WIDTH",
    "STAMP_ROOM",
    "TEXT_HEIGHT",
    "Page",
    "apply",
    "axes_style",
    "figure_font",
    "figure_rc",
    "fit_to_content",
    "page_size",
    "stated",
]

DEFAULT: dict[str, str] = {"family": "serif", "font": "STIXGeneral", "mathtext": "stix"}
"""`[tool.port.figures]` as shipped; used where no `pyproject.toml` is found."""

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
"""Text and marks, axes and ticks, grid lines: the truth page's and the paper figures' one palette (T- #673 G7)."""

PAGE_MARGIN = 1.01
"""Inches of margin on every side of the paper's US-letter page, the running
head and folio inside it (`geometry`'s `includeheadfoot`): the submission's
1 in minimum with 0.01 in to spare (T- #740)."""

HEAD_AND_FOOT = 58.0 / 72.27
"""Inches of the page `includeheadfoot` gives the running head and folio:
`llncs`'s `\\headheight` 12 pt, `\\headsep` 16 pt and `\\footskip` 30 pt."""

PAPER_WIDTH = 8.5 - 2 * PAGE_MARGIN
"""A text column, 6.48 in (468.31 pt, `pdflatex`): the width every paper
figure draws at (#280, #339, T- #740).

Measured from the genomic figures `docs/plots/` then tracked: 20.03 in
wide, so `\\includegraphics[width=\\linewidth]` scales them by **0.240** and
a 10 pt tick label lands at **2.4 pt** on the page. At a text column the
figure is included at 1:1, so a declared size is the size on the page and
nothing has to be undone at the point of inclusion.
"""

TEXT_HEIGHT = 11.0 - 2 * PAGE_MARGIN - HEAD_AND_FOOT
"""The text block's height, 8.18 in (590.99 pt, `pdflatex`) (T- #740)."""

MIN_FONT_SIZE = 10.0
"""Points: the submission's smallest text, and every paper figure's text
size at 1:1 (T- #740)."""

CAPTION_ROOM = 1.5
"""Inches of the text block a figure leaves for its caption (T- #733).

A page drawn `PAPER_WIDTH` by `TEXT_HEIGHT` and included at
`width=\\linewidth` under a one-line `\\caption` is too large for the page,
by 23.0 pt on `llncs`'s 122 by 193 mm block (`pdflatex`); 1.5 in leaves a
caption of several lines with `\\textfloatsep`.
"""

Page = Literal["third", "half", "three_quarters", "full"]
"""The heights a paper figure is drawn at, as a share of the text block."""

PAGE_FRACTIONS: dict[Page, float] = {
    "third": 1 / 3,
    "half": 1 / 2,
    "three_quarters": 3 / 4,
    "full": 1.0,
}
"""Each `Page`'s share of the text block less `CAPTION_ROOM`."""


FIT_MARGIN = 0.03
"""Inches of white `fit_to_content` leaves at a page's head and sides."""

STAMP_ROOM = 0.17
"""Inches `fit_to_content` leaves at a page's foot: a `MIN_FONT_SIZE` stamp's
row, 12 pt (T- #740)."""


def page_size(page: Page = "full", columns: int = 1) -> tuple[float, float]:
    """Inches, width by height, of one of `columns` figures on a row `page` tall.

    The row is `PAGE_FRACTIONS[page]` of `TEXT_HEIGHT - CAPTION_ROOM`, so
    "full" is the whole text block less its caption and every share leaves
    its caption that room in proportion; each figure is `PAPER_WIDTH /
    columns` wide, included at `width=\\linewidth` in a minipage of
    `1/columns` of the line at 1:1, e.g. the two spatial maps on a "third"
    row, and a 2x2 grid at "half" or "three_quarters" (T- #733).
    """
    if page not in PAGE_FRACTIONS:
        message = f"page {page!r} is not one of {sorted(PAGE_FRACTIONS)}"
        raise ValueError(message)
    if columns < 1:
        message = f"columns {columns} is under 1"
        raise ValueError(message)

    return PAPER_WIDTH / columns, PAGE_FRACTIONS[page] * (TEXT_HEIGHT - CAPTION_ROOM)


def fit_to_content(figure: Any) -> None:
    """The page cut to what its axes draw, `FIT_MARGIN` at the head and
    sides and `STAMP_ROOM` at the foot; each axis keeps its size in inches.

    For a page whose axes hold their aspect (a spatial map, `set_aspect`
    "equal"), which otherwise leaves the white its aspect does not fill as
    bands round the axes. What an axis anchors -- its legend, title and
    texts -- moves with it; a figure-level text or legend does not, so a
    caller anchors its key to an axis (PR- #715).
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


PYPROJECT = ROOT / "pyproject.toml"


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
