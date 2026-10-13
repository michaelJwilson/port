"""One figure font, palette and page geometry for every figure.

The font is read from `pyproject.toml`'s `[tool.port.figures]` (else
:data:`DEFAULT`) and replaces `cnaster.plotting`'s DejaVu Serif and seaborn's
theme (#342); a face matplotlib cannot find is refused. The palette and page
geometry have no `cnaster` counterpart (T- #673 G7, T- #733).
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
    "TRACK_FONT_SIZE",
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
"""Text and marks, axes and ticks, grid lines (T- #673 G7)."""

PAGE_MARGIN = 1.01
"""Inches of margin on every side of the US-letter page, head and folio inside it (T- #740)."""

HEAD_AND_FOOT = 58.0 / 72.27
"""Inches `includeheadfoot` gives the running head and folio (`llncs`: 12 + 16 + 30 pt)."""

PAPER_WIDTH = 8.5 - 2 * PAGE_MARGIN
"""A text column, 6.48 in: the width every paper figure draws at, included at 1:1 (#280, #339, T- #740)."""

TEXT_HEIGHT = 11.0 - 2 * PAGE_MARGIN - HEAD_AND_FOOT
"""The text block's height, 8.18 in (590.99 pt, `pdflatex`) (T- #740)."""

MIN_FONT_SIZE = 8.0
"""Points: every paper figure's text size at 1:1 except genomic tracks' (#743)."""

TRACK_FONT_SIZE = 6.0
"""Points: the RDR and BAF tracks' labels, ticks, clone names and state keys (#743)."""

CAPTION_ROOM = 1.0
"""Inches of the text block a figure leaves for its caption (T- #733, T- #791)."""

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
"""Inches `fit_to_content` leaves at a page's foot for a `MIN_FONT_SIZE` stamp (T- #740)."""


SERIES = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)  # fmt: skip
"""The categorical order, validated: adjacent CVD dE >= 9.1, normal >= 19.6 (#749 WP9)."""

NEUTRAL_COLOUR = "#b5b3ad"
"""`normal`, and the `(1, 1)` state."""


def page_size(page: Page = "full", columns: int = 1) -> tuple[float, float]:
    """Inches, width by height, of one of `columns` figures on a row `page` tall.

    Height is `PAGE_FRACTIONS[page]` of `TEXT_HEIGHT - CAPTION_ROOM`; width
    `PAPER_WIDTH / columns` (T- #733). Raises `ValueError` on a bad `page` or `columns`.
    """
    if page not in PAGE_FRACTIONS:
        message = f"page {page!r} is not one of {sorted(PAGE_FRACTIONS)}"
        raise ValueError(message)
    if columns < 1:
        message = f"columns {columns} is under 1"
        raise ValueError(message)

    return PAPER_WIDTH / columns, PAGE_FRACTIONS[page] * (TEXT_HEIGHT - CAPTION_ROOM)


def fit_to_content(figure: Any) -> None:
    """Cut the page to what its axes draw, `FIT_MARGIN` at head and sides, `STAMP_ROOM` at foot.

    Each axis keeps its size in inches and moves with what it anchors;
    figure-level texts and legends do not move (PR- #715).
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
