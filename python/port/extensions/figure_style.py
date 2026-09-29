"""One figure font, stated in `pyproject.toml` (`[tool.port.figures]`).

`cnaster.plotting` sets `font.family` to DejaVu Serif when it is imported and
its plots set seaborn's theme when they run, and `port`'s combined figure
resets to matplotlib's defaults (`page_style`, #342), so a run's figures
carried two faces. :func:`figure_rc` is the one set of `rcParams` every
figure takes: the face, its family and the matching math fonts.

The values are read from `pyproject.toml` where the package runs from a
checkout, and are :data:`DEFAULT` otherwise, which a test holds equal to the
file. A face matplotlib cannot find is refused by name rather than left to
fall back silently to another.
"""

from __future__ import annotations

import contextlib
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

__all__ = ["DEFAULT", "apply", "figure_font", "figure_rc", "stated"]

DEFAULT: dict[str, str] = {"family": "serif", "font": "STIXGeneral", "mathtext": "stix"}
"""`[tool.port.figures]` as shipped; used where no `pyproject.toml` is found."""

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
