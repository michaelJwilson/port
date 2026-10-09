"""Patched `write_fig` against `cnaster`'s, byte for byte at its defaults (#195).

At `FIGURE_DPI` the page geometry stays and the raster coarsens; pixels are
`tests/test_figure_groups.py`'s.
"""

import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pytest
from cnaster.utils import write_fig as upstream
from port.patch.utils import write_fig
from port.patch.utils import write_fig as patched
from port.pipeline import FIGURE_DPI, FIGURE_SWAPS, SWAPS

from tests.figure_checks import CREATION_DATE, wide_rasterized_figure

mpl.use("Agg")


MEDIA_BOX = re.compile(rb"/MediaBox \[([^\]]*)\]")
"""The page geometry, which a resolution change must not move."""


def _written(tmp_path: Path, name: str, writer: Any, **keywords: Any) -> bytes:
    path = tmp_path / f"{name}.pdf"
    writer(str(path), wide_rasterized_figure(), **keywords)

    return CREATION_DATE.sub(b"", path.read_bytes())


@pytest.mark.patch
# NB one figure's form (#403): reruns where this module or the lock changes
@pytest.mark.deprecate
def test_at_its_defaults_it_is_cnasters_function_byte_for_byte(tmp_path: Path) -> None:
    """Called as `cnaster` calls it, the patch writes `cnaster`'s bytes (#517)."""

    assert _written(tmp_path, "upstream", upstream) == _written(
        tmp_path, "patched", patched
    )


@pytest.mark.snapshot
def test_the_default_writes_the_same_page_with_a_coarser_raster(
    tmp_path: Path,
) -> None:
    """At `FIGURE_DPI` the `MediaBox` is unchanged and the file is smaller."""

    assert FIGURE_DPI < 300, "the row no longer lowers the resolution"

    at_300 = _written(tmp_path, "upstream", upstream)
    at_default = _written(tmp_path, "patched", patched, dpi=FIGURE_DPI)

    boxes = (MEDIA_BOX.search(at_300), MEDIA_BOX.search(at_default))
    assert all(boxes), "no page geometry found in one of the files"
    assert boxes[0].group(1) == boxes[1].group(1)  # type: ignore[union-attr]

    assert len(at_default) < len(at_300), (
        f"the default wrote {len(at_default)} bytes against {len(at_300)}"
    )


@pytest.mark.infra
def test_the_figure_swap_is_kept_out_of_the_default_table() -> None:
    """`write_fig` is in `FIGURE_SWAPS` (#195), not `SWAPS`, and no row is in both."""

    names = {swap.name for swap in SWAPS}
    figures = {swap.name: swap.ticket for swap in FIGURE_SWAPS}

    assert "write_fig" not in names
    assert figures["write_fig"] == 195
    assert not names & set(figures), f"in both tables: {names & set(figures)}"


@pytest.mark.infra
def test_png_copies_are_the_same_bytes_on_every_write(tmp_path: Path) -> None:
    """With `png_copy` a PNG lands beside the PDF, byte-identical across writes (#452)."""

    write_fig(str(tmp_path / "plain.pdf"), wide_rasterized_figure())
    assert not (tmp_path / "plain.png").exists()

    write_fig(str(tmp_path / "one.pdf"), wide_rasterized_figure(), png_copy=True)
    write_fig(str(tmp_path / "two.pdf"), wide_rasterized_figure(), png_copy=True)

    one, two = (tmp_path / "one.png").read_bytes(), (tmp_path / "two.png").read_bytes()
    assert one[:8] == b"\x89PNG\r\n\x1a\n"
    assert one == two
