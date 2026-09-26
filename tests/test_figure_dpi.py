"""`write_fig` at a resolution the figures are read at (#195).

**The one `port` replacement that changes its output**, so the claim here is
in two halves. At the same `dpi`, with the grouping put back, it is
`cnaster`'s function byte for byte -- which is what says the patch is two
defaults and not a rewrite. At its own defaults it writes the same page with
a coarser raster, which is the 4.39x.

The second default -- one rasterizing group per axes rather than one per
artist -- is `tests/test_figure_groups.py`, because its referee is a pixel
comparison rather than a byte one.
"""

import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

CREATION_DATE = re.compile(rb"/CreationDate \(D:\d+Z?\)")
"""matplotlib writes a clock into every PDF; #103 owns pinning it."""

MEDIA_BOX = re.compile(rb"/MediaBox \[([^\]]*)\]")
"""The page geometry, which a resolution change must not move."""


def _figure() -> Any:
    """A panel of the shape the genomic plots write: wide, and rasterized.

    Rasterized because that is what puts the PDF backend into mixed mode,
    where it allocates a full-figure `RendererAgg` per rasterizing group --
    the allocation the dpi decides the size of.
    """
    import matplotlib.pyplot as plt

    generator = np.random.default_rng(7)
    figure, axes = plt.subplots(figsize=(20, 4), dpi=300, facecolor="white")

    for _ in range(4):
        axes.scatter(
            generator.random(2_000),
            generator.random(2_000),
            s=2,
            rasterized=True,
        )

    axes.set_xlabel("position")
    axes.set_title("a panel of the shape the genomic plots write")

    return figure


def _written(tmp_path: Path, name: str, writer: Any, **keywords: Any) -> bytes:
    path = tmp_path / f"{name}.pdf"
    writer(str(path), _figure(), **keywords)

    return CREATION_DATE.sub(b"", path.read_bytes())


@pytest.mark.patch
def test_at_the_same_dpi_it_is_cnasters_function_byte_for_byte(tmp_path: Path) -> None:
    """The patch is a default, not a rewrite.

    Handed `cnaster`'s own `dpi=300`, and with the grouping put back, the two
    write identical files, so every difference a run sees comes from the two
    defaults and none from the code. That is what makes the change reviewable
    as two numbers rather than as a diff, and what would catch a patch that
    quietly dropped `transparent` or the tight bounding box along with them.
    """
    from cnaster.utils import write_fig as upstream
    from port.patch.utils import write_fig as patched

    assert _written(tmp_path, "upstream", upstream, dpi=300) == _written(
        tmp_path, "patched", patched, dpi=300, group_rasters=False
    )


@pytest.mark.snapshot
def test_the_default_writes_the_same_page_with_a_coarser_raster(
    tmp_path: Path,
) -> None:
    """The page is where it was; the raster behind it is a quarter the pixels.

    Pinned as `snapshot` because it judges nothing scientific: it records
    that the only thing the default moves is the resolution. The page
    geometry is the load-bearing part -- a dpi that changed the `MediaBox`
    would be cropping the figure rather than sampling it, and the difference
    is invisible in a file size.
    """
    from cnaster.utils import write_fig as upstream
    from port.patch.utils import FIGURE_DPI
    from port.patch.utils import write_fig as patched

    assert FIGURE_DPI < 300, "the default no longer lowers the resolution"

    at_300 = _written(tmp_path, "upstream", upstream)
    at_default = _written(tmp_path, "patched", patched, group_rasters=False)

    boxes = (MEDIA_BOX.search(at_300), MEDIA_BOX.search(at_default))
    assert all(boxes), "no page geometry found in one of the files"
    assert boxes[0].group(1) == boxes[1].group(1)  # type: ignore[union-attr]

    assert len(at_default) < len(at_300), (
        f"the default wrote {len(at_default)} bytes against {len(at_300)}"
    )


@pytest.mark.infra
def test_the_figure_swap_is_kept_out_of_the_default_table() -> None:
    """`CLAUDE.md` forbids a silent behaviour change, and this is one.

    Every row of `SWAPS` reproduces `cnaster` bitwise, which is what the
    whole-run test asserts. A figure written at half the dpi is a different
    file by design, so it lives in `FIGURE_SWAPS`.

    **The entry point installs it by default now**, and this test is what
    stops that becoming a merge of the two tables. The default decides what
    a user gets; the table decides what can still be claimed. Keeping them
    apart is what lets `run_cnaster_port` be 47 per cent faster while
    `SWAPS` remains the set that reproduces `cnaster` -- and `--no-figure-swaps`
    is the arm that does.

    The table has a second row since #299, `plot_clones_genomic`, which
    changes the figure for the same reason, so what is pinned is that
    `write_fig` is in it with its ticket and that no row is in both tables.
    """
    from port.pipeline import FIGURE_SWAPS, SWAPS

    names = {swap.name for swap in SWAPS}
    figures = {swap.name: swap.ticket for swap in FIGURE_SWAPS}

    assert "write_fig" not in names
    assert figures["write_fig"] == 195
    assert not names & set(figures), f"in both tables: {names & set(figures)}"
