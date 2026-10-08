"""Figure checks shared across tests: what a written page or panel must show."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import genomic_plot_instance, integer_copies


def mirror_key_holds(legend_ax: Any, edge_ax: Any) -> None:
    """The mirror key's geometry on `legend_ax`: its two swatches one above
    the other, not overlapping, their left edges on `edge_ax`'s (0.5 px);
    `MIRROR` right of them, its centre on the white between them (0.5 px),
    and left of the colour bar's title (PR- #715)."""
    from port.patch.plot_copy_number_profile import MIRROR, TITLE

    renderer = legend_ax.figure.canvas.get_renderer()
    upper, lower = sorted(
        (p.get_window_extent(renderer) for p in legend_ax.patches[:4:2]),
        key=lambda b: -b.y0,
    )
    (mirror,) = [t for t in legend_ax.texts if t.get_text() == MIRROR]
    (title,) = [t for t in legend_ax.texts if t.get_text() == TITLE]
    label = mirror.get_window_extent(renderer)
    left = edge_ax.get_window_extent(renderer).x0

    assert MIRROR == "Local Mirror"
    assert upper.x0 == pytest.approx(left, abs=0.5)
    assert lower.x0 == pytest.approx(left, abs=0.5)
    assert lower.y1 < upper.y0
    assert (label.y0 + label.y1) / 2 == pytest.approx(
        (lower.y1 + upper.y0) / 2, abs=0.5
    )
    assert label.x0 > max(upper.x1, lower.x1)
    assert label.x1 < title.get_window_extent(renderer).x0


CREATION_DATE = re.compile(rb"/CreationDate \(D:\d+Z?\)")
"""matplotlib writes a clock into every PDF; #103 owns pinning it."""


def wide_rasterized_figure() -> Any:
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


def genomic_plot_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
    instance = genomic_plot_instance()
    keywords = {
        "res_combine": instance["result"],
        "df_cnv": integer_copies(instance["rng"], 24, 3),
    }

    return instance["arguments"], keywords


def recorded_combined_calls(
    tmp_path: Path, n_clones: int = 3, tall: float = 1.0
) -> tuple[Any, Any]:
    """A run's three recorded calls on the 3 by 3 fixture, its rows `tall`
    times as far apart as its columns, and its slide."""
    from cnaster.he import get_he_image
    from port.extensions.combined_figure import Call, Recorded
    from port.sim.he_slide import mock_he, write_he_slide
    from port.sim.truth import clone_bands

    arguments, keywords = genomic_plot_arguments()
    n_spots = arguments[1].shape[2]
    rows, columns = np.unravel_index(np.arange(n_spots), (3, 3))
    coords = np.column_stack([rows, tall * columns]).astype(float)
    assignment = pd.Series([f"clone {k % n_clones}" for k in range(n_spots)])
    write_he_slide(mock_he(clone_bands(3, 3, 3), (3, 3), seed=1), tmp_path)
    recorded = Recorded(
        genomic=Call(arguments, keywords),
        spatial=Call((coords, assignment), {}),
        profile=Call((keywords["df_cnv"].assign(START=0, END=1),), {}),
    )

    return recorded, get_he_image(str(tmp_path), pos=None)


def panels_in_order(letters: list[Any], panels: dict[str, list[Any]]) -> list[str]:
    """`panels`' names top to bottom by their axes' tops, each lettered in
    turn: the k-th letter from the head reads `(a)`, `(b)`, ... and sits
    under the panel before its own and over the panel after it."""
    figure = next(iter(panels.values()))[0].get_figure(root=True)
    renderer = figure.canvas.get_renderer()

    def extent(axes: list[Any]) -> tuple[float, float]:
        boxes = [ax.get_window_extent(renderer) for ax in axes]
        return min(b.y0 for b in boxes), max(b.y1 for b in boxes)

    order = sorted(panels, key=lambda name: -extent(panels[name])[1])
    boxes = sorted(
        (t.get_window_extent(renderer) for t in letters), key=lambda b: -b.y1
    )
    texts = sorted(letters, key=lambda t: -t.get_window_extent(renderer).y1)

    assert [t.get_text() for t in texts] == [f"({k})" for k in "abc"[: len(order)]]
    for k, box in enumerate(boxes):
        middle = (box.y0 + box.y1) / 2
        if k > 0:
            assert middle < extent(panels[order[k - 1]])[0]
        if k + 1 < len(order):
            assert middle > extent(panels[order[k + 1]])[1]
    return order


def compare_run_artifacts(
    baseline: Path, patched_output: Path
) -> tuple[list[str], list[str]]:
    """Every artifact of two runs, as (bitwise, differing) names.

    A PDF counts as reproduced when it agrees with its creation timestamp
    removed; everything else has to agree raw.
    """
    same: list[str] = []
    differ: list[str] = []

    for left in sorted(path for path in baseline.rglob("*") if path.is_file()):
        right = patched_output / left.relative_to(baseline)

        if not right.exists():
            differ.append(f"{left.name} (missing)")
            continue

        first, second = left.read_bytes(), right.read_bytes()

        if left.suffix == ".pdf":
            first, second = (
                CREATION_DATE.sub(b"", first),
                CREATION_DATE.sub(b"", second),
            )

        (same if first == second else differ).append(left.name)

    return same, differ
