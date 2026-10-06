"""The helpers the two solver-study figures share (T- #617 WP9, T- #673 G1).

`copy_state_plot` and `potts_plot` carried these word for word. Their
statistics (`bars`, `ranks`) live in `port.qa.statistics` and the stamp's
commit in `port.qa.provenance`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from port.qa import provenance

__all__ = ["key_below", "merged", "stamp", "tab20", "tt"]


def tt(name: str) -> str:
    """`name` in typewriter type inside running text: mathtext's `\\mathtt`, a hyphen kept a hyphen."""
    body = (
        name.replace("_", r"\_")
        .replace("-", r"{\text{-}}")
        .replace("+", "{+}")
        .replace(" ", r"\ ")
    )
    return rf"$\mathtt{{{body}}}$"


def tab20(number: int) -> tuple[float, float, float, float]:
    """The `number`th entry's colour, counting from 1: `tab20`, cycled."""
    from matplotlib import colormaps

    return tuple(colormaps["tab20"]((number - 1) % 20))  # type: ignore[return-value]


def merged(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One record from streams of the same manifest: their rows together, a realization done in any of them."""
    record = dict(records[0])
    for other in records[1:]:
        if Path(other["manifest"]).stem != Path(record["manifest"]).stem:
            msg = f"{other['manifest']} is not {record['manifest']}"
            raise ValueError(msg)
        record["problems"] = {**other["problems"], **record["problems"]}
        record["rows"] = [*record["rows"], *other["rows"]]
        record["done"] = sorted({*record["done"], *other["done"]})
    return record


def stamp(fig: Any, record: dict[str, Any]) -> str:
    """The figure's reference, drawn on it: a hash of the record it plots and the code's commit.

    `data` is `port.studies.records.digest` of the record; `code` is
    `provenance.commit`. Two figures with one stamp were drawn from one record
    by one commit.
    """
    from port.studies import records

    text = provenance.stamp(records.digest(record))
    fig.text(
        0.995,
        0.005,
        text,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
        family="monospace",
    )
    return text


def key_below(
    ax: Any,
    title: str,
    stages: list[tuple[dict[str, Any], str]],
    entries: list[tuple[str, Any, float]],
    notes: list[str],
    *,
    fontsize: float = 7.5,
    top: float = -0.17,
    row: float = 0.055,
    columns: int = 2,
    marks: bool = True,
) -> None:
    """A key under `ax`, its `title` above the axes' top right: the stages' markers in their own column, then the methods in two columns of name and missed %.

    `stages` are `(plot keywords, label)`, drawn as a marker or a line; `entries` are
    `(name, colour, missed %)`, filled down the first column then the second;
    `notes` are footnote lines under both. Coordinates are `ax`'s, so the
    figure's bottom margin must hold `top - row * (rows + notes)`.
    """
    from matplotlib.lines import Line2D

    t = ax.transAxes

    def text(x: float, y: float, s: str, **kw: Any) -> None:
        ax.text(
            x,
            y,
            s,
            transform=t,
            va="center",
            clip_on=False,
            **{"fontsize": fontsize, **kw},
        )

    def mark(x: float, y: float, keywords: dict[str, Any]) -> None:
        if keywords.get("line"):
            line = Line2D([x - 0.02, x + 0.02], [y, y], transform=t, color=keywords.get("color", "k"),
                          lw=keywords.get("lw", 0.9), clip_on=False)  # fmt: skip
        else:
            plot = {k: v for k, v in keywords.items() if k != "line"}
            line = Line2D(
                [x], [y], transform=t, linestyle="none", clip_on=False, **plot
            )
        ax.add_line(line)

    # NB the title sits above the axes' top-right corner, clear of the x label and the key
    ax.text(1.0, 1.01, title, transform=t, ha="right", va="bottom", fontsize=fontsize)
    header = top
    # NB one column where the panel is narrow (half a 122 mm page), two where it is wide
    rows = -(-len(entries) // columns)
    # NB without the stage column the methods take its room too
    spans = (
        ((0.38 if marks else 0.02, 1.0),)
        if columns == 1
        else ((0.30, 0.62), (0.67, 0.99))
    )
    # NB a figure whose panels share their stages keys them once (`marks` on one panel only)
    for k, (keywords, label) in enumerate(stages if marks else []):
        y = header - row * (k + 1)
        mark(0.02, y, keywords)
        text(0.05 if columns == 2 else 0.08, y, label)
    for k, (name, colour, missed) in enumerate(entries):
        left, right = spans[k // rows]
        y = header - row * (k % rows + 1)
        mark(left, y, {"marker": "o", "color": colour, "markersize": 5})
        text(left + (0.05 if columns == 1 else 0.025), y, name)
        text(right, y, f"{missed:.1f}%", ha="right")
    bottom = header - row * (max(rows, len(stages)) + 1)
    for k, note in enumerate(notes):
        text(
            0.0,
            bottom - row * k,
            note,
            ha="left",
            color="0.25",
            fontsize=fontsize - 0.5,
        )
