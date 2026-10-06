"""The helpers the two solver-study figures share (T- #617 WP9, T- #673 G1).

`copy_state_plot` and `potts_plot` carried these word for word. Their
statistics (`bars`, `ranks`) live in `port.qa.statistics` and the stamp's
commit in `port.qa.provenance`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from port.qa import provenance

__all__ = ["merged", "stamp", "tab20", "tt"]


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

    `data` is `provenance.digest` of the pickled record; `code` is
    `provenance.commit`. Two figures with one stamp were drawn from one record
    by one commit.
    """
    text = provenance.stamp(provenance.digest(pickle.dumps(record)))
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
