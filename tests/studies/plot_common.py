"""The plot helpers the study figures share (T- #617 WP9).

`copy_state_plot` and `potts_plot` carried these three word for word.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["ranks", "stamp", "tt"]


def tt(name: str) -> str:
    """`name` in typewriter type inside running text: mathtext's `\\mathtt`, a hyphen kept a hyphen."""
    body = (
        name.replace("_", r"\_")
        .replace("-", r"{\text{-}}")
        .replace("+", "{+}")
        .replace(" ", r"\ ")
    )
    return rf"$\mathtt{{{body}}}$"


def ranks(values: dict[str, float]) -> dict[str, int]:
    """1 for the lowest value, ties sharing the lower rank."""
    order = pd.Series(values, dtype=float).dropna().rank(method="min")
    return {str(k): int(v) for k, v in order.items()}


def stamp(fig: Any, record: dict[str, Any]) -> str:
    """The figure's reference, drawn on it: a hash of the record it plots and the code's commit.

    `data` is the first 8 hex digits of SHA-256 over the pickled record;
    `code` is the repository's short commit, `+` where the tree differs from
    it. Two figures with one stamp were drawn from one record by one commit.
    """
    import hashlib
    import subprocess

    data = hashlib.sha256(pickle.dumps(record)).hexdigest()[:8]
    here = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=here,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=here,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    text = f"data {data} · code {commit or 'unknown'}{'+' if dirty else ''}"
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
