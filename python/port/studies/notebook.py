"""A study notebook, built and executed reproducibly (#749 WP6).

`copy_start_notebook` and `clone_label_notebook` each carried this builder
and this entry point; they now carry only their cells and their paths.
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["build", "main"]


def build(intro: str, cells: Sequence[tuple[str, str]], out: Path) -> None:
    """The notebook: `intro`'s markdown, then `cells` as `(kind, source)`, executed beside `out`."""
    import nbformat
    from nbclient import NotebookClient

    v4: Any = nbformat.v4

    notebook = v4.new_notebook()
    # NB format 4.4: no cell ids, which are random per build and would make
    #    every rebuild a diff.
    notebook.nbformat_minor = 4
    notebook.cells = [v4.new_markdown_cell(intro)] + [
        v4.new_code_cell(source) if kind == "code" else v4.new_markdown_cell(source)
        for kind, source in cells
    ]
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    NotebookClient(
        notebook, timeout=600, resources={"metadata": {"path": str(out.parent)}}
    ).execute()
    for cell in notebook.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            cell.metadata = {}
            cell.execution_count = None
            for output in cell.get("outputs", []):
                output.pop("execution_count", None)
    writer: Any = nbformat.write
    writer(notebook, out)
    # NB the cells as `ruff format` sets them, so the repository's check
    #    passes on a rebuilt notebook.
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if ruff is not None:
        subprocess.run([ruff, "format", "-q", str(out)], check=True)


def main(
    argv: list[str] | None,
    *,
    description: str,
    summarize: Callable[[dict[str, Any], Path], None],
    data: Path,
    intro: Path,
    cells: Sequence[tuple[str, str]],
    notebook: Path,
) -> None:
    """`[RESULTS.pkl]`: summarize the results into `data` where given, then build `notebook`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("results", type=Path, nargs="?")
    arguments = parser.parse_args(argv)
    if arguments.results is not None:
        with arguments.results.open("rb") as fh:
            summarize(pickle.load(fh), data)
    build(intro.read_text(), cells, notebook)
