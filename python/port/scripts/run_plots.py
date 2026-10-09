"""`run_plots`: redraw a run's figures from its `cnamaste.h5` (T- #817).

    run_plots OUTPUT/cnamaste.h5 [--out DIR] [--only NAME ...]

Each page kept under `figures/` is redrawn to the path the run wrote, under
`--out` (default: the file's directory); `--no-plots` runs keep pages too.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Draw every page kept in the file, or those `--only` names."""
    from port.extensions.figure_record import replay

    parser = argparse.ArgumentParser(prog="run_plots", description=__doc__)
    parser.add_argument("file", type=Path, help="a run's cnamaste.h5")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the pages go; the file's directory by default",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=(),
        metavar="NAME",
        help="these pages, by name (e.g. clones_genomic)",
    )
    arguments = parser.parse_args(argv)

    out = arguments.file.parent if arguments.out is None else arguments.out
    written = replay(arguments.file, out, tuple(arguments.only))
    for path in written:
        print(path)
    if not written:
        print(f"run_plots: {arguments.file} keeps no page", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
