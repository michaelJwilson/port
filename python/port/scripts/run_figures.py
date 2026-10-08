"""`run_figures`: regenerate the dev instance's figures, by default under `.cache/plots/` (T- #673 G4).

    run_figures [--out DIR] [--cnaster]

Runs `run_cnaster_port` (or plain `cnaster` with `--cnaster`) on the dev
instance and its copy lattice and copies the PNGs into `DIR`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and copy its figures into `--out`."""
    from port.qa.benchmark import figures
    from port.qa.provenance import PLOTS

    parser = argparse.ArgumentParser(prog="run_figures", description=__doc__)
    parser.add_argument("--cnaster", action="store_true",
                        help="run plain cnaster rather than run_cnaster_port")  # fmt: skip
    parser.add_argument(
        "--out", type=Path, default=PLOTS, help="where the figures go (untracked)"
    )
    arguments = parser.parse_args(argv)
    figures(arguments.out, cnaster=arguments.cnaster)
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
