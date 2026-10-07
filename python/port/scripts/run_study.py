"""`run_study --<study> [ARGS ...]`: one study, by hand, outside the suite (#489, #492; T- #673 G5).

A study measures `snakes_and_ladders`' machinery or `port`'s on problems port
captured or drew, and states the document its numbers are in. Each is a
module of `port.studies` whose `main(argv)` takes the arguments after its
flag with its own parser; `run_study --<study> --help` prints them.

    run_study --potts-stream MANIFEST OUT_DIR [--problems N] ...
    run_study --copy-state-plot STREAM.record [EARLIER.record ...]
    run_study --population run MANIFEST OUT ...
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

STUDIES = (
    "calicost_figures",
    "clone_label_notebook",
    "clone_labels",
    "clone_starts",
    "cna_lengths",
    "copy_start_notebook",
    "copy_starts",
    "copy_state_plot",
    "copy_state_stream",
    "field_strength",
    "metrics_history",
    "paper_figures",
    "population",
    "potts_plot",
    "potts_stream",
)
"""The `port.studies` modules a flag runs; the rest are their parts."""


def _usage() -> str:
    flags = "\n".join(f"  --{name.replace('_', '-')}" for name in STUDIES)
    return f"usage: run_study --<study> [ARGS ...]\n\nstudies:\n{flags}\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study the first argument names on the rest."""
    args = list(sys.argv[1:] if argv is None else argv)
    flag = args[0] if args else ""
    name = flag.removeprefix("--").replace("-", "_")
    if not flag.startswith("--") or name not in STUDIES:
        sys.stderr.write(_usage())
        return 0 if flag in {"-h", "--help"} else 2
    module = importlib.import_module(f"port.studies.{name}")
    status = module.main(args[1:])
    return int(status or 0)


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
