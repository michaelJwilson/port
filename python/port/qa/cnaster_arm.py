"""`cnaster` at its pin, scored by `run_audit` (T- #833): `--no-patch` with #105's row alone installed.

    python -m port.qa.cnaster_arm --sim --sample PATH -- --no-patch --no-plots

`run_audit`'s arguments, run under one `SWAPS` row, `normal_baf_bin_filter`.
At the pin, a bin `cnaster`'s own filter removes leaves its genes
`is_interval` with a null `bin_id`, and the gene-level writer's `int` cast
ends the run with an `IndexError` before `clone_labels.tsv` and
`cnv_seglevel.tsv` are written (#105; measured on `3381575a`). The row's
frame is `cnaster`'s with those genes marked `is_interval = False`; every other
binding is `cnaster`'s, and `--no-patch` is required so that holds.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

ROW = "normal_baf_bin_filter"


def main(argv: Sequence[str] | None = None) -> int:
    from port.pipeline import SWAPS, patched
    from port.scripts import run_audit

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--no-patch" not in arguments:
        msg = "the cnaster arm is --no-patch; pass it after --"
        raise SystemExit(msg)
    with patched(tuple(s for s in SWAPS if s.name == ROW)):
        return run_audit.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
