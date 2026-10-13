"""`tests/sim_audit.py::main` and `score`, kept because ledger rows name them (#362, T- #673 G3).

Both delegate to `port.qa.audit` / `run_audit --sim`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from port.qa.audit import score_sample as score
from port.qa.scripts.run_audit import main as run_audit

__all__ = ["main", "score"]


def main(argv: Sequence[str] | None = None) -> int:
    """`run_audit --sim` with these arguments."""

    return run_audit(["--sim", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
