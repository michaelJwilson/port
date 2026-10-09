"""`tests/recovery_audit.py::main` and `score`, kept because the metrics ledger names
them (#313, T- #673 G3).

Both delegate to `port.qa.audit` (`run_audit --recovery`).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from port.qa.audit import score_truth as score

__all__ = ["main", "score"]


def main(argv: Sequence[str] | None = None) -> int:
    """`run_audit --recovery` with these arguments."""
    from port.qa.scripts.run_audit import main as run_audit

    return run_audit(["--recovery", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
