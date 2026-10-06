"""`tests/recovery_audit.py::main`, kept because the ledger's runs name it (#313, T- #673 G3).

The audit is `port.qa.audit` behind `run_audit --recovery`. `docs/metrics/runs.tsv`
is append-only and its `--record` rows name this `main` as their
`test`, which `tests/test_metrics_table.py` resolves; `definitions.tsv` names
`tests.recovery_audit.score` as the scorer of their metrics. Both stay here,
delegating.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from port.qa.audit import score_truth as score

__all__ = ["main", "score"]


def main(argv: Sequence[str] | None = None) -> int:
    """`run_audit --recovery` with these arguments."""
    from port.scripts.run_audit import main as run_audit

    return run_audit(["--recovery", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
