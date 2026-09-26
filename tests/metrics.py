"""One row per measured run in `docs/metrics.md`, and the best of them (#409).

`python -m tests.metrics --record [--instance dev] [--lattice] -- [flags]`
runs `tests.recovery_audit` in its own process, so `peak_gb` is that run's,
and appends one row: the commit, the fixture and a digest of the data it
built, the arguments that reproduce it, then the tracked metrics.

`python -m tests.metrics --best clone_ari [--fixture dev]` prints the row that
maximizes a metric.

A row is recorded against a commit, so the inputs must be committed first
(`--dirty` records anyway and marks the commit `+`). A metric added later is
a new column, and older rows read `—` until back-filled (#409, deferred).
`.gitattributes` merges the file as `union`, so two branches that each append
a row both keep theirs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "docs" / "metrics.md"

KEYS = ("commit", "date", "fixture", "fixture_hash", "args")
METRICS = {
    "clone_ari": ("ari", 4),
    "clone_ari_int": ("ari_integer", 4),
    "copy_ari": ("copy_ari", 4),
    "state_ari": ("state_ari", 4),
    "wall_s": ("wall", 1),
    "peak_gb": ("peak_gb", 2),
}
"""Column -> (`tests.recovery_audit.Recovery` field, decimals)."""

COLUMNS = (*KEYS, *METRICS)
UNMEASURED = "—"

INPUTS = ("python", "src", "tests", "pyproject.toml", "uv.lock", "Cargo.lock")
"""What a row's commit must hold for the row to be that commit's."""


def fixture_hash(truth: Any) -> str:
    """A digest of the data a fixture built, not of the code that built it.

    Every field of the `CoreInferenceTruth`, by dtype, shape and bytes: a row
    reproduces only where the same arrays still come out, whatever changed in
    between.
    """
    digest = hashlib.sha256()
    for field in fields(truth):
        value = getattr(truth, field.name)
        digest.update(field.name.encode())
        if isinstance(value, np.ndarray):
            digest.update(f"{value.dtype.str}{value.shape}".encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:8]


def fixture_name(instance: str, *, lattice: bool, loh: bool) -> str:
    return instance + ("-lattice" if lattice else "") + ("-loh" if loh else "")


def read(path: Path = TABLE) -> list[dict[str, str]]:
    """The table's rows, each keyed by `COLUMNS`."""
    lines = [line for line in path.read_text().splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if tuple(header) != COLUMNS:
        msg = f"{path.name} header is {header}, expected {list(COLUMNS)}"
        raise ValueError(msg)
    return [
        dict(zip(COLUMNS, (c.strip() for c in line.strip("|").split("|")), strict=True))
        for line in lines[1:]
        if not line.startswith("| ---")
    ]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def row(recovery: dict[str, Any], *, fixture: str, args: str, dirty: bool) -> str:
    if "|" in args:
        msg = f"a `|` in the arguments would split the row: {args}"
        raise ValueError(msg)
    cells = {
        "commit": _git("rev-parse", "--short=7", "HEAD") + ("+" if dirty else ""),
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "fixture": fixture,
        "fixture_hash": recovery["fixture_hash"],
        "args": args,
    }
    for column, (key, decimals) in METRICS.items():
        value = recovery.get(key)
        cells[column] = UNMEASURED if value is None else f"{value:.{decimals}f}"
    return "| " + " | ".join(cells[c] for c in COLUMNS) + " |"


def record(arguments: argparse.Namespace) -> int:
    dirty = bool(_git("status", "--porcelain", "--", *INPUTS))
    if dirty and not arguments.dirty:
        print("inputs are uncommitted; commit them, or pass --dirty")
        return 1

    fixture = fixture_name(
        arguments.instance, lattice=arguments.lattice, loh=arguments.loh
    )
    audit = [
        "--states", str(arguments.states),
        "--outer", str(arguments.outer),
        "--iterations", str(arguments.iterations),
        *(item for entry in arguments.set for item in ("--set", entry)),
    ]  # fmt: skip
    flags = [f for f in arguments.flags if f != "--"]
    command = [
        sys.executable, "-m", "tests.recovery_audit",
        "--instance", arguments.instance,
        *(["--lattice"] if arguments.lattice else []),
        *(["--loh"] if arguments.loh else []),
        *audit,
        "--", *flags,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith("RECOVERY ")
    ]
    if completed.returncode or not lines:
        print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
        return completed.returncode or 1

    recovery = json.loads(lines[-1].removeprefix("RECOVERY "))
    line = row(
        recovery,
        fixture=fixture,
        args=shlex.join([*audit, "--", *flags]),
        dirty=dirty,
    )
    with TABLE.open("a") as table:
        table.write(line + "\n")
    print(line)
    return 0


def best(metric: str, fixture: str | None) -> dict[str, str] | None:
    """The row maximizing `metric`, over `fixture`'s rows if given."""
    rows = [
        r
        for r in read()
        if r[metric] != UNMEASURED and (fixture is None or r["fixture"] == fixture)
    ]
    return max(rows, key=lambda r: float(r[metric]), default=None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.metrics", description=__doc__
    )
    parser.add_argument("--record", action="store_true", help="run and append a row")
    parser.add_argument("--best", choices=list(METRICS), help="the row maximizing it")
    parser.add_argument("--fixture", default=None, help="with --best, one fixture")
    parser.add_argument("--dirty", action="store_true", help="record uncommitted")
    parser.add_argument(
        "--instance", default="dev", choices=["calicost", "critical", "dev"]
    )
    parser.add_argument("--lattice", action="store_true")
    parser.add_argument("--loh", action="store_true")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)

    if arguments.record:
        return record(arguments)
    if arguments.best:
        found = best(arguments.best, arguments.fixture)
        print(UNMEASURED if found is None else json.dumps(found))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
