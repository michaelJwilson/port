"""One row per measured run in `docs/metrics.md`, and the best of them (#409).

`python -m tests.metrics --record --note "..." [--instance dev] [--lattice] -- [flags]`
runs `tests.recovery_audit` in its own process, so `peak_gb` is that run's,
and appends one row: the commit, the UTC timestamp, the fixture and a digest
of the data it built, the test that computed the metrics, the arguments that
reproduce it, the tracked metrics, and a note of at most `NOTE_CHARS`
characters, written as a commit subject, stating what change the row
measures.

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
import math
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

KEYS = ("commit", "timestamp", "fixture", "fixture_hash", "test", "args")
METRICS = {
    "clone_ari": ("ari", 4),
    "clone_ari_int": ("ari_integer", 4),
    "copy_ari": ("copy_ari", 4),
    "state_ari": ("state_ari", 4),
    "exact_altered": ("exact_altered", 4),
    "exact_altered_pf": ("exact_altered_minor", 4),
    "exact_loh": ("exact_loh", 4),
    "exact_loh_pf": ("exact_loh_pf", 4),
    "exact_bgain": ("exact_balanced_gain", 4),
    "exact_bgain_pf": ("exact_balanced_gain_pf", 4),
    "exact_ugain": ("exact_unbalanced_gain", 4),
    "exact_ugain_pf": ("exact_unbalanced_gain_pf", 4),
    "wall_s": ("wall", 1),
    "peak_gb": ("peak_gb", 2),
}
"""Column -> (`tests.recovery_audit.Recovery` field, decimals)."""

COLUMNS = (*KEYS, *METRICS, "note")
UNMEASURED = "—"

TEST = "tests/recovery_audit.py::main"
"""The function computing a `--record` row's metrics, as `path::name`."""

SIM_TEST = "tests/sim_audit.py::main"
"""The same, for a `--record --sample` row (#467)."""

NOTE_CHARS = 72
"""A commit subject's limit, so a note reads as one."""

TIMESTAMP = "%Y-%m-%dT%H:%MZ"

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


def check_identity(fixture: str, digest: str, rows: list[dict[str, str]]) -> None:
    """One name, one dataset (#588): refuse a row whose fixture holds another
    `fixture_hash` in `rows`, or whose hash another fixture; raises otherwise.

    A new generation under an old name would otherwise share its history
    panel with data it was never measured on.
    """
    hashes = {r["fixture_hash"] for r in rows if r["fixture"] == fixture} - {digest}
    names = {r["fixture"] for r in rows if r["fixture_hash"] == digest} - {fixture}
    if hashes or names:
        msg = (
            f"{fixture} hashes to {digest}, but the table also holds {fixture} "
            f"as {sorted(hashes)} and {digest} as {sorted(names)}: one fixture "
            "name names one dataset"
        )
        raise ValueError(msg)


def append(line: str, *, fixture: str, digest: str) -> None:
    """Append `line` to `TABLE`, after `check_identity` against its rows."""
    check_identity(fixture, digest, read())
    with TABLE.open("a") as table:
        table.write(line + "\n")
    print(line)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def check_note(note: str) -> None:
    """One line, not blank, within `NOTE_CHARS`, no `|`; raises otherwise."""
    if not note.strip() or "\n" in note or "|" in note or len(note) > NOTE_CHARS:
        msg = (
            f"the note must be one line of 1 to {NOTE_CHARS} characters "
            f"without `|`, got {len(note)}: {note!r}"
        )
        raise ValueError(msg)


def row(
    recovery: dict[str, Any],
    *,
    fixture: str,
    args: str,
    note: str,
    dirty: bool,
    test: str = TEST,
) -> str:
    if "|" in args:
        msg = f"a `|` in the arguments would split the row: {args}"
        raise ValueError(msg)
    check_note(note)
    cells = {
        "commit": _git("rev-parse", "--short=7", "HEAD") + ("+" if dirty else ""),
        "timestamp": datetime.datetime.now(datetime.UTC).strftime(TIMESTAMP),
        "fixture": fixture,
        "fixture_hash": recovery["fixture_hash"],
        "test": test,
        "args": args,
        "note": note,
    }
    for column, (key, decimals) in METRICS.items():
        value = recovery.get(key)
        # NB NaN is a class the sample does not plant, unmeasured as None is
        missing = value is None or math.isnan(value)
        cells[column] = UNMEASURED if missing else f"{value:.{decimals}f}"
    return "| " + " | ".join(cells[c] for c in COLUMNS) + " |"


def record(arguments: argparse.Namespace) -> int:
    if arguments.note is None:
        print("--record needs --note: what change this row measures")
        return 1
    check_note(arguments.note)
    dirty = bool(_git("status", "--porcelain", "--", *INPUTS))
    if dirty and not arguments.dirty:
        print("inputs are uncommitted; commit them, or pass --dirty")
        return 1

    if arguments.sample is not None:
        return record_sample(arguments, dirty=dirty)

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
        note=arguments.note,
        dirty=dirty,
    )
    append(line, fixture=fixture, digest=recovery["fixture_hash"])
    return 0


def record_sample(arguments: argparse.Namespace, *, dirty: bool) -> int:
    """A row for a simulated sample: `tests.sim_audit` in its own process (#467).

    `r0` is `dev_tree`'s realization 0, drawn if absent and refused unless it
    is the one `tests.sim_stages` names; `easy` and `hard` are CalicoST's.
    The fixture hash is the sample's content hash (`realization_hash`), and
    a name the table already holds under another hash is refused before the
    run (`check_identity`).
    """
    from tests.sim_stages import r0, realization_hash

    if arguments.sample == "r0":
        path = r0()
        sample = "generated/dev_tree/r0"
    else:
        from tests.sim_audit import SAMPLES
        from tests.sim_fixtures import SIM_ROOT

        sample = SAMPLES.get(arguments.sample, arguments.sample)
        path = SIM_ROOT / sample

    # NB checked before the run as well as at the append, so a refused name
    #    costs no run
    digest = realization_hash(path)
    check_identity(arguments.sample, digest, read())

    audit = [
        *(item for entry in arguments.set for item in ("--set", entry)),
        *(["--oracle-start"] if arguments.oracle_start else []),
    ]
    flags = [f for f in arguments.flags if f != "--"]
    command = [
        sys.executable, "-m", "tests.sim_audit", "--sample", sample,
        *audit, "--", *flags,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("SIM ")]
    if completed.returncode or not lines:
        print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
        return completed.returncode or 1

    recovery = json.loads(lines[-1].removeprefix("SIM "))
    recovery["fixture_hash"] = digest
    line = row(
        recovery,
        fixture=arguments.sample,
        args=shlex.join([*audit, "--", *flags]),
        note=arguments.note,
        dirty=dirty,
        test=SIM_TEST,
    )
    append(line, fixture=arguments.sample, digest=recovery["fixture_hash"])
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
        "--note", default=None, help=f"with --record, <= {NOTE_CHARS} characters"
    )
    parser.add_argument(
        "--instance", default="dev", choices=["calicost", "critical", "dev"]
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="with --record, a simulated sample: r0, easy, hard or a sim/ path",
    )
    parser.add_argument(
        "--oracle-start", action="store_true", help="with --sample, the planted clones"
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
