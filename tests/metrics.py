"""One measured value per line in `docs/metrics/`, and the best of them (#409, #620).

Three append-only, tab-separated files, each with a header line:

- `ledger.tsv`: `run_id fixture fixture_hash metric definition value`, one
  line per measured value; an unmeasured metric has no line;
- `runs.tsv`: `run_id timestamp commit arm test note`, one line per run;
- `definitions.tsv`: `metric definition since scorer meaning`, what each
  metric means. Changing what a metric measures appends a definition, never
  edits one, and values under different definitions are not compared.

`python -m tests.metrics --record --note "..." [--instance dev] [--lattice] -- [flags]`
runs `tests.recovery_audit` in its own process, so `peak_gb` is that run's
(`--sample easy` runs `tests.sim_audit` instead), appends the `runs` line and
one `ledger` line per measured metric under its latest definition, and
re-renders `docs/metrics.md`. The note is at most `NOTE_CHARS` characters,
written as a commit subject, stating what change the run measures.

`python -m tests.metrics --best clone_ari [--fixture dev]` prints the ledger
line, joined to its run, that maximizes a metric under its latest definition.
`--render` writes `docs/metrics.md`, a generated view: one table per fixture,
metrics as columns.

A run is recorded against a commit, so the inputs must be committed first
(`--dirty` records anyway and marks the commit `+`). A new metric appends a
definition and adds lines, not columns. `.gitattributes` merges the files as
`union`, so two branches that each append a run both keep theirs.
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
from collections.abc import Iterable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "docs" / "metrics.md"
"""The generated view; `--render` writes it, nothing else does."""

LEDGER_DIR = ROOT / "docs" / "metrics"
LEDGER = LEDGER_DIR / "ledger.tsv"
RUNS = LEDGER_DIR / "runs.tsv"
DEFINITIONS = LEDGER_DIR / "definitions.tsv"

LEDGER_COLUMNS = ("run_id", "fixture", "fixture_hash", "metric", "definition", "value")
RUN_COLUMNS = ("run_id", "timestamp", "commit", "arm", "test", "note")
DEFINITION_COLUMNS = ("metric", "definition", "since", "scorer", "meaning")

KEYS = ("commit", "timestamp", "fixture", "fixture_hash", "test", "args")
METRICS = {
    "clone_ari": ("ari", 4),
    "clone_ari_int": ("ari_integer", 4),
    "copy_ari": ("copy_ari", 4),
    "copy_ari_loh": ("copy_ari_loh", 4),
    "copy_ari_bgain": ("copy_ari_balanced_gain", 4),
    "copy_ari_ugain": ("copy_ari_unbalanced_gain", 4),
    "copy_ari_pf": ("copy_ari_pf", 4),
    "copy_ari_loh_pf": ("copy_ari_loh_pf", 4),
    "copy_ari_bgain_pf": ("copy_ari_balanced_gain_pf", 4),
    "copy_ari_ugain_pf": ("copy_ari_unbalanced_gain_pf", 4),
    "state_ari": ("state_ari", 4),
    "exact_altered": ("exact_altered", 4),
    "exact_altered_pf": ("exact_altered_minor", 4),
    "exact_loh": ("exact_loh", 4),
    "exact_loh_pf": ("exact_loh_pf", 4),
    "exact_bgain": ("exact_balanced_gain", 4),
    "exact_bgain_pf": ("exact_balanced_gain_pf", 4),
    "exact_ugain": ("exact_unbalanced_gain", 4),
    "exact_ugain_pf": ("exact_unbalanced_gain_pf", 4),
    "exact_neutral": ("exact_neutral", 4),
    "wall_s": ("wall", 1),
    "peak_gb": ("peak_gb", 2),
}
"""Metric -> (`tests.recovery_audit.Recovery` field, decimals); the seed of
`definitions.tsv`, each as definition 1."""

COLUMNS = (*KEYS, *METRICS, "note")
"""The keys of a `read()` row: a run with its metrics, one per key."""

UNMEASURED = "—"
"""What `read()` and `--best` give for a metric with no ledger line."""

CONVERTED = " [converted: "
"""Where a converted run's note (#620) ends and the conversion's remark on it
begins: the note as recorded, then what `metrics.md`'s preamble said of it."""

TEST = "tests/recovery_audit.py::main"
"""The function computing a `--record` run's metrics, as `path::name`."""

SIM_TEST = "tests/sim_audit.py::main"
"""The same, for a `--record --sample` run (#467)."""

NOTE_CHARS = 72
"""A commit subject's limit, so a note reads as one."""

TIMESTAMP = "%Y-%m-%dT%H:%MZ"

INPUTS = ("python", "src", "tests", "pyproject.toml", "uv.lock", "Cargo.lock")
"""What a run's commit must hold for the run to be that commit's."""


def fixture_hash(truth: Any) -> str:
    """A digest of the data a fixture built, not of the code that built it.

    Every field of the `CoreInferenceTruth`, by dtype, shape and bytes: a run
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


def read_tsv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """`path`'s lines after its header, keyed by `columns`; raises on a header
    or a line that is not exactly `columns`."""
    lines = path.read_text().splitlines()
    if not lines or tuple(lines[0].split("\t")) != tuple(columns):
        msg = f"{path.name} header is {lines[:1]}, expected {list(columns)}"
        raise ValueError(msg)
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(columns):
            msg = f"{path.name}:{number} has {len(cells)} cells, not {len(columns)}"
            raise ValueError(msg)
        rows.append(dict(zip(columns, cells, strict=True)))
    return rows


def append_tsv(
    path: Path, columns: Sequence[str], rows: Iterable[dict[str, str]]
) -> None:
    """Append `rows` to `path` in `columns` order; refuses a cell holding a
    tab or a newline, which would split it."""
    lines = []
    for row in rows:
        cells = [row[c] for c in columns]
        if any("\t" in c or "\n" in c for c in cells):
            msg = f"a tab or newline would split the line: {cells}"
            raise ValueError(msg)
        lines.append("\t".join(cells) + "\n")
    with path.open("a") as out:
        out.writelines(lines)


def runs() -> list[dict[str, str]]:
    return read_tsv(RUNS, RUN_COLUMNS)


def ledger() -> list[dict[str, str]]:
    return read_tsv(LEDGER, LEDGER_COLUMNS)


def definitions() -> list[dict[str, str]]:
    return read_tsv(DEFINITIONS, DEFINITION_COLUMNS)


def latest() -> dict[str, str]:
    """Each metric's latest definition, by number."""
    found: dict[str, str] = {}
    for d in definitions():
        if d["metric"] in METRICS:
            current = found.get(d["metric"], "0")
            found[d["metric"]] = max(current, d["definition"], key=int)
    return found


def read() -> list[dict[str, str]]:
    """One row per run that measured anything, keyed by `COLUMNS`, in `runs`
    order: each metric's value under its latest definition, `UNMEASURED`
    where it has no line."""
    current = latest()
    lines: dict[str, list[dict[str, str]]] = {}
    for line in ledger():
        lines.setdefault(line["run_id"], []).append(line)
    rows = []
    for run in runs():
        mine = lines.get(run["run_id"], [])
        if not mine:
            continue
        row = {
            "commit": run["commit"],
            "timestamp": run["timestamp"],
            "fixture": mine[0]["fixture"],
            "fixture_hash": mine[0]["fixture_hash"],
            "test": run["test"],
            "args": run["arm"],
            "note": run["note"],
        }
        row |= dict.fromkeys(METRICS, UNMEASURED)
        for line in mine:
            if current.get(line["metric"]) == line["definition"]:
                row[line["metric"]] = line["value"]
        rows.append({c: row[c] for c in COLUMNS})
    return rows


def check_identity(fixture: str, digest: str, rows: list[dict[str, str]]) -> None:
    """One name, one dataset (#588): refuse a fixture holding another
    `fixture_hash` in `rows`, or a hash another fixture; raises otherwise.

    A new generation under an old name would otherwise share its history
    panel with data it was never measured on.
    """
    hashes = {r["fixture_hash"] for r in rows if r["fixture"] == fixture} - {digest}
    names = {r["fixture"] for r in rows if r["fixture_hash"] == digest} - {fixture}
    if hashes or names:
        msg = (
            f"{fixture} hashes to {digest}, but the ledger also holds {fixture} "
            f"as {sorted(hashes)} and {digest} as {sorted(names)}: one fixture "
            "name names one dataset"
        )
        raise ValueError(msg)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def check_note(note: str) -> None:
    """One line, not blank, within `NOTE_CHARS`, no `|` or tab; raises otherwise."""
    bad = "\n" in note or "|" in note or "\t" in note
    if not note.strip() or bad or len(note) > NOTE_CHARS:
        msg = (
            f"the note must be one line of 1 to {NOTE_CHARS} characters "
            f"without `|` or a tab, got {len(note)}: {note!r}"
        )
        raise ValueError(msg)


def run_id(commit: str, fixture: str, timestamp: str, taken: set[str]) -> str:
    """`<commit>-<fixture>-<HHMM>`, with `-2`, `-3`, ... where `taken` holds it."""
    base = f"{commit}-{fixture}-{timestamp[11:13]}{timestamp[14:16]}"
    found, k = base, 1
    while found in taken:
        k += 1
        found = f"{base}-{k}"
    return found


def entries(
    recovery: dict[str, Any],
    *,
    fixture: str,
    args: str,
    note: str,
    commit: str,
    timestamp: str,
    taken: set[str],
    current: dict[str, str],
    test: str = TEST,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """A run's `runs` line and its `ledger` lines, one per measured metric
    under `current`'s definition of it; NaN and None have no line."""
    check_note(note)
    identifier = run_id(commit, fixture, timestamp, taken)
    run = {
        "run_id": identifier,
        "timestamp": timestamp,
        "commit": commit,
        "arm": args,
        "test": test,
        "note": note,
    }
    lines = []
    for metric, (key, decimals) in METRICS.items():
        value = recovery.get(key)
        # NB NaN is a class the sample does not plant, unmeasured as None is
        if value is None or math.isnan(value):
            continue
        lines.append(
            {
                "run_id": identifier,
                "fixture": fixture,
                "fixture_hash": recovery["fixture_hash"],
                "metric": metric,
                "definition": current[metric],
                "value": f"{value:.{decimals}f}",
            }
        )
    return run, lines


def write(
    recovery: dict[str, Any],
    *,
    fixture: str,
    args: str,
    note: str,
    dirty: bool,
    test: str = TEST,
) -> str:
    """Append a run to `runs` and `ledger`, after `check_identity`, and
    re-render the view; returns the run's id."""
    recorded = ledger()
    check_identity(fixture, recovery["fixture_hash"], recorded)
    run, lines = entries(
        recovery,
        fixture=fixture,
        args=args,
        note=note,
        commit=_git("rev-parse", "--short=7", "HEAD") + ("+" if dirty else ""),
        timestamp=datetime.datetime.now(datetime.UTC).strftime(TIMESTAMP),
        taken={r["run_id"] for r in runs()},
        current=latest(),
        test=test,
    )
    append_tsv(RUNS, RUN_COLUMNS, [run])
    append_tsv(LEDGER, LEDGER_COLUMNS, lines)
    TABLE.write_text(render())
    print("\t".join(run[c] for c in RUN_COLUMNS))
    for line in lines:
        print("\t".join(line[c] for c in LEDGER_COLUMNS))
    return run["run_id"]


def record(arguments: argparse.Namespace) -> int:
    if arguments.note is None:
        print("--record needs --note: what change this run measures")
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
    write(
        recovery,
        fixture=fixture,
        args=shlex.join([*audit, "--", *flags]),
        note=arguments.note,
        dirty=dirty,
    )
    return 0


def record_sample(arguments: argparse.Namespace, *, dirty: bool) -> int:
    """A run on a simulated sample: `tests.sim_audit` in its own process (#467).

    `r0` is `dev_tree`'s realization 0, drawn if absent and refused unless it
    is the one `tests.sim_stages` names; `easy` and `hard` are CalicoST's.
    The fixture hash is the sample's content hash (`realization_hash`), and
    a name the ledger already holds under another hash is refused before the
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

    # NB checked before the run as well as at the write, so a refused name
    #    costs no run
    digest = realization_hash(path)
    check_identity(arguments.sample, digest, ledger())

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
    write(
        recovery,
        fixture=arguments.sample,
        args=shlex.join([*audit, "--", *flags]),
        note=arguments.note,
        dirty=dirty,
        test=SIM_TEST,
    )
    return 0


def best(metric: str, fixture: str | None) -> dict[str, str] | None:
    """The ledger line maximizing `metric` under its latest definition, over
    `fixture`'s lines if given, joined to its run."""
    definition = latest()[metric]
    lines = [
        line
        for line in ledger()
        if line["metric"] == metric
        and line["definition"] == definition
        and (fixture is None or line["fixture"] == fixture)
    ]
    found = max(lines, key=lambda line: float(line["value"]), default=None)
    if found is None:
        return None
    run = next(r for r in runs() if r["run_id"] == found["run_id"])
    return run | found


VIEW = """\
# Metrics

Generated by `python -m tests.metrics --render` from
[`docs/metrics/`](metrics/) (#620); do not edit by hand, as the next
`--record` or `--render` overwrites it. `ledger.tsv` holds one measured value
per line, `runs.tsv` one line per run, `definitions.tsv` what each metric
means, versioned. One table per fixture, one row per run, one column per
metric and definition (`metric` for definition 1, `metric.N` for definition
N); a blank cell has no ledger line. A note ending `[converted: ...]` carries
what the hand-edited table's preamble said of that run.
"""


def _header(metric: str, definition: str) -> str:
    return metric if definition == "1" else f"{metric}.{definition}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render() -> str:
    """`docs/metrics.md` from `docs/metrics/`: one table per fixture in order
    of first appearance, runs in `runs` order, then the definitions."""
    order = {(d["metric"], d["definition"]): k for k, d in enumerate(definitions())}
    values: dict[str, dict[tuple[str, str], str]] = {}
    fixtures: dict[str, dict[str, str]] = {}
    for line in ledger():
        fixtures.setdefault(line["fixture"], {})[line["run_id"]] = line["fixture_hash"]
        key = (line["metric"], line["definition"])
        values.setdefault(line["run_id"], {})[key] = line["value"]

    out = [VIEW]
    every = runs()
    for fixture, held in fixtures.items():
        mine = [r for r in every if r["run_id"] in held]
        keys = sorted(
            {k for r in mine for k in values[r["run_id"]]}, key=lambda k: order[k]
        )
        hashes = ", ".join(f"`{h}`" for h in dict.fromkeys(held.values()))
        out += [f"## `{fixture}` ({hashes})", ""]
        out += _table(
            [
                "run_id",
                "timestamp",
                "test",
                "arm",
                *(_header(*k) for k in keys),
                "note",
            ],
            (
                [
                    r["run_id"],
                    r["timestamp"],
                    r["test"],
                    r["arm"],
                    *(values[r["run_id"]].get(k, "") for k in keys),
                    r["note"],
                ]
                for r in mine
            ),
        )
        out.append("")
    out += ["## Definitions", ""]
    out += _table(
        DEFINITION_COLUMNS,
        ([d[c] for c in DEFINITION_COLUMNS] for d in definitions()),
    )
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.metrics", description=__doc__
    )
    parser.add_argument(
        "--record", action="store_true", help="run, append to the ledger, re-render"
    )
    parser.add_argument("--best", choices=list(METRICS), help="the line maximizing it")
    parser.add_argument("--fixture", default=None, help="with --best, one fixture")
    parser.add_argument(
        "--render", action="store_true", help="write docs/metrics.md from the ledger"
    )
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
    if arguments.render:
        TABLE.write_text(render())
        print(TABLE)
        return 0
    if arguments.best:
        found = best(arguments.best, arguments.fixture)
        print(UNMEASURED if found is None else json.dumps(found))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
