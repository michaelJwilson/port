"""The metrics ledger: one measured value per line in `docs/metrics/`, and the best of them (#409, #620).

Three append-only, tab-separated files, each with a header line:

- `ledger.tsv`: `run_id fixture fixture_hash metric definition value`, one
  line per measured value; an unmeasured metric has no line;
- `runs.tsv`: `run_id timestamp commit arm test note`, one line per run;
- `definitions.tsv`: `metric definition since scorer meaning`, what each
  metric means. Changing what a metric measures appends a definition, never
  edits one, and values under different definitions are not compared.

`run_ledger` (`port.scripts.run_ledger`) records, renders and queries it;
this module is its table API, moved from `tests.metrics` (T- #673 G2). A
run is recorded against `provenance.head`, so the files live in the
checkout `provenance.ROOT` names. `.gitattributes` merges them as `union`,
so two branches that each append a run both keep theirs.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from port.qa import provenance

ROOT = provenance.ROOT
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
begins: the note as recorded, then what the hand-edited `docs/metrics.md`'s
preamble said of it."""

TEST = "tests/recovery_audit.py::main"
"""The function computing a `--record` run's metrics, as `path::name`."""

SIM_TEST = "tests/sim_audit.py::main"
"""The same, for a `--record --sample` run (#467)."""

NOTE_CHARS = 72
"""A commit subject's limit, so a note reads as one."""

TIMESTAMP = "%Y-%m-%dT%H:%MZ"


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
    """Append a run to `runs` and `ledger`, after `check_identity`; returns
    the run's id."""
    recorded = ledger()
    check_identity(fixture, recovery["fixture_hash"], recorded)
    run, lines = entries(
        recovery,
        fixture=fixture,
        args=args,
        note=note,
        commit=provenance.head() + ("+" if dirty else ""),
        timestamp=datetime.datetime.now(datetime.UTC).strftime(TIMESTAMP),
        taken={r["run_id"] for r in runs()},
        current=latest(),
        test=test,
    )
    append_tsv(RUNS, RUN_COLUMNS, [run])
    append_tsv(LEDGER, LEDGER_COLUMNS, lines)
    print("\t".join(run[c] for c in RUN_COLUMNS))
    for line in lines:
        print("\t".join(line[c] for c in LEDGER_COLUMNS))
    return run["run_id"]


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

Generated by `run_ledger --render` from `docs/metrics/` (#620):
one row per run, one column per metric under its latest definition, `—`
where the run has no ledger line. A note ending `[converted: ...]` carries
what the hand-edited table's preamble said of that run.
"""


def render() -> str:
    """The wide view of `read()`: `VIEW`, then one Markdown table, `COLUMNS`
    as its header, one row per run in `runs` order."""
    rows = read()
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
        *("| " + " | ".join(r[c] for c in COLUMNS) + " |" for r in rows),
    ]
    return VIEW + "\n" + "\n".join(lines) + "\n"


def parse(text: str) -> list[dict[str, str]]:
    """`render()`'s rows back, each keyed by `COLUMNS`; raises on another header."""
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if tuple(header) != COLUMNS:
        msg = f"the header is {header}, expected {list(COLUMNS)}"
        raise ValueError(msg)
    return [
        dict(zip(COLUMNS, (c.strip() for c in line.strip("|").split("|")), strict=True))
        for line in lines[2:]
    ]
