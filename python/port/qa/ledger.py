"""The metrics ledger: one measured value per line in `docs/metrics/`, and the best of them (#409, #620).

Three append-only, tab-separated files, each with a header line:

- `ledger.tsv`: `run_id fixture fixture_hash metric definition value`, one
  line per measured value; an unmeasured metric has no line;
- `runs.tsv`: `run_id timestamp commit arm test note benchmark`, one line
  per run; `benchmark` is `true` where the run is one of a sweep over every
  fixture the ledger held at its commit (`last_benchmark`), else `false`;
- `definitions.tsv`: `metric definition since scorer meaning`, what each
  metric means. Changing what a metric measures appends a definition, never
  edits one, and values under different definitions are not compared.

`run_ledger` (`port.scripts.run_ledger`) records, renders and queries it;
this module is its table API, moved from `tests.metrics` (T- #673 G2). A
run is recorded against `provenance.head`, so the files live in the
checkout `port.extensions.repository.ROOT` names. `.gitattributes` merges them as `union`,
so two branches that each append a run both keep theirs.

A dataset is the pair `fixture`, `fixture_hash`: the name a run was asked
for, and the 8 hex digits of the data it ran on, in their own columns. A name
holds every generation drawn under it, one hash each; a hash holds one name
(`check_identity`), and `best` takes a name alone only while it holds one
hash.
"""

from __future__ import annotations

import datetime
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from port.extensions.repository import ROOT
from port.qa import provenance

LEDGER_DIR = ROOT / "docs" / "metrics"
LEDGER = LEDGER_DIR / "ledger.tsv"
RUNS = LEDGER_DIR / "runs.tsv"
DEFINITIONS = LEDGER_DIR / "definitions.tsv"

LEDGER_COLUMNS = ("run_id", "fixture", "fixture_hash", "metric", "definition", "value")
RUN_COLUMNS = ("run_id", "timestamp", "commit", "arm", "test", "note", "benchmark")
DEFINITION_COLUMNS = ("metric", "definition", "since", "scorer", "meaning")

KEYS = ("commit", "timestamp", "fixture", "fixture_hash", "test", "args", "benchmark")
METRICS = {
    "clone_ari": ("ari", 4),
    "clone_ari_int": ("ari_integer", 4),
    "clone_ari_int_99": ("ari_integer_99", 4),
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
"""Metric -> (`port.qa.audit.Recovery` field, decimals); the seed of
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
"""The audit a `--record` run's metrics come from, as `path::name`: the name
every recovery row has carried since #409. It delegates to `run_audit
--recovery` (T- #673 G3) and stays so the append-only rows resolve."""

SIM_TEST = "tests/sim_audit.py::main"
"""The same, for a `--record --sample` run (#467): `run_audit --sim`."""

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
            "benchmark": run["benchmark"],
        }
        row |= dict.fromkeys(METRICS, UNMEASURED)
        for line in mine:
            if current.get(line["metric"]) == line["definition"]:
                row[line["metric"]] = line["value"]
        rows.append({c: row[c] for c in COLUMNS})
    return rows


def check_identity(fixture: str, digest: str, rows: list[dict[str, str]]) -> None:
    """One hash, one name (#588): refuse `digest` where `rows` hold it under
    another fixture name; raises otherwise.

    A name may hold several hashes, one per generation: the pair is the
    dataset, and a history panel is drawn per pair. A dataset under two names
    would split its history in two.
    """
    if re.fullmatch(r"[0-9a-f]{8}", digest) is None:
        msg = f"a fixture hash is 8 hex digits, got {digest!r}"
        raise ValueError(msg)
    names = {r["fixture"] for r in rows if r["fixture_hash"] == digest} - {fixture}
    if names:
        msg = (
            f"{fixture} hashes to {digest}, but the ledger holds {digest} as "
            f"{sorted(names)}: one dataset has one name"
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
    benchmark: bool = False,
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
        "benchmark": "true" if benchmark else "false",
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
    benchmark: bool = False,
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
        benchmark=benchmark,
    )
    append_tsv(RUNS, RUN_COLUMNS, [run])
    append_tsv(LEDGER, LEDGER_COLUMNS, lines)
    print("\t".join(run[c] for c in RUN_COLUMNS))
    for line in lines:
        print("\t".join(line[c] for c in LEDGER_COLUMNS))
    return run["run_id"]


def resolve(
    fixture: str, digest: str | None, rows: list[dict[str, str]]
) -> tuple[str, str | None]:
    """The dataset `fixture` (and `digest`, if given) names in `rows`: a name
    alone resolves to its one hash, and raises where it holds several."""
    if digest is not None:
        return fixture, digest
    hashes = sorted({r["fixture_hash"] for r in rows if r["fixture"] == fixture})
    if len(hashes) > 1:
        msg = f"{fixture} holds {len(hashes)} datasets, pass one hash: {hashes}"
        raise ValueError(msg)
    return fixture, (hashes[0] if hashes else None)


def best(
    metric: str, fixture: str | None, digest: str | None = None
) -> dict[str, str] | None:
    """The ledger line maximizing `metric` under its latest definition, over
    one dataset's lines if given (`fixture`, and `digest` where the name holds
    several: `resolve`), joined to its run."""
    definition = latest()[metric]
    recorded = ledger()
    dataset = None if fixture is None else resolve(fixture, digest, recorded)
    lines = [
        line
        for line in recorded
        if line["metric"] == metric
        and line["definition"] == definition
        and (dataset is None or (line["fixture"], line["fixture_hash"]) == dataset)
    ]
    found = max(lines, key=lambda line: float(line["value"]), default=None)
    if found is None:
        return None
    run = next(r for r in runs() if r["run_id"] == found["run_id"])
    return run | found


def last_benchmark() -> list[dict[str, str]]:
    """The runs of the latest commit with a `benchmark` run, in `runs` order: the last sweep over every fixture."""
    marked = [r for r in runs() if r["benchmark"] == "true"]
    if not marked:
        return []
    commit = marked[-1]["commit"]
    return [r for r in marked if r["commit"] == commit]


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
