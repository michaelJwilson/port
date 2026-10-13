"""cnamaste's ledger: one append-only row per (capture commit, fixture hash), scored from the staged file (Ticket#836).

    uv run --locked --project cnamaste python -m cnamaste.ledger \
        cnamaste/tests/data/sim_<hash>.hdf5 [...] [--replace] [--date YYYY-MM-DD]

Each staged file is scored against the planted truth it stores (`/truth`) by
`audit.scoring.ledger_scores` and appended to `docs/ledger/ledger.tsv` as one row. A row
for the same capture commit and fixture hash is refused unless `--replace`,
which rewrites that row in place. `docs/ledger/README.md` states the columns.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from cnamaste.audit import PROJECT
from cnamaste.audit.capture import Capture
from cnamaste.audit.scoring import ledger_scores, planted
from cnamaste.audit.sims import NAMES, SUPPORTED_SIMS

LEDGER_DIR = PROJECT / "docs" / "ledger"

LEDGER = LEDGER_DIR / "ledger.tsv"

COLUMNS = (
    "date",
    "commit",
    "fixture",
    "fixture_hash",
    "spots",
    "bins",
    "clone_ari",
    "clone_ari_int_099",
    "clone_ari_int_100",
    "copy_ari_pf",
    "exact_altered_pf",
    "run_seconds",
)
"""The ledger's header, in order; `README.md` defines each."""

DECIMALS = {
    "clone_ari": 4,
    "clone_ari_int_099": 4,
    "clone_ari_int_100": 4,
    "copy_ari_pf": 4,
    "exact_altered_pf": 4,
    "run_seconds": 1,
}
"""A float column's printed decimals."""


def scored(path: Path) -> dict[str, str]:
    """The staged file at `path` as a ledger row, but its `date`."""
    sim = Capture(path)
    try:
        sim_hash = str(sim.config["fixture_hash"])
        stored = sim.truth()
        # NB a staged file captured before `/truth` (easy's) is scored against its committed sample's truth
        truth = planted(stored) if stored else planted(PROJECT.parent / SUPPORTED_SIMS[sim_hash][0])
        values = ledger_scores(sim, truth)
        commit = str(sim.config["capture_commit"])
    finally:
        sim.close()
    row = {"commit": commit, "fixture": NAMES[sim_hash], "fixture_hash": sim_hash}
    for key, value in values.items():
        row[key] = "NA" if value != value else f"{value:.{DECIMALS[key]}f}" if key in DECIMALS else str(value)
    return row


def read(path: Path = LEDGER) -> list[dict[str, str]]:
    """The ledger's rows, keyed by `COLUMNS`; refused on another header or a short line."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if not lines or tuple(lines[0].split("\t")) != COLUMNS:
        raise ValueError(f"{path.name}: the header is {lines[:1]}, expected {list(COLUMNS)}")
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(COLUMNS):
            raise ValueError(f"{path.name}:{number} has {len(cells)} cells, not {len(COLUMNS)}")
        rows.append(dict(zip(COLUMNS, cells, strict=True)))
    return rows


def latest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Each fixture hash's last row in file order."""
    return {row["fixture_hash"]: row for row in rows}


def write(rows: list[dict[str, str]], path: Path = LEDGER) -> None:
    path.write_text("\n".join(["\t".join(COLUMNS), *("\t".join(r[c] for c in COLUMNS) for r in rows)]) + "\n")


def record(row: dict[str, str], rows: list[dict[str, str]], *, replace: bool) -> list[dict[str, str]]:
    """`rows` with `row` appended; where a row has its commit and hash, refused, or with `replace` rewritten in place."""
    key = (row["commit"], row["fixture_hash"])
    at = [i for i, r in enumerate(rows) if (r["commit"], r["fixture_hash"]) == key]
    if at and not replace:
        raise SystemExit(f"{LEDGER.name} has a row for commit {key[0]}, fixture {key[1]}: --replace to rewrite it")
    if at:
        return [row if i == at[0] else r for i, r in enumerate(rows)]
    return [*rows, row]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("staged", type=Path, nargs="+", help="staged files, sim_<hash>.hdf5")
    parser.add_argument("--replace", action="store_true", help="rewrite a row of the same commit and fixture hash")
    parser.add_argument("--date", default=datetime.datetime.now(datetime.UTC).date().isoformat())
    arguments = parser.parse_args(argv)
    rows = read()
    for path in arguments.staged:
        row = {"date": arguments.date, **scored(path)}
        rows = record(row, rows, replace=arguments.replace)
        print("\t".join(row[c] for c in COLUMNS))
    write(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
