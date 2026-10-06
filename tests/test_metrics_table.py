"""The metrics ledger in `docs/metrics/` is consistent, and its latest `dev`
run is the fixture built now (#409, #620).

The ledger guards are what make a value comparable: a line whose run or
definition is missing is a number with nothing to say what it measured. The
hash check is what makes a run reproducible rather than a number with a
commit beside it: a change that moves what `dev_instance` builds leaves every
earlier run describing data that no longer comes out, and fails here until a
run on the new data is recorded.
"""

import ast
import datetime
import hashlib
import re
from pathlib import Path
from typing import Any

import pytest
from port.qa import ledger as metrics
from port.qa.ledger import (
    COLUMNS,
    CONVERTED,
    METRICS,
    NOTE_CHARS,
    ROOT,
    TEST,
    TIMESTAMP,
    check_identity,
    check_note,
    definitions,
    ledger,
    parse,
    read,
    render,
    runs,
)

from tests import fixtures
from tests.metrics import fixture_hash


@pytest.mark.infra
def test_every_run_parses_and_is_in_timestamp_order() -> None:
    every = runs()
    assert every, "docs/metrics/runs.tsv has no runs"

    ids = [run["run_id"] for run in every]
    assert len(ids) == len(set(ids)), "a run_id appears twice"
    for run in every:
        assert re.fullmatch(r"[0-9a-f]{7}\+?", run["commit"]), run
        assert run["run_id"].startswith(f"{run['commit']}-"), run
        datetime.datetime.strptime(run["timestamp"], TIMESTAMP).replace(
            tzinfo=datetime.UTC
        )
        check_note(run["note"].split(CONVERTED)[0])

    stamps = [run["timestamp"] for run in every]
    assert stamps == sorted(stamps)


@pytest.mark.infra
def test_every_ledger_line_names_a_run_and_a_definition() -> None:
    """Each line's `run_id` is in `runs`, its `(metric, definition)` in
    `definitions`, and its value a number."""
    ids = {run["run_id"] for run in runs()}
    defined = {(d["metric"], d["definition"]) for d in definitions()}
    lines = ledger()
    assert lines, "docs/metrics/ledger.tsv has no lines"

    assert {line["run_id"] for line in lines} - ids == set()
    assert {(line["metric"], line["definition"]) for line in lines} - defined == set()
    for line in lines:
        assert re.fullmatch(r"[0-9a-f]{8}", line["fixture_hash"]), line
        float(line["value"])
    keys = [(line["run_id"], line["metric"]) for line in lines]
    assert len(keys) == len(set(keys)), "a run measures a metric twice"


@pytest.mark.infra
def test_definitions_are_numbered_from_one_and_cover_every_metric() -> None:
    """Append-only versions: 1, 2, ... per metric, and every `METRICS` key
    has one, so `--record` can name it."""
    numbers: dict[str, list[int]] = {}
    for d in definitions():
        numbers.setdefault(d["metric"], []).append(int(d["definition"]))
        assert d["since"], d
        assert d["scorer"], d
        assert d["meaning"], d

    assert set(METRICS) <= set(numbers)
    for metric, found in numbers.items():
        assert found == list(range(1, len(found) + 1)), metric


CONVERTED_ROWS = 73
CONVERTED_SHA256 = "0bcb7c57193ac696ed08cca106200ab817914289c3900ba35187373810b44e10"
"""The SHA-256 of the 73 data lines of the hand-edited `docs/metrics.md` at
ef2261d, joined by newlines: the table the ledger was converted from (#620),
read from git, not from the ledger."""


@pytest.mark.infra
def test_the_render_rebuilds_the_converted_rows() -> None:
    """`--render`'s first 73 rows, parsed back and written in the old table's
    form, hash to the table they were converted from: every cell of every
    converted run survives the ledger, `[converted: ...]` remarks aside."""
    rows = parse(render())[:CONVERTED_ROWS]
    lines = [
        "| "
        + " | ".join(
            row[c].split(CONVERTED)[0] if c == "note" else row[c] for c in COLUMNS
        )
        + " |"
        for row in rows
    ]

    assert len(lines) == CONVERTED_ROWS
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == CONVERTED_SHA256


@pytest.mark.infra
def test_a_recorded_run_writes_one_line_per_measured_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write` appends one `runs` line and a `ledger` line per finite metric,
    under its latest definition, and `read` and `best` give them back."""
    for name in ("LEDGER", "RUNS", "DEFINITIONS"):
        path = tmp_path / getattr(metrics, name).name
        path.write_text(getattr(metrics, name).read_text())
        monkeypatch.setattr(metrics, name, path)
    monkeypatch.setattr("port.qa.provenance.head", lambda: "abcdef0")
    recovery = {"fixture_hash": "07b82e92", "ari": 0.98765, "wall": 12.34}
    recovery |= {"copy_ari": float("nan"), "peak_gb": None}

    found = metrics.write(
        recovery, fixture="dev", args="-- --sal", note="a test run", dirty=False
    )

    assert found.startswith("abcdef0-dev-")
    mine = [line for line in metrics.ledger() if line["run_id"] == found]
    assert {line["metric"]: line["value"] for line in mine} == {
        "clone_ari": "0.9877",
        "wall_s": "12.3",
    }
    assert read()[-1]["copy_ari"] == metrics.UNMEASURED
    assert not (metrics.ROOT / "docs" / "metrics.md").exists()
    with pytest.raises(ValueError, match="one dataset"):
        metrics.write(recovery, fixture="easy", args="", note="x", dirty=False)


def _defines(test: str) -> bool:
    path, _, name = test.partition("::")
    source = Path(ROOT, path)
    if not source.is_file():
        return False
    tree = ast.parse(source.read_text())
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body
    )


@pytest.mark.infra
def test_every_run_names_a_test_that_exists() -> None:
    assert _defines(TEST), TEST
    missing = {run["test"] for run in runs() if not _defines(run["test"])}
    assert not missing, f"no such path::function: {sorted(missing)}"


@pytest.mark.infra
@pytest.mark.parametrize(
    "note",
    ["", "   ", "a | b", "a\tb", "one\ntwo", "x" * (NOTE_CHARS + 1)],
)
def test_a_note_that_is_not_one_short_line_is_refused(note: str) -> None:
    check_note("x" * NOTE_CHARS)
    with pytest.raises(ValueError, match="note"):
        check_note(note)


@pytest.mark.infra
def test_the_latest_dev_run_is_the_dev_fixture_built_now() -> None:
    recorded = [row for row in read() if row["fixture"] == "dev"]
    assert recorded, "no dev run: run_ledger --record"

    built = fixture_hash(fixtures.dev_instance())

    assert built == recorded[-1]["fixture_hash"], (
        f"dev_instance now builds {built}, the latest dev run is "
        f"{recorded[-1]['fixture_hash']}: record a run on the new data"
    )


@pytest.mark.infra
def test_the_hash_is_of_the_data_and_moves_with_it() -> None:
    truth = fixtures.critical_instance()

    assert fixture_hash(truth) == fixture_hash(fixtures.critical_instance())
    assert fixture_hash(truth) != fixture_hash(fixtures.critical_instance(seed=1))


@pytest.mark.snapshot
def test_each_fixture_name_holds_one_hash_and_each_hash_one_name() -> None:
    """A history panel is one dataset (#588): across the ledger, `fixture`
    and `fixture_hash` map one to one."""
    lines = ledger()
    hashes: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    for line in lines:
        hashes.setdefault(line["fixture"], set()).add(line["fixture_hash"])
        names.setdefault(line["fixture_hash"], set()).add(line["fixture"])

    assert {f: h for f, h in hashes.items() if len(h) > 1} == {}
    assert {h: f for h, f in names.items() if len(f) > 1} == {}
    for fixture, held in hashes.items():
        check_identity(fixture, held.pop(), lines)


@pytest.mark.snapshot
def test_the_calicost_runs_carry_the_shipped_samples_hash() -> None:
    """`easy` and `hard` lines hash the committed sample as
    `realization_hash` reads it now (#588)."""
    from tests.sim_audit import SAMPLES
    from tests.sim_fixtures import SIM_ROOT
    from tests.sim_stages import realization_hash

    for name, sample in SAMPLES.items():
        recorded = {
            line["fixture_hash"] for line in ledger() if line["fixture"] == name
        }
        assert recorded == {realization_hash(SIM_ROOT / sample)}, name


@pytest.mark.infra
def test_a_name_under_another_hash_is_refused() -> None:
    rows = [{"fixture": "easy", "fixture_hash": "2d4ce9a9"}]

    check_identity("easy", "2d4ce9a9", rows)
    check_identity("hard", "8797710b", rows)
    with pytest.raises(ValueError, match="one dataset"):
        check_identity("easy", "1065eb5b", rows)
    with pytest.raises(ValueError, match="one dataset"):
        check_identity("hard", "2d4ce9a9", rows)


@pytest.mark.infra
def test_the_history_plots_draw_from_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tests.studies.metrics_history` writes both figures from the ledger,
    and each stamps the hash of the ledger's history rows.

    `first_parent` reads `origin/main`, absent from a shallow checkout, so
    the merge order is the ledger's own order here.
    """
    from matplotlib.figure import Figure

    from tests.studies import metrics_history

    rows = metrics_history.history()
    monkeypatch.setattr(
        metrics_history,
        "first_parent",
        lambda: list(dict.fromkeys(r["commit"].rstrip("+") for r in rows)),
    )
    stamps: list[str] = []
    text = Figure.text

    def spy(self: Figure, x: float, y: float, s: str, *a: Any, **k: Any) -> Any:
        stamps.append(s)
        return text(self, x, y, s, *a, **k)

    monkeypatch.setattr(Figure, "text", spy)
    out = [tmp_path / "history.png", tmp_path / "classes.png"]

    metrics_history.main([str(p) for p in out])

    assert all(p.stat().st_size > 0 for p in out)
    data = hashlib.sha256(repr(rows).encode()).hexdigest()[:8]
    assert [s.split()[1] for s in stamps] == [data, data]


def _history_row(commit: str, fixture: str, clone_ari: str) -> dict[str, str]:
    row = dict.fromkeys(COLUMNS, metrics.UNMEASURED)
    row |= {"commit": commit, "fixture": fixture, "fixture_hash": "0" * 8}
    return row | {"note": f"HISTORY #{commit} x", "clone_ari": clone_ari}


@pytest.mark.infra
def test_a_run_of_unchanged_merges_keeps_its_first_and_last_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merges a..d hold one value, e moves it, f adds a fixture, g repeats f:
    a and d stand for a..d with one horizontal `SKIP` between them, below
    the axis; e and f each start a tick; g joins f's run of two, drawn as
    is. Tick labels stay plain `#NNN`."""
    from matplotlib.axes import Axes

    from tests.studies import metrics_history
    from tests.studies.metrics_history import SKIP, axis, label, ticks

    rows = [_history_row(c, "easy", "0.5") for c in "abcd"]
    rows += [_history_row("e", "easy", "0.6")]
    rows += [_history_row(c, "easy", "0.6") for c in "fg"]
    rows += [_history_row(c, "hard", "0.1") for c in "fg"]
    order = {c: k for k, c in enumerate("abcdefg")}

    groups = ticks(rows, order)
    shown, folds = axis(groups, {r["commit"]: label(r) for r in rows})

    assert groups == [["a", "b", "c", "d"], ["e"], ["f", "g"]]
    assert shown == [("a", "#a"), ("d", "#d"), ("e", "#e"), ("f", "#f"), ("g", "#g")]
    assert folds == [0]

    labels: list[str] = []
    marks: list[tuple[str, float, float]] = []
    set_labels, annotate = Axes.set_xticklabels, Axes.annotate

    def spy_labels(self: Axes, names: list[str], *a: Any, **k: Any) -> Any:
        labels.extend(names)
        return set_labels(self, names, *a, **k)

    def spy_annotate(self: Axes, text: str, *a: Any, **k: Any) -> Any:
        marks.append((text, k["xy"][0], k["rotation"]))
        return annotate(self, text, *a, **k)

    monkeypatch.setattr(Axes, "set_xticklabels", spy_labels)
    monkeypatch.setattr(Axes, "annotate", spy_annotate)
    monkeypatch.setattr(metrics_history, "first_parent", lambda: list(order))
    metrics_history.figure(rows, tmp_path / "folded.png")

    assert labels == ["#a", "#d", "#e", "#f", "#g"]
    assert not any(SKIP in name for name in labels)
    ((text, x, rotation),) = marks
    assert text == SKIP
    assert 0 < x < 1
    assert rotation == 0
