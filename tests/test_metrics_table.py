"""The `docs/metrics/` ledger is consistent and its latest `dev` run hashes the fixture
built now (#409, #620).
"""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from port.qa import ledger as metrics
from port.qa.ledger import (
    COLUMNS,
    NOTE_CHARS,
    check_identity,
    check_note,
    ledger,
    read,
)
from port.sim import truth as sim_truth
from port.sim.fixtures import SAMPLES, SIM_ROOT, realization_hash
from port.studies import metrics_history
from port.studies.metrics_history import SKIP, axis, label, ticks

from tests import ROOT
from tests.metrics import fixture_hash


@pytest.mark.smoke
def test_a_recorded_run_writes_one_line_per_measured_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write` appends one `runs` line and a `ledger` line per finite metric; `read` and `best` return them."""
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
    assert {(line["fixture"], line["fixture_hash"]) for line in mine} == {
        ("dev", "07b82e92")
    }
    run = next(r for r in metrics.runs() if r["run_id"] == found)
    assert run["benchmark"] == "false"
    assert {line["metric"]: line["value"] for line in mine} == {
        "clone_ari": "0.9877",
        "wall_s": "12.3",
    }
    assert read()[-1]["copy_ari"] == metrics.UNMEASURED
    assert not (ROOT / "docs" / "metrics.md").exists()
    with pytest.raises(ValueError, match="one dataset"):
        metrics.write(recovery, fixture="easy", args="", note="x", dirty=False)


@pytest.mark.warning
@pytest.mark.parametrize(
    "note",
    ["", "   ", "a | b", "a\tb", "one\ntwo", "x" * (NOTE_CHARS + 1)],
)
def test_a_note_that_is_not_one_short_line_is_refused(note: str) -> None:
    check_note("x" * NOTE_CHARS)
    with pytest.raises(ValueError, match="note"):
        check_note(note)


@pytest.mark.analytic
def test_the_hash_is_of_the_data_and_moves_with_it() -> None:
    truth = sim_truth.critical_instance()

    assert fixture_hash(truth) == fixture_hash(sim_truth.critical_instance())
    assert fixture_hash(truth) != fixture_hash(sim_truth.critical_instance(seed=1))


@pytest.mark.snapshot
def test_each_hash_holds_one_name() -> None:
    """Each `fixture_hash` sits under one `fixture`, and every pair passes `check_identity` (#588, #739)."""
    lines = ledger()
    names: dict[str, set[str]] = {}
    for line in lines:
        names.setdefault(line["fixture_hash"], set()).add(line["fixture"])

    assert {h: f for h, f in names.items() if len(f) > 1} == {}
    for digest, held in names.items():
        check_identity(held.pop(), digest, lines)


@pytest.mark.snapshot
def test_the_calicost_runs_carry_the_shipped_samples_hash() -> None:
    """`easy` and `hard` lines hash the committed sample as `realization_hash` reads it now (#588)."""

    for name, sample in SAMPLES.items():
        recorded = {
            line["fixture_hash"] for line in ledger() if line["fixture"] == name
        }
        assert recorded == {realization_hash(SIM_ROOT / sample)}, name


@pytest.mark.smoke
def test_best_takes_a_name_alone_only_where_it_holds_one_hash() -> None:
    """`easy` holds one hash; `dev_tree_1s_hard_r0` holds two generations."""
    found = metrics.best("clone_ari", "easy")
    assert found is not None
    assert (found["fixture"], found["fixture_hash"]) == ("easy", "2d4ce9a9")
    with pytest.raises(ValueError, match="2 datasets"):
        metrics.best("clone_ari", "dev_tree_1s_hard_r0")
    found = metrics.best("clone_ari", "dev_tree_1s_hard_r0", "9ec90dc2")
    assert found is not None
    assert found["fixture_hash"] == "9ec90dc2"


@pytest.mark.warning
def test_a_hash_under_another_name_is_refused() -> None:
    """A new generation under a held name is a new dataset; a held hash under a new name is refused."""
    rows = [{"fixture": "easy", "fixture_hash": "2d4ce9a9"}]

    check_identity("easy", "2d4ce9a9", rows)
    check_identity("hard", "8797710b", rows)
    check_identity("easy", "1065eb5b", rows)
    with pytest.raises(ValueError, match="one dataset"):
        check_identity("hard", "2d4ce9a9", rows)
    with pytest.raises(ValueError, match="8 hex digits"):
        check_identity("easy", "2d4ce9a", rows)


@pytest.mark.smoke
def test_the_history_plots_draw_from_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`port.studies.metrics_history` writes both figures, each stamped with the history rows' hash."""

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


@pytest.mark.smoke
def test_a_run_of_unchanged_merges_keeps_its_first_and_last_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged merges collapse to one `SKIP`; a change or new fixture starts a tick labelled `#NNN`."""

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
