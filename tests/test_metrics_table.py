"""`docs/metrics.md` parses, and its latest `dev` row is the fixture built now (#409).

The hash check is what makes a row reproducible rather than a number with a
commit beside it: a change that moves what `dev_instance` builds leaves every
earlier row describing data that no longer comes out, and fails here until a
row on the new data is recorded.
"""

import ast
import datetime
import re
from pathlib import Path

import pytest

from tests import fixtures
from tests.metrics import (
    COLUMNS,
    METRICS,
    NOTE_CHARS,
    ROOT,
    TEST,
    TIMESTAMP,
    UNMEASURED,
    check_identity,
    check_note,
    fixture_hash,
    read,
)


@pytest.mark.infra
def test_every_row_parses_and_is_in_timestamp_order() -> None:
    rows = read()
    assert rows, "docs/metrics.md has no rows"

    for row in rows:
        assert set(row) == set(COLUMNS)
        assert re.fullmatch(r"[0-9a-f]{7}\+?", row["commit"]), row["commit"]
        datetime.datetime.strptime(row["timestamp"], TIMESTAMP).replace(
            tzinfo=datetime.UTC
        )
        check_note(row["note"])
        assert re.fullmatch(r"[0-9a-f]{8}", row["fixture_hash"]), row
        for column in METRICS:
            if row[column] != UNMEASURED:
                float(row[column])

    stamps = [row["timestamp"] for row in rows]
    assert stamps == sorted(stamps)


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
def test_every_row_names_a_test_that_exists() -> None:
    assert _defines(TEST), TEST
    missing = {row["test"] for row in read() if not _defines(row["test"])}
    assert not missing, f"no such path::function: {sorted(missing)}"


@pytest.mark.infra
@pytest.mark.parametrize(
    "note",
    ["", "   ", "a | b", "one\ntwo", "x" * (NOTE_CHARS + 1)],
)
def test_a_note_that_is_not_one_short_line_is_refused(note: str) -> None:
    check_note("x" * NOTE_CHARS)
    with pytest.raises(ValueError, match="note"):
        check_note(note)


@pytest.mark.infra
def test_the_latest_dev_row_is_the_dev_fixture_built_now() -> None:
    recorded = [row for row in read() if row["fixture"] == "dev"]
    assert recorded, "no dev row: python -m tests.metrics --record"

    built = fixture_hash(fixtures.dev_instance())

    assert built == recorded[-1]["fixture_hash"], (
        f"dev_instance now builds {built}, the latest dev row is "
        f"{recorded[-1]['fixture_hash']}: record a row on the new data"
    )


@pytest.mark.infra
def test_the_hash_is_of_the_data_and_moves_with_it() -> None:
    truth = fixtures.critical_instance()

    assert fixture_hash(truth) == fixture_hash(fixtures.critical_instance())
    assert fixture_hash(truth) != fixture_hash(fixtures.critical_instance(seed=1))


@pytest.mark.snapshot
def test_each_fixture_name_holds_one_hash_and_each_hash_one_name() -> None:
    """A history panel is one dataset (#588): across `read()`, `fixture` and
    `fixture_hash` map one to one."""
    rows = read()
    hashes: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    for row in rows:
        hashes.setdefault(row["fixture"], set()).add(row["fixture_hash"])
        names.setdefault(row["fixture_hash"], set()).add(row["fixture"])

    assert {f: h for f, h in hashes.items() if len(h) > 1} == {}
    assert {h: f for h, f in names.items() if len(f) > 1} == {}
    for row in rows:
        check_identity(row["fixture"], row["fixture_hash"], rows)


@pytest.mark.snapshot
def test_the_calicost_rows_carry_the_shipped_samples_hash() -> None:
    """`easy` and `hard` rows hash the committed sample as
    `realization_hash` reads it now (#588)."""
    from tests.sim_audit import SAMPLES
    from tests.sim_fixtures import SIM_ROOT
    from tests.sim_stages import realization_hash

    for name, sample in SAMPLES.items():
        recorded = {row["fixture_hash"] for row in read() if row["fixture"] == name}
        assert recorded == {realization_hash(SIM_ROOT / sample)}, name


@pytest.mark.infra
def test_a_name_under_another_hash_is_refused() -> None:
    rows = [{"fixture": "easy", "fixture_hash": "23989aa4"}]

    check_identity("easy", "23989aa4", rows)
    check_identity("hard", "1ae26365", rows)
    with pytest.raises(ValueError, match="one dataset"):
        check_identity("easy", "1065eb5b", rows)
    with pytest.raises(ValueError, match="one dataset"):
        check_identity("hard", "23989aa4", rows)
