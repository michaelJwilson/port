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
