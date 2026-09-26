"""`docs/metrics.md` parses, and its latest `dev` row is the fixture built now (#409).

The hash check is what makes a row reproducible rather than a number with a
commit beside it: a change that moves what `dev_instance` builds leaves every
earlier row describing data that no longer comes out, and fails here until a
row on the new data is recorded.
"""

import datetime
import re

import pytest

from tests import fixtures
from tests.metrics import COLUMNS, METRICS, UNMEASURED, fixture_hash, read


@pytest.mark.infra
def test_every_row_parses_and_is_in_date_order() -> None:
    rows = read()
    assert rows, "docs/metrics.md has no rows"

    for row in rows:
        assert set(row) == set(COLUMNS)
        assert re.fullmatch(r"[0-9a-f]{7}\+?", row["commit"]), row["commit"]
        datetime.date.fromisoformat(row["date"])
        assert re.fullmatch(r"[0-9a-f]{8}", row["fixture_hash"]), row
        for column in METRICS:
            if row[column] != UNMEASURED:
                float(row[column])

    dates = [row["date"] for row in rows]
    assert dates == sorted(dates)


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
