"""`tests.correspondence`'s rows, one test per tier and referee (#850)."""

import pytest

from tests.correspondence import Correspondence, rows


def _agrees(row: Correspondence, request: pytest.FixtureRequest) -> None:
    if row.config:
        request.getfixturevalue("cnaster_config")
    theirs = row.call(row.upstream, *row.inputs(), **row.keywords)
    ours = row.call(row.ours, *row.inputs(), **row.keywords)
    row.compare(ours, theirs)


@pytest.mark.patch
@pytest.mark.parametrize("row", rows("", "patch"))
def test_patch(row: Correspondence, request: pytest.FixtureRequest) -> None:
    """Port's drop-in returns what the `cnaster` function it replaces returns, by the row's comparison."""
    _agrees(row, request)


@pytest.mark.patch
@pytest.mark.merge
@pytest.mark.parametrize("row", rows("merge", "patch"))
def test_patch_merge(row: Correspondence, request: pytest.FixtureRequest) -> None:
    """As `test_patch`, for rows over the gate's 5-second cap."""
    _agrees(row, request)
