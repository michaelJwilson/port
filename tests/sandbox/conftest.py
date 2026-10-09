"""Every test collected here carries `sandbox`, so the per-PR gate can deselect it (#851)."""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the items under this directory before `-m` deselects."""
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(pytest.mark.sandbox)
