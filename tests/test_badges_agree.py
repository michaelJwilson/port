"""`scripts.badges` writes the same bytes twice.

That the badges are what it derives is `tests/test_rules.py`'s `badge-*` rules.
"""

from pathlib import Path

import pytest

from scripts.badges import write


@pytest.mark.infra
def test_the_generator_is_idempotent(tmp_path: Path) -> None:
    """Two generator runs write the same bytes."""
    before = {path.name: path.read_bytes() for path in write()}
    after = {path.name: path.read_bytes() for path in write()}

    assert before == after
