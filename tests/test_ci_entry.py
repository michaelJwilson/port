"""The input digest a badge records moves with an input and not otherwise (#403).

That `scripts.ci`'s steps select every test once is `tests/test_rules.py`'s `ci-steps`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from port.qa.provenance import inputs_hash


@pytest.mark.infra
def test_the_input_hash_moves_with_an_input_and_not_otherwise(tmp_path: Path) -> None:
    """An edit moves the input digest; a badge write does not."""

    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "a.py").write_text("x = 1\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / ".badges").mkdir()
    first = inputs_hash(tmp_path)
    (tmp_path / ".badges" / "measurements.json").write_text("{}\n")
    assert inputs_hash(tmp_path) == first, "a badge write moved the digest"

    (tmp_path / "python" / "a.py").write_text("x = 2\n")
    assert inputs_hash(tmp_path) != first, "an edit to the code did not move it"

    (tmp_path / "python" / "a.py").write_text("x = 1\n")
    assert inputs_hash(tmp_path) == first
    (tmp_path / "python" / "a.py").rename(tmp_path / "python" / "b.py")
    assert inputs_hash(tmp_path) != first, "a rename did not move it"
