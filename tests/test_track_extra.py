"""`aim` is an optional extra: the recording seam works without it (#251, #312).

The packaging rule is `tests/test_rules.py`'s `aim-extra`.
"""

from __future__ import annotations

import pytest
from sal import track


@pytest.mark.smoke
def test_the_recording_seam_imports_without_aim() -> None:
    """`record` through the seam runs without `aim`, using its null store."""

    tracked = track.current()

    assert tracked.is_null, "outside a track() block the bound run must be null"

    tracked.record(0, objective=1.0, seconds=0.5)
