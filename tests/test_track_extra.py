"""`aim` is an optional extra: the recording seam works without it (#251, #312).

`infra`: asserts `port`'s packaging rule, not a scientific result.
"""

from __future__ import annotations

import tomllib

import pytest
from sal import track

from tests import ROOT


@pytest.mark.smoke
def test_the_recording_seam_imports_without_aim() -> None:
    """`record` through the seam runs without `aim`, using its null store."""

    tracked = track.current()

    assert tracked.is_null, "outside a track() block the bound run must be null"

    tracked.record(0, objective=1.0, seconds=0.5)


@pytest.mark.infra
def test_aim_is_declared_as_an_extra_and_never_imported_here() -> None:
    """`aim` is declared as an extra and no `python/port/` module imports it at module scope."""
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = manifest["project"]["optional-dependencies"]

    assert any(name.startswith("aim") for name in extras["track"]), (
        "the `track` extra must pin aim"
    )
    assert not any(
        name.startswith("aim") for name in manifest["project"]["dependencies"]
    ), "aim is an extra, not a dependency"

    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in (ROOT / "python" / "port").rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.startswith(("import aim", "from aim"))
    ]

    assert not offenders, (
        f"module-scope aim import makes the extra mandatory: {offenders}"
    )
