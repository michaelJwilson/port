"""`aim` is an extra, and the recording path does not need it installed.

**#251.** `snakes_and_ladders.track.Run` is a `runtime_checkable` Protocol
written with `aim.Run`'s own signatures, so `aim.Run` satisfies it
structurally and sal imports `aim` nowhere at module scope. That property is
the whole justification for declaring `aim` in an extra rather than as a
dependency: if it were false, a default install would have to carry a web
server and two unfixed advisories (`pyproject.toml`, the `track` extra).

**The seam is in the revision `port` pins** since #312's step 0 moved
`[tool.uv.sources]`'s lock from `c9f4250` to `186bc59`; before that the two
tests that import it skipped. They import it directly now, so a pin that
loses `track.py` fails here rather than skipping.

`infra`: these assert `port`'s own packaging rule, not anything about
`cnaster` or about a scientific result. The packaging test passes whether or
not a reader installed the extra, which is deliberate -- a test that
asserted `aim` was absent would fail for exactly the reader who followed the
README.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.infra
def test_the_recording_seam_imports_without_aim() -> None:
    """Import and use the seam, and let it pick its own null store.

    Calls `record` rather than only importing: the claim is that an
    untracked run costs nothing and needs nothing, and an import alone would
    not catch a store that reached for `aim` on first use.
    """
    from snakes_and_ladders import track

    tracked = track.current()

    assert tracked.is_null, "outside a track() block the bound run must be null"

    tracked.record(0, objective=1.0, seconds=0.5)


@pytest.mark.infra
def test_a_store_satisfies_the_protocol_structurally() -> None:
    """`Run` is satisfied by shape, which is what lets `aim.Run` in unimported.

    `MemoryRun` is upstream's in-process store. If it stopped satisfying
    `Run`, the Protocol would have drifted from the three members a hook
    uses, and `aim.Run` would be no more admissible than anything else.
    """
    from snakes_and_ladders import track

    assert isinstance(track.MemoryRun(), track.Run)
    assert isinstance(track.NULL_RUN, track.Run)


@pytest.mark.infra
def test_aim_is_declared_as_an_extra_and_never_imported_here() -> None:
    """The packaging half: an extra, and no module-scope import in `port`.

    A module-scope `import aim` anywhere in `python/port/` would make the
    extra mandatory in fact while staying optional in `pyproject.toml`,
    which is the failure this refuses. Tracking code reaches `aim` only
    through `snakes_and_ladders.track`, and an entry point that opens a
    store imports it inside the function that does.
    """
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
