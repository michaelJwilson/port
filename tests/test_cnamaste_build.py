"""`cnamaste/` is the pinned `cnaster`'s run path, renamed, plus declared edits, and owns its code and environment (T- #836).

Two build guards, not tests of the subject: `tests/test_cnamaste_end2end.py`
is what judges cnamaste.
"""

from __future__ import annotations

import ast

import pytest

from tests import ROOT

FORBIDDEN = frozenset({"cnaster", "port", "sal", "snakes_and_ladders", "oxiport"})
"""What cnamaste copies from rather than imports."""


@pytest.mark.infra
def test_the_copy_is_the_pinned_cnaster_renamed_plus_declared_edits() -> None:
    """`scripts.vendor_cnamaste --check`: every copied file is the pin's, renamed, or declared in `DEPARTED`."""
    from scripts.vendor_cnamaste import check, copy

    assert check(copy()) == []


@pytest.mark.infra
def test_cnamaste_imports_nothing_it_does_not_own() -> None:
    """No module under `cnamaste/python/`, `tests/` or `scripts/` imports `cnaster`, port, sal or `oxiport`."""
    found = []
    for path in sorted(
        p
        for d in ("python", "tests", "scripts")
        for p in (ROOT / "cnamaste" / d).rglob("*.py")
    ):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    and node.module
                    and not node.level
                    else []
                )
            )
            found += [
                f"{path.name}: {n}" for n in names if n.split(".")[0] in FORBIDDEN
            ]
    assert found == []


@pytest.mark.infra
def test_cnamastes_environment_holds_none_of_them() -> None:
    """`cnamaste/uv.lock` locks no `cnaster`, port, sal or `oxiport`: its environment is its own."""
    import tomllib

    locked = {
        p["name"]
        for p in tomllib.loads((ROOT / "cnamaste" / "uv.lock").read_text())["package"]
    }
    assert locked & (FORBIDDEN | {"snakes-and-ladders"}) == set()
    assert "cnamaste" in locked
