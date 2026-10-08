"""No `port` module imports another's private name, `sandbox/` included (#763)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "python" / "port"

ADMITTED: dict[tuple[str, str], str] = {
    (
        "port.patch.hmm_nophasing.nb_logpmf",
        "_nb_logpmf_1d",
    ): "cnaster's own name, which the row replaces",
}
"""Admitted cross-module private imports, each with why (`docs/port-forward.md`); only
shrinks.
"""


def _private_imports() -> set[tuple[str, str, str]]:
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "port"
            ):
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        found.add(
                            (
                                str(path.relative_to(PACKAGE.parent)),
                                str(node.module),
                                alias.name,
                            )
                        )
    return found


@pytest.mark.infra
def test_no_module_imports_another_modules_private_name() -> None:
    """Every cross-module import of a `_name` is in `ADMITTED`, and every admitted one is still used."""
    found = _private_imports()
    unadmitted = sorted(
        f"{path}: {module}.{name}"
        for path, module, name in found
        if (module, name) not in ADMITTED
    )
    stale = sorted(set(ADMITTED) - {(module, name) for _, module, name in found})

    assert not unadmitted, unadmitted
    assert not stale, stale
