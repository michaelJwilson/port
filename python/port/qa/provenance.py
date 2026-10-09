"""Provenance: the commit a result was produced by, and the data it was produced from.

One home for the git readers and figure stamps the ledger, the badges and the
studies each carried (T- #673 G1). `CLAUDE.md`: every figure prints a hash of
the data it plots and the commit that drew it, and a ledger row names its
commit; both read `commit` here.

`ROOT` is the checkout this module is imported from, so these read a
repository only under an editable install (`port.pth`), which is how the
ledger, the audits and the studies run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from port.extensions.repository import ROOT, commit, dirty, head

__all__ = [
    "CONFIGS",
    "INPUTS",
    "calibration",
    "commit",
    "digest",
    "dirty",
    "head",
    "inputs_hash",
    "stamp",
]

PLOTS = ROOT / ".cache" / "plots"
"""The default root of every generated figure, untracked (`.cache/` is in
`.gitignore`): a figure is a result, regenerated on demand by the command that
draws it, and rule `docs-png` (`tests/test_rules.py`) guards that no PNG is tracked under
`docs/`. Laid out as `docs/plots/` was at `ba34716`; moved from
`port.qa.provenance.PLOTS` (T- #673 G3)."""

CONFIGS = ROOT / "configs"
"""Settings a script measured rather than a person chose: each `<name>.json`
is written by `run_calibrate` and read by the studies and samplers that use it
(#749 WP1). Beside `python/`, so a calibration is a reviewed change to a file
and never a constant in a test or a study."""

INPUTS = (
    "python",
    "scripts",
    "src",
    "tests",
    "configs",
    "pyproject.toml",
    "uv.lock",
    "Cargo.lock",
)
"""What a measured figure is a function of: the code, the tests, the locks and
the configuration that selects them. The ledger refuses a run while these
differ from the commit; the badges skip a pass while their digest holds."""


def calibration(name: str) -> dict[str, Any]:
    """`CONFIGS/<name>.json`, as `run_calibrate` wrote it; its `_provenance` says how."""
    import json

    path = CONFIGS / f"{name}.json"
    if not path.exists():
        msg = f"no calibration {path}: run_calibrate writes it"
        raise FileNotFoundError(msg)
    found: dict[str, Any] = json.loads(path.read_text())
    return found


def digest(payload: bytes) -> str:
    """The data half of a stamp: SHA-256 over `payload`, 8 hex digits."""
    return hashlib.sha256(payload).hexdigest()[:8]


def stamp(data: str, *paths: str) -> str:
    """`data <data> · code <commit>`: what a figure prints, `paths` scoping its `+`."""
    return f"data {data} · code {commit(*paths)}"


def inputs_hash(root: Path = ROOT) -> str:
    """One digest over every `INPUTS` file under `root`, so an unchanged tree skips a pass (#403).

    Paths and contents both enter it, so a renamed file moves it. Bytecode
    and the extension module are outputs, and are left out.
    """
    found = hashlib.sha256()
    files: list[Path] = []

    for name in INPUTS:
        path = root / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix in {".py", ".rs", ".pyi", ".toml", ".cfg"}
            )

    for path in sorted(files):
        found.update(path.relative_to(root).as_posix().encode())
        found.update(path.read_bytes())

    for rc in sorted(root.glob(".coveragerc*")):
        found.update(rc.name.encode())
        found.update(rc.read_bytes())

    return found.hexdigest()[:16]
