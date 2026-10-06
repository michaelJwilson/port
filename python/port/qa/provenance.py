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
import subprocess
from pathlib import Path

__all__ = [
    "INPUTS",
    "ROOT",
    "commit",
    "digest",
    "dirty",
    "head",
    "inputs_hash",
    "stamp",
]

ROOT = Path(__file__).resolve().parents[3]
"""The checkout: `python/port/qa/` three levels down."""

INPUTS = ("python", "src", "tests", "pyproject.toml", "uv.lock", "Cargo.lock")
"""What a measured figure is a function of: the code, the tests, the locks and
the configuration that selects them. The ledger refuses a run while these
differ from the commit; the badges skip a pass while their digest holds."""


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def head() -> str:
    """The checkout's commit, 7 hex digits; raises outside a repository."""
    return _git("rev-parse", "--short=7", "HEAD")


def dirty(*paths: str, untracked: bool = False) -> bool:
    """Whether the tree differs from `head` under `paths` (all of it if none).

    An untracked file counts only where `untracked` says so: the ledger's
    inputs count one, a figure's stamp does not.
    """
    flags = () if untracked else ("--untracked-files=no",)
    return bool(_git("status", "--porcelain", *flags, "--", *(paths or (".",))))


def commit(*paths: str) -> str:
    """`head`, with `+` where tracked files under `paths` differ from it."""
    return head() + ("+" if dirty(*paths) else "")


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
