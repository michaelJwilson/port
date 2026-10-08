"""Where the checkout is, and which commit it is at: the one place the package locates the repository (#749 WP11).

Every module that reads or writes a repository path (`pyproject.toml`,
`docs/`, `.cache/plots/`, `configs/`) joins it to `ROOT` rather than counting
`__file__`'s parents itself. Under an installed wheel `ROOT` is the
environment's `site-packages` parent, not a checkout; the modules that need a
checkout are QA and study tools, and this is the one line to change if that
ever has to hold.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["ROOT", "commit", "dirty", "head"]

ROOT = Path(__file__).resolve().parents[3]
"""The checkout: `python/port/extensions/` three levels down."""


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
    """`head`, with `+` where tracked files under `paths` differ from it.

    Here rather than in `port.qa.provenance`, which re-exports it, because a
    run names its commit in `cnamaste.h5` and a run reaches no QA module (T- #817).
    """
    return head() + ("+" if dirty(*paths) else "")
