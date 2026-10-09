"""The checkout's root and commit: the one place the package locates the repository (#749 WP11).

Under an installed wheel `ROOT` is not a checkout.
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
    """Whether the tree differs from `head` under `paths` (all if none); untracked files count only if `untracked`."""
    flags = () if untracked else ("--untracked-files=no",)
    return bool(_git("status", "--porcelain", *flags, "--", *(paths or (".",))))


def commit(*paths: str) -> str:
    """`head`, with `+` where tracked files under `paths` differ from it (T- #817)."""
    return head() + ("+" if dirty(*paths) else "")
