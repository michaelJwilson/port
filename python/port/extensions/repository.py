"""Where the checkout is: the one place the package locates the repository (#749 WP11).

Every module that reads or writes a repository path (`pyproject.toml`,
`docs/`, `.cache/plots/`, `configs/`) joins it to `ROOT` rather than counting
`__file__`'s parents itself. Under an installed wheel `ROOT` is the
environment's `site-packages` parent, not a checkout; the modules that need a
checkout are QA and study tools, and this is the one line to change if that
ever has to hold.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ROOT"]

ROOT = Path(__file__).resolve().parents[3]
"""The checkout: `python/port/extensions/` three levels down."""
