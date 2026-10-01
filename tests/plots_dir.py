"""Where the figure generators write by default: `.cache/plots/`, untracked.

`docs/plots/` no longer tracks PNGs (`.github/workflows/figures.yml` uploads, and no longer commits): a figure
is a result, regenerated on demand by the command that draws it, and a
generator whose default output is the repository re-adds what was removed.
`.cache/` is in `.gitignore`, so a run with no output argument leaves
`git status` clean. `tests/test_ci_entry.py` guards that no PNG is tracked
under `docs/plots/`.
"""

from pathlib import Path

PLOTS = Path(__file__).resolve().parent.parent / ".cache" / "plots"
"""The default root of every generated figure, mirroring `docs/plots/`'s layout."""
