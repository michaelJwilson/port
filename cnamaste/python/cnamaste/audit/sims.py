"""The staged fixtures: each one's committed sample and staged file, and the ones a bare run selects (Ticket#836)."""

from __future__ import annotations

SUPPORTED_SIMS: dict[str, tuple[str, str]] = {
    "2d4ce9a9": ("sim/numcnas1.2_cnasize5e7_ploidy2_random0", "tests/data/sim_2d4ce9a9.hdf5"),
}
"""Fixture hash -> (the committed sample, relative to the repository; its staged file, relative to `cnamaste/`):
the fixtures every file can select (`--sims`)."""

DEFAULT_SIMS = ("2d4ce9a9",)

TRUTH_SIMS: dict[str, str] = {
    "8797710b": "tests/data/sim_8797710b.hdf5",
    "23acef50": "tests/data/sim_23acef50.hdf5",
    "e4bbe03f": "tests/data/sim_e4bbe03f.hdf5",
}
"""Fixture hash -> its staged file (captured at efb2e9e, with `/truth`), scored by `test_truth.py` alone: CalicoST
hard, `dev_tree_easy_square` r0 and `dev_tree_easy` r0. The other files pin values per fixture in tables that hold
easy's alone; a fixture moves to `SUPPORTED_SIMS` with its tables (PR- #899). Only shrinks."""
