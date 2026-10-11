"""The staged fixtures: each one's name, staged file and inputs, and the ones a bare run selects (Ticket#836)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sim:
    name: str
    """The fixture's name, as the ledger writes it."""
    staged: str
    """Its staged file, relative to `cnamaste/`."""
    sample: str | None
    """Its committed inputs, relative to the repository; None where they are not committed."""
    regenerate: str = ""
    """How to regenerate inputs that are not committed."""


SUPPORTED_SIMS: dict[str, Sim] = {
    "2d4ce9a9": Sim("CalicoST easy", "tests/data/sim_2d4ce9a9.hdf5", "sim/numcnas1.2_cnasize5e7_ploidy2_random0"),
    "8797710b": Sim("CalicoST hard", "tests/data/sim_8797710b.hdf5", "sim/numcnas6.3_cnasize1e7_ploidy2_random0"),
    "e4bbe03f": Sim("dev_tree_easy r0", "tests/data/sim_e4bbe03f.hdf5", None,
                    "realization 0 drawn from sim/manifests/dev_tree_easy.toml: two slices of hexagonal Visium spots, 6,000 spots"),
    "23acef50": Sim("dev_tree_easy_square r0", "tests/data/sim_23acef50.hdf5", None,
                    "realization 0 drawn from sim/manifests/dev_tree_easy_square.toml: two slices of standard Visium spots "
                    "(55 um) on a square lattice, 6,000 spots"),
}
"""Fixture hash -> its name, staged file and inputs."""

DEFAULT_SIMS = ("2d4ce9a9", "23acef50")
"""The fixtures a bare `pytest` runs: CalicoST easy, and two slices of standard Visium spots on a square lattice.
CalicoST hard and `dev_tree_easy` r0 are opt-in (`--sims`)."""

FN_DEFAULT_SIMS = ("2d4ce9a9",)
"""The fixtures a bare `pytest` runs the per-function `test_fn_*` files on: CalicoST easy; `--sims` widens them."""

INPUTS_ENV = "CNAMASTE_SIM_INPUTS"
"""Where regenerated inputs are, `$CNAMASTE_SIM_INPUTS/<hash>/` each, when `--sim-inputs` is not given."""
