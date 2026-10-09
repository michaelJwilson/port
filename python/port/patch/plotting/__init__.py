"""Drop-in replacements for `cnaster`'s plotting entry points (#278)."""

from port.patch.plotting.spatial import plot_clones_spatial

MIRRORS: tuple[str, ...] = (
    "cnaster.plot_genomic",
    "cnaster.plotting",
)
"""The `cnaster` modules this package stands in for, as `UNIFIERS` requires (#289, #309, #517)."""

__all__ = [
    "MIRRORS",
    "plot_clones_spatial",
]
