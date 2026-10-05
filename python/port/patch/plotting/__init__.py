"""Drop-in replacements for `cnaster`'s plotting entry points (#278).

Every module here replaces a named `cnaster` function, which is what puts it
under `patch/`. What they have in common is the reason the ticket exists:
each one re-derives how `pred_cnv` is laid out, and each one branches on a
second column of the state parameters that the fit cannot produce.
"""

from port.patch.plotting.clone_paths import clone_path, clone_paths, state_vector
from port.patch.plotting.spatial import plot_clones_spatial

MIRRORS: tuple[str, ...] = (
    "cnaster.plot_genomic",
    "cnaster.plotting",
)
"""The pair this package stands in for, as `UNIFIERS` requires.

Both modules re-derive how `pred_cnv` is laid out and both guard the state
parameters' second column, so the duplication is between them rather than
inside either. That is what makes this a unifier and not two patches that
happen to share a helper -- the same reason `patch/lattice` is exempt from
the naming rule. `plot_loh_density`'s replacement, the third, is set aside
under `sandbox/patch/plotting/` (#289, #517 step 8).

`cnaster.plotting` joined with `spatial` (#309), and it is also the name the
package carries, which is where `FIGURE_SWAPS` finds `plot_clones_spatial`.
"""

__all__ = [
    "MIRRORS",
    "clone_path",
    "clone_paths",
    "plot_clones_spatial",
    "state_vector",
]
