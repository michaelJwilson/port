"""Set aside (#556): the clone-labelling problem of a drawn sample with its copy states and profiles known.

Ticket: #556 -- dev_tree's clone field is 10x CalicoST easy/hard's; the
  solvers are compared here on problems whose field strength matches.
Measurement: `docs/study-field-strength.md`, on the 1-slice dev_tree
  (`sim/manifests/dev_tree_1s*.toml`), 5 realizations x 25 random starts.
Exit: `color_merge` graduates to `extensions/` if it lowers `--sal`'s energy
  end to end on dev_tree, easy and hard; `field` stays the study's, since
  the pipeline never knows the planted law.

No pipeline: `port.sim.draw.realize` yields each realization in memory, and
the field is the draw's own log-likelihood of each spot under each clone
(`field`). Only the labels are unknown, so a solver is judged on the labelling
problem alone, against the planted labels and TRW-S's lower bound.
"""

from port.sandbox.known_field.color_merge import color_merge, merge_deltas
from port.sandbox.known_field.field import (
    CLIP,
    KnownProblem,
    baf_field,
    hex_graph,
    overdispersion,
    problems,
    rdr_field,
)

__all__ = [
    "CLIP",
    "KnownProblem",
    "baf_field",
    "color_merge",
    "hex_graph",
    "merge_deltas",
    "overdispersion",
    "problems",
    "rdr_field",
]
