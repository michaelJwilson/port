"""Set aside (#540): copy-state starts at known clones on drawn realizations, polished by the HMM's Baum-Welch.

Ticket: #540 -- which start places the HMM's copy states; revisited on the
  known-law sims of #556 with the clones known.
Measurement: `docs/study-copy-states.md`, on `sim/manifests/dev_tree_1s*.toml`.
Exit: a start graduates to `extensions/copy_starts`' default if, polished by
  Baum-Welch, it reaches the best log-likelihood on easy, hard and dev_tree
  and misses no more states than `--sal`'s `kmeans++x5+em`; else this stays
  the study's.
"""

from port.sandbox.known_copy.hmm import Fit, baum_welch, decode, degenerate, missed
from port.sandbox.known_copy.problem import (
    FLOOR,
    KnownCopyProblem,
    floored_bins,
    problems,
)

__all__ = [
    "FLOOR",
    "Fit",
    "KnownCopyProblem",
    "baum_welch",
    "decode",
    "degenerate",
    "floored_bins",
    "missed",
    "problems",
]
