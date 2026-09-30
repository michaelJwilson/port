"""`port.sandbox.patch.hmm_nophasing.nb_logpmf`, re-exported at the path `sandbox.known_copy` imports (#560).

Ticket: #560 -- the log-space negative binomial moved to
  `port.sandbox.patch.hmm_nophasing.nb_logpmf` once the patch kernels imported it.
Measurement: none of its own; the kernel's is in that module.
Exit: retire when `sandbox.known_copy` imports from `port.patch`.
"""

from __future__ import annotations

from port.patch.hmm_nophasing.nb_logpmf import _dense_nb_logpmf, _nb_logpmf_1d, patched

__all__ = ["_dense_nb_logpmf", "_nb_logpmf_1d", "patched"]
