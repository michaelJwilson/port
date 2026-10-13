"""Set aside (T- #831): #327's decode, one `(A, B)` per continuous state shared by every clone.

Ticket: #327 -- what `cnaster`'s per-state interface can carry; T- #831 set it
  aside with `run_cnaster_port --copy-decode shared`, which no `--sal` run takes.
Measurement: #362 on CalicoST's simulated samples: the lattice decode is best
  on 6 of 8 fits by copy ARI and within 0.004 on the other 2.
Exit: retire unless a study needs a per-state decode beside the lattice's.
"""

from __future__ import annotations

import numpy as np
from sal.opt.termination import Termination

from port.extensions.copy_likelihood import (
    CopyFit,
    Pseudobulk,
    candidates,
    pair_rate_and_share,
    pseudobulk_log_pmf,
)

__all__ = ["shared_decode"]


def shared_decode(
    clones: list[tuple[np.ndarray, Pseudobulk, float]],
    *,
    n_states: int,
    normal: int,
    max_total_copy: int,
    max_allele_copy: int | None = None,
) -> CopyFit:
    """Each continuous state's `(A, B)`, one pair shared by every clone.

    The continuous paths, shifts and dispersions are held, so the
    likelihood is a sum over states of terms each depending on one state's
    pair, and each state's argmax over the lattice solves the one-pair-per-
    state MILP exactly. `normal` is `(1, 1)`; a state no clone visits is too.
    """
    lattice = candidates(max_total_copy, max_allele_copy)
    log_mu, p = pair_rate_and_share(lattice)
    states = np.ones((n_states, 2), dtype=np.int64)
    paths = [np.asarray(path, dtype=np.int64) for path, _, _ in clones]
    total = 0.0

    for k in np.unique(np.concatenate(paths)):
        state = int(k)
        scores = np.zeros(len(lattice))

        for path, (_, bulk, shift) in zip(paths, clones, strict=True):
            bins = np.flatnonzero(path == state)

            if bins.size:
                scores += np.array(
                    [
                        np.sum(pseudobulk_log_pmf(log_mu[i] - shift, p[i], bulk, bins))
                        for i in range(len(lattice))
                    ]
                )

        best = (
            int(np.flatnonzero((lattice[:, 0] == 1) & (lattice[:, 1] == 1))[0])
            if state == normal
            else int(np.argmax(scores))
        )
        states[state] = lattice[best]
        total += float(scores[best])

    bulks = [bulk for _, bulk, _ in clones]
    return CopyFit(
        [states[path] for path in paths],
        states,
        paths,
        np.array([shift for _, _, shift in clones], dtype=np.float64),
        np.ones(len(clones)),
        bulks[0].dispersion,
        bulks[0].taus,
        total,
        # NB exact: each state's argmax over the lattice is the MILP's.
        termination=Termination.after(1, converged=True),
    )
