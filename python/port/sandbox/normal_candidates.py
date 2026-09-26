"""Normal candidates from the RDR+BAF fit rather than from BAF alone (#320).

`cnaster.normal_spot.determine_normal_candidates` takes the BAF-only clone
with the least BAF deviation and relaxes a log-count percentile until it
admits the whole clone. A tumor clone whose events BAF cannot see -- a
balanced gain such as (3, 3) -- is merged into that clone by the BAF-only
stage and enters the RDR baseline with it: 399 of 400 spots on the lattice
instance, 440 of 879 candidates on the dev instance at convergence
(`docs/audit-recovery.md`).

**Two passes.** The first is an ordinary `run_cnaster_port` run. Its final
RDR+BAF fit names a normal clone the way the shift's pin does
(`neutral_state`: the clone with the largest share of bins in balanced
states), and that clone's spots become the candidates of a second run with
the same configuration. RDR sees what BAF could not, so a clone carrying
balanced gains is separated from the normal one before its spots are summed
into the baseline.

**Measured** (`python -m tests.recovery_audit --two-pass-normal`, against
the default arm; `docs/audit-recovery.md`):

| instance, config | tumor candidates | ARI | mean `mu` err | altered copies exact |
| --- | ---: | ---: | ---: | ---: |
| dev, 5 states, 1 x 3 | 380 -> 49 | 0.919 -> 0.859 | 0.186 -> 0.035 | 0.000 -> 0.471 |
| dev, 10 states, 3 x 30 | 440 -> 0 | 1.000 -> 0.000 (one clone) | diverged | -- |
| lattice, 5 states, 1 x 3 | -> 110 of 110 | 0.606 -> 0.000 (one clone) | -- | -- |
| lattice, 9 states, 3 x 30 | 399 -> 0 | 1.000 -> 1.000 | 0.089 -> 0.026 | 0.484 -> 0.749 |

Where the first pass labels the clones exactly, the second equals #313's
oracle arm to the digit, including the oracle's one-clone collapse at dev
converged. Where it does not (lattice at 5 states, ARI 0.606), the clone it
calls normal is a tumor clone and the run collapses.

**In the sandbox, not the default.** Two of four arms collapse, it costs a
second whole run (144 to 301 s against 35 to 99 s), and a clean baseline
exposes a second defect downstream: the RDR-stage refinement merges to one
clone. It stays here until that is understood (#320).
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = ["candidates", "normal_clone_spots", "two_pass"]


def normal_clone_spots(fit: Any) -> np.ndarray:
    """The spots of the clone a finished fit calls normal.

    `fit` is the mapping `run_cnaster` saves as `rdrbaf_final_*_smp.npz`.
    The normal clone is the one whose decoded path spends the largest share
    of bins in the pinned neutral state, which is how `port`'s shift chooses
    what to pin (#299).
    """
    from port.patch.hmm_nophasing.shifted_emission import neutral_state

    log_mu = np.asarray(fit["new_log_mu"], dtype=np.float64).reshape(-1)
    p_binom = np.asarray(fit["new_p_binom"], dtype=np.float64).reshape(-1)
    path = np.asarray(fit["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1) % log_mu.size

    neutral = neutral_state(log_mu, p_binom, path)
    normal = int(np.argmax((path == neutral).mean(axis=0)))
    assignment = np.asarray(fit["new_assignment"], dtype=np.int64)

    spots: np.ndarray = assignment == normal
    return spots


@contextlib.contextmanager
def candidates(spots: np.ndarray) -> Iterator[None]:
    """`determine_normal_candidates` returns `spots` for the block.

    `cnaster`'s own selection still runs, so its logging and side effects are
    unchanged; only what it returns is replaced, after a check that the two
    cover the same spots.
    """
    import cnaster.scripts.run_cnaster as pipeline

    original = pipeline.determine_normal_candidates
    chosen = np.asarray(spots, dtype=bool)

    def replaced(*arguments: Any, **keywords: Any) -> np.ndarray:
        found = np.asarray(original(*arguments, **keywords))

        if found.shape != chosen.shape:
            msg = f"{chosen.size} candidate flags for {found.size} spots"
            raise ValueError(msg)

        return chosen

    pipeline.determine_normal_candidates = replaced

    try:
        yield
    finally:
        pipeline.determine_normal_candidates = original


def two_pass(argv: Sequence[str]) -> int:
    """Run `run_cnaster_port` twice, the second with the first fit's normal clone.

    `argv` is what `run_cnaster_port` takes: the configuration, then flags.
    The first pass's outputs are kept beside the second's, in
    `<output_dir>_first_pass`, so each directory holds one run.
    """
    from port.scripts.run_cnaster import main

    config = Path(next(argument for argument in argv if not argument.startswith("-")))
    output = Path(yaml.safe_load(config.read_text())["paths"]["output_dir"])

    status = main(list(argv))

    if status:
        return status

    fit = np.load(
        next(output.rglob("rdrbaf_final_nstates*_smp.npz")), allow_pickle=True
    )
    spots = normal_clone_spots(fit)

    first = output.parent / f"{output.name}_first_pass"
    shutil.rmtree(first, ignore_errors=True)
    shutil.move(str(output), str(first))
    output.mkdir(parents=True)

    with candidates(spots):
        return main(list(argv))


if __name__ == "__main__":  # pragma: no cover - a sandbox entry, run by hand
    import sys

    raise SystemExit(two_pass(sys.argv[1:]))
