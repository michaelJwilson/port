"""#490: one arm of the clone-start study.

`run_study --clone-starts SAMPLE START SEED [FLAGS ...]` runs
`--sal --no-plots FLAGS` on `SAMPLE` from the named BAF-stage start and prints
one `ROW` line: start, seed, BAF-stage clone ARI, final clone ARI, clones,
copy ARI, exact altered, phase-free exact altered, wall. `docs/study-clone-starts.md`
holds the sweep.

START is one of
  grid2         cnaster's phasing grid, 2 x 2 rectangles per slice (the default);
  grid3         the same grid at 3 x 3 (`phasing.npart_phasing = 3`);
  grow          `port.sandbox.wolff_init.umi_grow`: clones grown from the deepest
                spots, each to an equal share of SNP UMIs;
  normal-first  a first `--sal` pass names the normal clone (the largest share of
                (1, 1) bins); its spots are one start clone and the grid
                partitions the rest.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from port.qa.audit import audit_sample
from port.sim.fixtures import load_simulated

mpl.use("Agg")


def baf_ari(sample: Any, output: Path) -> float:
    """The BAF stage's clone ARI against the planted labels."""
    labels = pd.read_csv(
        next(output.glob("*/baf_clone_labels.tsv")), sep="\t", index_col=0
    )["clone_label"]
    truth = pd.Series(sample.labels, index=sample.barcodes)
    common = labels.index.intersection(truth.index)
    return float(adjusted_rand_score(truth[common], labels[common]))


def normal_spots(output: Path) -> set[str]:
    """Barcodes of the run's clone with the largest share of (1, 1) bins."""
    seg = pd.read_csv(next(output.glob("*/cnv_seglevel.tsv")), sep="\t")
    clones = sorted({c.split()[0] for c in seg.columns if c.startswith("clone")})
    share = {
        c: float(((seg[f"{c} A"] == 1) & (seg[f"{c} B"] == 1)).mean()) for c in clones
    }
    normal = int(max(share, key=lambda c: share[c]).removeprefix("clone"))
    labels = pd.read_csv(
        next(output.glob("*/clone_labels.tsv")), sep="\t", index_col=0
    )["clone_label"]
    return set(labels.index[labels.to_numpy() == normal])


def main(argv: list[str]) -> None:
    """One arm; one `ROW` line."""
    name, start, seed, flags = argv[0], argv[1], int(argv[2]), argv[3:]
    path = Path(name)
    sample = (
        load_simulated(path.name, path.parent)
        if path.is_absolute()
        else load_simulated(name)
    )
    overrides: dict[str, Any] = {"phasing.npart_phasing": 3} if start == "grid3" else {}
    arm = ["--sal", "--no-plots", *flags]
    opened = time.perf_counter()

    if start == "grow":
        from port.sandbox.wolff_init import wolff_start

        with wolff_start(seed=seed, method="grow"):
            recovery, output = audit_sample(sample, arm, overrides)
    elif start == "normal-first":
        import port.patch.hmrf as patch

        _, first = audit_sample(sample, arm, overrides)
        normal = normal_spots(first)
        order = pd.read_csv(
            next(first.glob("*/clone_labels.tsv")), sep="\t", index_col=0
        ).index
        original = patch.run_core_inference
        done: list[bool] = []

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("params") == "sp" and not done:
                done.append(True)
                index = args[5]
                n_spots = int(sum(len(i) for i in index))
                grid = np.empty(n_spots, dtype=np.int64)
                for c, spots in enumerate(index):
                    grid[spots] = c
                # NB the run's spot order is `clone_labels.tsv`'s.
                is_normal = np.array([b in normal for b in order[:n_spots]])
                _, labels = np.unique(
                    np.where(is_normal, grid.max() + 1, grid), return_inverse=True
                )
                args = (
                    *args[:5],
                    [np.flatnonzero(labels == c) for c in range(labels.max() + 1)],
                    *args[6:],
                )
            return original(*args, **kwargs)

        patch.run_core_inference = wrapped
        try:
            recovery, output = audit_sample(sample, arm, overrides)
        finally:
            patch.run_core_inference = original
    else:
        recovery, output = audit_sample(sample, arm, overrides)

    wall = time.perf_counter() - opened
    print(
        "ROW",
        start,
        seed,
        round(baf_ari(sample, output), 4),
        recovery.ari,
        recovery.n_clones,
        recovery.copy_ari,
        recovery.exact_altered,
        recovery.exact_altered_minor,
        round(wall, 1),
        flush=True,
    )
