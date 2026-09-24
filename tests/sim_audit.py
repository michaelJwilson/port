"""`run_cnaster_port` on CalicoST's simulated samples, scored (#362).

Run as `python -m tests.sim_audit [--sample easy|hard|<name>] [--set k=v]
[-- flags]`: one arm, one `SIM` line of JSON on stdout.

The four ARIs are `tests.recovery_audit`'s, on the sample's truth:

- **clone**: fitted clone per spot against the planted clone;
- **clone, integer**: the same after merging fitted clones of one decoded
  `(A, B)` profile (#344);
- **copy state**: per matched clone-bin, the decoded state `Z` against the
  planted `(A, B)` at the bin's midpoint;
- **copy state, integer**: the decoded `(A, B)` against the same.

`--pure` first redraws the tumour spots as pure tumour
(`tests.sim_fixtures.purify`): the simulated spots carry about 8 per cent
normal admixture, which no pair `(A, B)` at `p = A / (A + B)` can fit.

`--decorrelated` first zeroes, in every spot, the tested genes whose shared
tumour expression offset is largest (`tests.sim_fixtures.decorrelate`,
#372), keeping the 20 per cent whose expression follows the normal baseline
times the planted copy factor.

`--oracle-start` sets `annotation.clone_label` to the sample's
`truth_clone_labels.tsv`, `cnaster`'s own known-labels mode: the planted
clones start phasing and the BAF stage in place of the grid
(`run_cnaster.py:244`) and the read-depth stage (`:1059`), and the normal
baseline is taken from the planted normal spots (`annotation.py:34`). It
also sets `hmrf.fixed_assignment`, which holds them there: no ICM move, floor
merge or clone loss (`hmrf.py:287`). The arm that says what the copy decode
recovers when the clones are right.

Each planted clone is matched to the fitted clone it overlaps most (Hungarian
on the spot overlap). `exact` is the share of matched clone-bins whose
decoded `(A, B)` is the planted pair, and `exact_altered` the same over bins
where the planted pair is not `(1, 1)`: `run_sim_analysis`'s `correct_rate`
without the phase flip. `exact_altered_minor` allows it: a decoded `(B, A)`
counts, so the minor and major copies are scored and the phase is not.
"""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from tests.recovery_audit import integer_clones
from tests.sim_fixtures import EASY, HARD, SimulatedSample, load_simulated

SAMPLES = {"easy": EASY, "hard": HARD}


@dataclass
class SimRecovery:
    """One arm on one sample."""

    sample: str
    arm: str
    wall: float
    peak_gb: float
    ari: float
    ari_integer: float
    state_ari: float
    copy_ari: float
    n_clones: int
    n_integer_clones: int
    exact: float
    exact_altered: float
    exact_altered_minor: float
    bins: int
    clone_of: dict[int, int] = field(default_factory=dict)


def _barcode(values: pd.Series) -> np.ndarray:
    barcodes: np.ndarray = values.astype(str).to_numpy()
    return barcodes


def read_run(sample: SimulatedSample, output: Path) -> dict[str, Any]:
    """Fitted labels per truth spot, and per-bin `Z`, `A`, `B` per fitted clone."""
    run = next(output.rglob("rdrbaf_final_nstates*_smp.npz"))
    fit = np.load(run, allow_pickle=True)
    table = pd.read_csv(run.parent / "clone_labels.tsv", sep="\t", comment="#")
    by_barcode = dict(
        zip(_barcode(table["barcode"]), table["clone_label"].to_numpy(), strict=True)
    )
    labels = np.array([by_barcode.get(b, -1) for b in sample.barcodes])

    seglevel = pd.read_csv(run.parent / "cnv_seglevel.tsv", sep="\t")
    n_fitted = int(labels.max()) + 1
    n_states = np.asarray(fit["new_log_mu"]).shape[0]
    pred = np.asarray(fit["pred_cnv"]).reshape(len(seglevel), -1) % n_states
    a = np.stack([seglevel[f"clone{c} A"].to_numpy() for c in range(n_fitted)], 1)
    b = np.stack([seglevel[f"clone{c} B"].to_numpy() for c in range(n_fitted)], 1)

    return {
        "labels": labels,
        "seglevel": seglevel,
        "pred": pred[:, :n_fitted] if pred.shape[1] >= n_fitted else pred,
        "a": a,
        "b": b,
    }


def score(sample: SimulatedSample, output: Path, arm: str, wall: float) -> SimRecovery:
    """The four ARIs, exact copies, and the matching behind them."""
    from sklearn.metrics import adjusted_rand_score

    run = read_run(sample, output)
    fitted = run["labels"]
    scored = fitted >= 0
    ari = float(adjusted_rand_score(sample.labels[scored], fitted[scored]))
    merged = integer_clones(run["a"], run["b"])
    ari_integer = float(
        adjusted_rand_score(sample.labels[scored], merged[fitted[scored]])
    )

    overlap = np.zeros((sample.n_clones, int(fitted.max()) + 1), dtype=np.int64)
    np.add.at(overlap, (sample.labels[scored], fitted[scored]), 1)
    rows, columns = linear_sum_assignment(-overlap)
    clone_of = dict(zip(rows.tolist(), columns.tolist(), strict=True))

    seglevel = run["seglevel"]
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    chromosome = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    planted = sample.copies_at(chromosome, middle)
    covered = planted[:, 0, 0] >= 0

    truth, state, pair = [], [], []
    for clone, fit in clone_of.items():
        truth.append(planted[covered, clone, 0] * 1_000 + planted[covered, clone, 1])
        state.append(run["pred"][covered, fit])
        pair.append(run["a"][covered, fit] * 1_000 + run["b"][covered, fit])

    t, z, ab = (np.concatenate(x) for x in (truth, state, pair))
    altered = t != 1_001
    swapped = (ab % 1_000) * 1_000 + ab // 1_000
    either = (t == ab) | (t == swapped)

    return SimRecovery(
        sample=sample.name,
        arm=arm,
        wall=round(wall, 2),
        peak_gb=0.0,
        ari=round(ari, 4),
        ari_integer=round(ari_integer, 4),
        state_ari=round(float(adjusted_rand_score(t, z)), 4),
        copy_ari=round(float(adjusted_rand_score(t, ab)), 4),
        n_clones=int(np.unique(fitted[scored]).size),
        n_integer_clones=int(np.unique(merged[fitted[scored]]).size),
        exact=round(float(np.mean(t == ab)), 4),
        exact_altered=round(float(np.mean((t == ab)[altered])), 4),
        exact_altered_minor=round(float(np.mean(either[altered])), 4),
        bins=int(covered.sum()),
        clone_of=clone_of,
    )


def run_arm(
    sample: SimulatedSample,
    flags: list[str],
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
    oracle: bool = False,
) -> tuple[SimRecovery, Path]:
    """`run_cnaster_port` with `flags` on `sample`, scored."""
    from port.scripts.run_cnaster import main

    from tests.sim_fixtures import write_sim_inputs

    root = Path(tempfile.mkdtemp()) if root is None else root
    known = {
        "annotation.clone_label": str(sample.path / "truth_clone_labels.tsv"),
        "hmrf.fixed_assignment": True,
    }
    config = write_sim_inputs(
        sample, root, {**(overrides or {}), **(known if oracle else {})}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = time.perf_counter()
        main([*flags, str(config)])
        wall = time.perf_counter() - started

    output = root / "output"
    arm = " ".join(["oracle-start", *flags] if oracle else flags) or "default"
    return score(sample, output, arm, wall), output


def main() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default="easy")
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--oracle-start",
        action="store_true",
        help="start the BAF stage from the planted clone labels",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="redraw the tumour spots pure (`tests.sim_fixtures.purify`) first",
    )
    parser.add_argument(
        "--decorrelated",
        action="store_true",
        help="zero the genes off baseline x copy number "
        "(`tests.sim_fixtures.decorrelate`) first, before `--pure`",
    )
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()

    sample = load_simulated(SAMPLES.get(arguments.sample, arguments.sample))

    if arguments.decorrelated:
        from tests.sim_fixtures import decorrelate

        kept = decorrelate(sample, Path(tempfile.mkdtemp()))
        sample = load_simulated(kept.name, kept.parent)

    if arguments.pure:
        from tests.sim_fixtures import purify

        pure = purify(sample, Path(tempfile.mkdtemp()))
        sample = load_simulated(pure.name, pure.parent)
    overrides = {
        k: yaml.safe_load(v)
        for k, _, v in (entry.partition("=") for entry in arguments.set)
    }
    flags = [f for f in arguments.flags if f != "--"]
    recovery, output = run_arm(
        sample, flags, overrides, arguments.root, oracle=arguments.oracle_start
    )
    recovery.peak_gb = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2
    )
    print(
        "SIM "
        + json.dumps({**asdict(recovery), "set": arguments.set, "output": str(output)})
    )


if __name__ == "__main__":
    main()
