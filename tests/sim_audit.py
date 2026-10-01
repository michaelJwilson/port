"""`run_cnaster_port` on CalicoST's simulated samples, scored (#362).

Run as `python -m tests.sim_audit [--sample easy|hard|<name>] [--set k=v]
[-- flags]`: one arm, one `SIM` line of JSON on stdout. `<name>` is under
`sim/`, so a sample `port.sim.draw` wrote is `generated/<name>/r<k>` (#445); it
runs on its own `config.yaml`.

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

The same two shares are also taken over three classes of planted pair:
`loh`, one haplotype at 0 (deletions, copy-neutral and amplified LOH);
`balanced_gain`, `A = B > 1`; and `unbalanced_gain`, both haplotypes present,
`A + B > 2` and `A != B`. Each `_pf` form is phase-free, as
`exact_altered_minor` is; for a balanced gain the two coincide. A class the
sample does not plant scores NaN. `exact_neutral` is the share of planted
`(1, 1)` bins decoded `(1, 1)`.

`copy_ari_<class>` is the copy-state ARI restricted to the clone-bins of one
planted class (#511), so a class is scored on how it partitions its own bins
rather than on the pairs it shares with the ~6,500 neutral bins, which
dominate `copy_ari`. ARI over bins whose planted pair takes one value is
undefined (sklearn returns 1 if the decode is constant there, else 0), so a
class planted as a single pair scores NaN, as does one not planted. Neutral
is one pair by definition and has no ARI; `exact_neutral` stands in for it.
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
from port.sim.files import located

from tests.recovery_audit import integer_clones
from tests.scoring import matched, overlap
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
    copy_ari_loh: float
    copy_ari_balanced_gain: float
    copy_ari_unbalanced_gain: float
    n_clones: int
    n_integer_clones: int
    exact: float
    exact_altered: float
    exact_altered_minor: float
    exact_loh: float
    exact_loh_pf: float
    exact_balanced_gain: float
    exact_balanced_gain_pf: float
    exact_unbalanced_gain: float
    exact_unbalanced_gain_pf: float
    exact_neutral: float
    bins: int
    clone_of: dict[int, int] = field(default_factory=dict)


def _scratch() -> Path:
    """Where generated samples go: `$PORT_SIM_CACHE`, reused when complete, else a temp dir."""
    import os

    cache = os.environ.get("PORT_SIM_CACHE")
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        return Path(cache)
    return Path(tempfile.mkdtemp())


def _barcode(values: pd.Series) -> np.ndarray:
    barcodes: np.ndarray = values.astype(str).to_numpy()
    return barcodes


NEUTRAL = 1_001
"""The planted pair `(1, 1)`, as `A * 1_000 + B`."""


def planted_classes(t: np.ndarray) -> dict[str, np.ndarray]:
    """Clone-bins by planted class, from pairs coded `A * 1_000 + B`.

    `loh` one haplotype at 0; `balanced_gain` `A = B > 1`; `unbalanced_gain`
    both present, `A + B > 2`, `A != B`; `neutral` `(1, 1)`. One partition
    for every per-class metric in `score`.
    """
    major, minor = t // 1_000, t % 1_000
    loh = np.minimum(major, minor) == 0
    gain = (major + minor > 2) & ~loh
    return {
        "loh": loh,
        "balanced_gain": gain & (major == minor),
        "unbalanced_gain": gain & (major != minor),
        "neutral": t == NEUTRAL,
    }


def class_ari(t: np.ndarray, ab: np.ndarray, where: np.ndarray) -> float:
    """Copy-state ARI over the bins in `where`; NaN where the planted pairs
    there take fewer than two values, ARI being undefined.
    """
    from sklearn.metrics import adjusted_rand_score

    if np.unique(t[where]).size < 2:
        return float("nan")
    return round(float(adjusted_rand_score(t[where], ab[where])), 4)


def read_run(sample: SimulatedSample, output: Path) -> dict[str, Any]:
    """Fitted labels per truth spot, and per-bin `Z`, `A`, `B` per fitted clone."""
    run = next(output.rglob("rdrbaf_final_nstates*_smp.npz"))
    fit = np.load(run, allow_pickle=True)
    table = pd.read_csv(run.parent / "clone_labels.tsv", sep="\t", comment="#")
    # NB CalicoST writes the barcodes as an index named `BARCODES` (#494);
    #    `cnaster` as a `barcode` column.
    barcodes = table["barcode"] if "barcode" in table else table.iloc[:, 0]
    by_barcode = dict(
        zip(_barcode(barcodes), table["clone_label"].to_numpy(), strict=True)
    )
    labels = np.array([by_barcode.get(b, -1) for b in sample.barcodes])

    seglevel = pd.read_csv(run.parent / "cnv_seglevel.tsv", sep="\t")
    n_fitted = int(labels.max()) + 1
    n_states = np.asarray(fit["new_log_mu"]).shape[0]
    pred = np.asarray(fit["pred_cnv"]).reshape(len(seglevel), -1) % n_states
    # NB CalicoST leaves out the column of a clone whose integer fit it
    #    skipped (#494); its bins read as -1, never as a planted pair.
    missing = np.full(len(seglevel), -1)
    a = np.stack(
        [seglevel.get(f"clone{c} A", missing) for c in range(n_fitted)], 1
    ).astype(np.int64)
    b = np.stack(
        [seglevel.get(f"clone{c} B", missing) for c in range(n_fitted)], 1
    ).astype(np.int64)

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
    integer = merged[fitted[scored]]
    written = next(output.rglob("clone_labels_integer.tsv"), None)

    if written is not None:
        # NB the run's own integer clones, under the merge agreement its
        #    configuration states (#518); the exact rule where none is written.
        table = pd.read_csv(written, sep="\t", comment="#")
        by_barcode = dict(
            zip(
                _barcode(table["barcode"]),
                table["integer_clone_label"].to_numpy(),
                strict=True,
            )
        )
        integer = np.array([by_barcode[b] for b in sample.barcodes[scored]])

    ari_integer = float(adjusted_rand_score(sample.labels[scored], integer))

    clone_of = matched(
        overlap(
            sample.labels[scored],
            fitted[scored],
            sample.n_clones,
            int(fitted.max()) + 1,
        )
    )

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
    altered = t != NEUTRAL
    swapped = (ab % 1_000) * 1_000 + ab // 1_000
    either = (t == ab) | (t == swapped)
    classes = planted_classes(t)
    loh, balanced, unbalanced = (
        classes[c] for c in ("loh", "balanced_gain", "unbalanced_gain")
    )

    def share(hit: np.ndarray, where: np.ndarray) -> float:
        """`hit`'s share over `where`; NaN where the sample plants none."""
        return round(float(np.mean(hit[where])), 4) if where.any() else float("nan")

    return SimRecovery(
        sample=sample.name,
        arm=arm,
        wall=round(wall, 2),
        peak_gb=0.0,
        ari=round(ari, 4),
        ari_integer=round(ari_integer, 4),
        state_ari=round(float(adjusted_rand_score(t, z)), 4),
        copy_ari=round(float(adjusted_rand_score(t, ab)), 4),
        copy_ari_loh=class_ari(t, ab, loh),
        copy_ari_balanced_gain=class_ari(t, ab, balanced),
        copy_ari_unbalanced_gain=class_ari(t, ab, unbalanced),
        n_clones=int(np.unique(fitted[scored]).size),
        n_integer_clones=int(np.unique(integer).size),
        exact=round(float(np.mean(t == ab)), 4),
        exact_altered=round(float(np.mean((t == ab)[altered])), 4),
        exact_altered_minor=round(float(np.mean(either[altered])), 4),
        exact_loh=share(t == ab, loh),
        exact_loh_pf=share(either, loh),
        exact_balanced_gain=share(t == ab, balanced),
        exact_balanced_gain_pf=share(either, balanced),
        exact_unbalanced_gain=share(t == ab, unbalanced),
        exact_unbalanced_gain_pf=share(either, unbalanced),
        exact_neutral=share(t == ab, classes["neutral"]),
        bins=int(covered.sum()),
        clone_of=clone_of,
    )


def _drawn_config(
    sample: SimulatedSample, root: Path, overrides: dict[str, Any]
) -> Path:
    """A `port.sim.draw` sample's own `config.yaml`, writing under `root` (#445)."""
    document: dict[str, Any] = yaml.safe_load((sample.path / "config.yaml").read_text())
    document["paths"]["output_dir"] = str(root / "output")
    document["paths"]["perf_path"] = str(root / "cnaster.perf")

    for key, value in overrides.items():
        section, _, name = key.partition(".")
        document[section][name] = value

    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(document))
    return config


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
        "annotation.clone_label": str(located(sample.path / "truth_clone_labels.tsv")),
        "hmrf.fixed_assignment": True,
    }
    settings = {**(overrides or {}), **(known if oracle else {})}
    if (sample.path / "snp").is_dir():
        config = _drawn_config(sample, root, settings)
    else:
        config = write_sim_inputs(sample, root, settings)

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
        "--window",
        default="",
        metavar="X0,X1,Y0,Y1",
        help="crop to the spots in this contiguous window first (`crop`)",
    )
    parser.add_argument(
        "--normal-fraction",
        default="",
        metavar="F1,F2,...",
        help="with --pure, tumour clone c's spots F_c normal instead",
    )
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()

    sample = load_simulated(SAMPLES.get(arguments.sample, arguments.sample))

    if arguments.window:
        from tests.sim_fixtures import crop

        window = tuple(float(v) for v in arguments.window.split(","))
        assert len(window) == 4
        cropped = crop(sample, _scratch(), window)
        sample = load_simulated(cropped.name, cropped.parent)
        print(f"WINDOW {list(window)} spots={sample.barcodes.size}", flush=True)

    if arguments.pure:
        from tests.sim_fixtures import purify

        normal = tuple(
            float(f) for f in arguments.normal_fraction.split(",") if f.strip()
        )
        print(f"PLANTED normal_fraction={list(normal)}", flush=True)
        pure = purify(sample, _scratch(), normal=normal)
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
