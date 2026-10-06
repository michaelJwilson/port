"""#540: copy-state starts for the BAF, then the BAF + RDR, HMM at oracle clones.

Subcommands of `python -m tests.studies.copy_starts`:

`capture SAMPLE OUT.npz`
    One `--sal --oracle-start --no-plots` arm on the sample, recording each
    `run_core_inference` call's inputs (`port.patch.hmrf.core_inference`'s
    `UPSTREAM`, read at call time). Each stage's initializer call is then
    rebuilt at the planted clones, pooled and stacked as `cnaster` pools
    them (`merge_pseudobulk_by_index_mix`, `clone_stack_obs`), so both
    stages start from oracle clones rather than the BAF stage's output.
    Written with each row's clone, bin, segment position and planted
    `(A, B)`: `copy_starts.write_captured`.
`run CAPTURE OUT.pkl [--arm ARM ...]`
    The arms of #540 on the captured calls through
    `port.sandbox.extensions.copy_starts`, each start polished by `sal`'s EM under
    `BUDGET_SECONDS`, `SEEDS`, `WORKERS`.

`docs/nb/copy_state_starts.ipynb` reads what `run` writes.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = [0, 1, 2]
WORKERS = 4
BUDGET_SECONDS = 60.0
"""The start and its polish, per trial: #489 gave 160 s; its starts stopped inside 60 s but for `hmc`."""


def capture(
    sample_name: str, out: Path, overrides: dict[str, Any] | None = None
) -> None:
    """Run `--sal --oracle-start` once, and rebuild each stage's initializer call at the planted clones.

    `overrides` sets configuration keys, `section.key` to a value, as
    `tests.sim_audit --set` does: the #551 segment floor, for one.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    from cnaster.hmrf_utils import clone_stack_obs
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
    from port.extensions import segments
    from port.patch.hmrf import core_inference
    from port.sandbox.extensions.copy_starts import write_captured
    from port.sim.fixtures import load_simulated

    from tests.sim_audit import run_arm

    path = Path(sample_name)
    sample = (
        load_simulated(path.name, path.parent)
        if path.is_absolute()
        else load_simulated(sample_name)
    )
    real = core_inference.UPSTREAM
    calls: list[dict[str, Any]] = []

    def capturing(**arguments: Any) -> Any:
        kept = (
            "single_X",
            "lengths",
            "single_base_nb_mean",
            "single_total_bb_RD",
            "single_tumor_prop",
            "initial_clone_index",
            "n_states",
            "log_sitewise_transmat",
            "params",
        )
        calls.append({k: arguments.get(k) for k in kept})
        return real(**arguments)

    core_inference.UPSTREAM = capturing
    try:
        with segments.recording() as lineage:
            _, output = run_arm(sample, ["--sal", "--no-plots"], overrides, oracle=True)
    finally:
        core_inference.UPSTREAM = real

    # NB the first call of each stage: BAF only (`params` without `m`), then
    #    BAF + RDR. Both at the BAF stage's initial clones, which
    #    `--oracle-start` sets to the planted labels.
    stages: dict[str, dict[str, Any]] = {}
    for call in calls:
        stage = "rdrbaf" if "m" in str(call["params"]) else "baf"
        stages.setdefault(stage, call)
    oracle = [np.asarray(i) for i in stages["baf"]["initial_clone_index"]]

    captured: dict[str, Any] = {}
    for stage, call in stages.items():
        single_X = np.asarray(call["single_X"])
        if single_X.shape[2] != sum(i.size for i in oracle):
            msg = f"{stage}: {single_X.shape[2]} spots against {sum(i.size for i in oracle)} planted"
            raise ValueError(msg)
        X, base, trials, tumor = merge_pseudobulk_by_index_mix(
            single_X,
            call["single_base_nb_mean"],
            call["single_total_bb_RD"],
            oracle,
            call["single_tumor_prop"],
        )
        stacked = clone_stack_obs(
            X, base, trials, call["lengths"], call["log_sitewise_transmat"], tumor
        )
        level = next(
            s
            for s in reversed(list(lineage.levels.values()))
            if s.n_segments == X.shape[0]
        )
        captured[stage] = {
            "X": stacked[0],
            "base_nb_mean": stacked[1],
            "total_bb_RD": stacked[2],
            "lengths": stacked[3],
            "log_sitewise_transmat": stacked[4],
            "n_states": int(call["n_states"]),
            "params": str(call["params"]),
            "n_clones": len(oracle),
            "contig": np.asarray(level.contig),
            "start": np.asarray(level.start),
            "length": np.asarray(level.length),
            "gene": np.asarray(level.gene).astype(str),
            # NB the run's configuration: `cnaster`'s initializers read its
            #    `hmm` section from the global configuration.
            "config": (output.parent / "config.yaml").read_text(),
        }

    # NB each clone column is a planted clone: `--oracle-start` orders the
    #    initial clones as the sample's labels.
    for held in captured.values():
        middle = held["start"] + held["length"] // 2
        chromosome = np.char.replace(held["contig"].astype(str), "chr", "")
        planted = sample.copies_at(chromosome, middle)
        held["planted"] = planted[:, : held["n_clones"], :]

    write_captured(out, captured, sample=str(sample.path))


def main(argv: list[str] | None = None) -> None:
    """The subcommands."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("capture")
    one.add_argument("sample")
    one.add_argument("out", type=Path)
    one.add_argument("--set", action="append", default=[], metavar="K=V")
    two = sub.add_parser("run")
    two.add_argument("capture", type=Path)
    two.add_argument("out", type=Path)
    two.add_argument("--arm", action="append", default=None)
    two.add_argument(
        "--only", action="append", default=None, help="these starts' jobs alone"
    )
    two.add_argument(
        "--merge", type=Path, default=None, help="an earlier run whose rows to keep"
    )
    arguments = parser.parse_args(argv)

    if arguments.command == "capture":
        import yaml

        overrides = {
            k: yaml.safe_load(v)
            for k, _, v in (entry.partition("=") for entry in arguments.set)
        }
        capture(arguments.sample, arguments.out, overrides)
    else:
        from port.sandbox.extensions.copy_starts import read_captured

        from tests.studies.copy_start_arms import ARMS, run_arms

        opened = time.perf_counter()
        results = run_arms(
            read_captured(arguments.capture),
            arguments.arm or list(ARMS),
            seeds=SEEDS,
            workers=WORKERS,
            seconds=BUDGET_SECONDS,
            only=tuple(arguments.only) if arguments.only else None,
        )
        if arguments.merge is not None:
            with arguments.merge.open("rb") as fh:
                earlier = pickle.load(fh)
            added = {
                (r["arm"], r["variant"], r["stage"], r["start"], r["seed"])
                for r in results["rows"]
            }
            results["rows"] = [
                r
                for r in earlier["rows"]
                if (r["arm"], r["variant"], r["stage"], r["start"], r["seed"])
                not in added
            ] + results["rows"]
        with arguments.out.open("wb") as fh:
            pickle.dump(results, fh, protocol=5)
        print(f"{time.perf_counter() - opened:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
