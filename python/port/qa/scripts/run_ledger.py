"""`run_ledger`: record a run into the metrics ledger, render it, or query it (#409, #620).

`run_ledger --record --note "..." [--instance dev] [--lattice] [--benchmark] [--cnaster] [--fixture NAME] -- [flags]`
runs `run_audit --recovery` (or `--sim` with `--sample`; `--cnaster` runs `port.qa.cnaster_arm` into
`docs/metrics/cnaster/`, T- #833) in its own process and appends its `runs` and `ledger` lines; inputs must
be committed unless `--dirty`. `run_ledger --best clone_ari [--fixture dev [--fixture-hash H]]`,
`run_ledger --last-benchmark`, `run_ledger --render [--out PATH]`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from port.extensions.repository import ROOT
from port.qa import ledger, provenance


def record(arguments: argparse.Namespace) -> int:
    if arguments.note is None:
        print("--record needs --note: what change this run measures")
        return 1
    ledger.check_note(arguments.note)
    dirty = provenance.dirty(*provenance.INPUTS, untracked=True)
    if dirty and not arguments.dirty:
        print("inputs are uncommitted; commit them, or pass --dirty")
        return 1

    if arguments.cnaster:
        if arguments.sample is None:
            print("--cnaster records a --sample run")
            return 1
        return record_sample(arguments, dirty=dirty)
    if arguments.sample is not None:
        return record_sample(arguments, dirty=dirty)

    fixture = ledger.fixture_name(
        arguments.instance, lattice=arguments.lattice, loh=arguments.loh
    )
    audit = [
        "--states", str(arguments.states),
        "--outer", str(arguments.outer),
        "--iterations", str(arguments.iterations),
        *(item for entry in arguments.set for item in ("--set", entry)),
    ]  # fmt: skip
    flags = [f for f in arguments.flags if f != "--"]
    command = [
        sys.executable, "-m", "port.qa.scripts.run_audit", "--recovery",
        "--instance", arguments.instance,
        *(["--lattice"] if arguments.lattice else []),
        *(["--loh"] if arguments.loh else []),
        *audit,
        "--", *flags,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith("RECOVERY ")
    ]
    if completed.returncode or not lines:
        print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
        return completed.returncode or 1

    recovery = json.loads(lines[-1].removeprefix("RECOVERY "))
    ledger.write(
        recovery,
        fixture=fixture,
        args=shlex.join([*audit, "--", *flags]),
        note=arguments.note,
        dirty=dirty,
        benchmark=arguments.benchmark,
    )
    return 0


def record_sample(arguments: argparse.Namespace, *, dirty: bool) -> int:
    """Record a run on a simulated sample via `run_audit --sim` (#467).

    `r0` must match `port.sim.fixtures.R0_HASH`; a content hash the ledger
    holds under another name is refused before the run.
    """
    from port.sim.fixtures import SAMPLES, SIM_ROOT, r0, realization_hash

    if arguments.sample == "r0":
        path = r0()
        sample = "generated/dev_tree/r0"
    else:
        sample = SAMPLES.get(arguments.sample, arguments.sample)
        path = SIM_ROOT / sample

    # NB checked before the run too, so a refused name costs no run
    digest = realization_hash(path)
    fixture = arguments.fixture or arguments.sample
    directory = ledger.CNASTER_DIR if arguments.cnaster else None
    ledger.check_identity(fixture, digest, ledger.ledger(directory))

    audit = [
        *(item for entry in arguments.set for item in ("--set", entry)),
        *(["--oracle-start"] if arguments.oracle_start else []),
    ]
    flags = [f for f in arguments.flags if f != "--"]
    command = [
        sys.executable, "-m", "port.qa.cnaster_arm" if arguments.cnaster else "port.qa.scripts.run_audit",
        "--sim", "--sample", sample,
        *audit, "--", *flags,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("SIM ")]
    if completed.returncode or not lines:
        print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
        return completed.returncode or 1

    recovery = json.loads(lines[-1].removeprefix("SIM "))
    recovery["fixture_hash"] = digest
    ledger.write(
        recovery,
        fixture=fixture,
        args=shlex.join([*audit, "--", *flags]),
        note=arguments.note,
        dirty=dirty,
        benchmark=arguments.benchmark,
        test=ledger.SIM_TEST,
        directory=directory,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_ledger", description=__doc__)
    parser.add_argument(
        "--record", action="store_true", help="run, append to runs and the ledger"
    )
    parser.add_argument(
        "--best", choices=list(ledger.METRICS), help="the line maximizing it"
    )
    parser.add_argument("--fixture", default=None,
                        help="with --best, one fixture; with --record --sample, the ledger's name for it")  # fmt: skip
    parser.add_argument("--cnaster", action="store_true",
                        help="with --record --sample, cnaster's arm (port.qa.cnaster_arm) into its own ledger, "
                             "docs/metrics/cnaster/ (T- #833)")  # fmt: skip
    parser.add_argument(
        "--fixture-hash",
        default=None,
        help="with --best --fixture, one generation where the name holds several",
    )
    parser.add_argument(
        "--render", action="store_true", help="the wide view, to stdout or --out"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="with --render, a file to write"
    )
    parser.add_argument("--dirty", action="store_true", help="record uncommitted")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="with --record, one run of a sweep over every fixture",
    )
    parser.add_argument(
        "--last-benchmark",
        action="store_true",
        help="the runs of the latest benchmark sweep, one per line",
    )
    parser.add_argument(
        "--note", default=None, help=f"with --record, <= {ledger.NOTE_CHARS} characters"
    )
    parser.add_argument(
        "--instance", default="dev", choices=["calicost", "critical", "dev"]
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="with --record, a simulated sample: r0, easy, hard or a sim/ path",
    )
    parser.add_argument(
        "--oracle-start", action="store_true", help="with --sample, the planted clones"
    )
    parser.add_argument("--lattice", action="store_true")
    parser.add_argument("--loh", action="store_true")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)

    if arguments.record:
        return record(arguments)
    if arguments.render:
        if arguments.out is None:
            sys.stdout.write(ledger.render())
        else:
            arguments.out.write_text(ledger.render())
            print(arguments.out)
        return 0
    if arguments.last_benchmark:
        for run in ledger.last_benchmark():
            print("\t".join(run[c] for c in ledger.RUN_COLUMNS))
        return 0
    if arguments.best:
        found = ledger.best(arguments.best, arguments.fixture, arguments.fixture_hash)
        print(ledger.UNMEASURED if found is None else json.dumps(found))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
