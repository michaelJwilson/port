"""`run_ledger`: record a run into the metrics ledger, render it, or query it (#409, #620).

`run_ledger --record --note "..." [--instance dev] [--lattice] -- [flags]`
runs `tests.recovery_audit` in its own process, so `peak_gb` is that run's
(`--sample easy` runs `tests.sim_audit` instead), appends the `runs` line and
one `ledger` line per measured metric under its latest definition. The note
is at most `NOTE_CHARS` characters, written as a commit subject, stating what
change the run measures.

`run_ledger --best clone_ari [--fixture dev]` prints the ledger line, joined
to its run, that maximizes a metric under its latest definition. `--render
[--out PATH]` prints the wide view, one row per run and one column per
metric, `—` where a run has no line, to stdout or to `PATH`. Nothing commits
it: the ledger is the record, and the view is generated on demand.

A run is recorded against a commit, so the inputs must be committed first
(`--dirty` records anyway and marks the commit `+`). A new metric appends a
definition and adds lines, not columns. The table API is `port.qa.ledger`;
this was `python -m tests.metrics` until T- #673 G2.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

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
        sys.executable, "-m", "tests.recovery_audit",
        "--instance", arguments.instance,
        *(["--lattice"] if arguments.lattice else []),
        *(["--loh"] if arguments.loh else []),
        *audit,
        "--", *flags,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=ledger.ROOT, capture_output=True, text=True, check=False
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
    )
    return 0


def record_sample(arguments: argparse.Namespace, *, dirty: bool) -> int:
    """A run on a simulated sample: `tests.sim_audit` in its own process (#467).

    `r0` is `dev_tree`'s realization 0, drawn if absent and refused unless it
    is the one `port.sim.fixtures.R0_HASH` names; `easy` and `hard` are CalicoST's.
    The fixture hash is the sample's content hash (`realization_hash`), and
    a name the ledger already holds under another hash is refused before the
    run (`check_identity`).
    """
    from port.sim.fixtures import SAMPLES, SIM_ROOT, r0, realization_hash

    if arguments.sample == "r0":
        path = r0()
        sample = "generated/dev_tree/r0"
    else:
        sample = SAMPLES.get(arguments.sample, arguments.sample)
        path = SIM_ROOT / sample

    # NB checked before the run as well as at the write, so a refused name
    #    costs no run
    digest = realization_hash(path)
    ledger.check_identity(arguments.sample, digest, ledger.ledger())

    audit = [
        *(item for entry in arguments.set for item in ("--set", entry)),
        *(["--oracle-start"] if arguments.oracle_start else []),
    ]
    flags = [f for f in arguments.flags if f != "--"]
    command = [
        sys.executable, "-m", "tests.sim_audit", "--sample", sample,
        *audit, "--", *flags,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=ledger.ROOT, capture_output=True, text=True, check=False
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("SIM ")]
    if completed.returncode or not lines:
        print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
        return completed.returncode or 1

    recovery = json.loads(lines[-1].removeprefix("SIM "))
    recovery["fixture_hash"] = digest
    ledger.write(
        recovery,
        fixture=arguments.sample,
        args=shlex.join([*audit, "--", *flags]),
        note=arguments.note,
        dirty=dirty,
        test=ledger.SIM_TEST,
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
    parser.add_argument("--fixture", default=None, help="with --best, one fixture")
    parser.add_argument(
        "--render", action="store_true", help="the wide view, to stdout or --out"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="with --render, a file to write"
    )
    parser.add_argument("--dirty", action="store_true", help="record uncommitted")
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
    if arguments.best:
        found = ledger.best(arguments.best, arguments.fixture)
        print(ledger.UNMEASURED if found is None else json.dumps(found))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
