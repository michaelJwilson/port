"""`run_audit`: one run of `run_cnaster_port` scored against its planted truth (T- #673 G3).

    run_audit --sim [--sample easy|hard|<name>] [--set K=V] [--oracle-start] [-- flags]
    run_audit --recovery [--instance dev] [--lattice] [--set K=V] [-- flags]
    run_audit --copy [--realizations 8] [--output PATH]
    run_audit --errors [--realizations 8] [--output PATH]

`--sim` prints one `SIM` line of JSON and the copy confusion on stderr;
`--recovery` one `RECOVERY` line, with the fixture's `fixture_hash`; `--copy`
one `COPY_AUDIT` line and a figure; `--errors` the realizations figure. A
`<name>` is under `sim/`, so a sample `port.sim.draw` wrote is
`generated/<name>/r<k>`, run on its own `config.yaml`. What each mode scores
is `port.qa.audit`'s; `run_ledger --record` runs `--recovery` or `--sim` and
appends the line it prints.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

MODES = ("sim", "recovery", "copy", "errors")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_audit", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in MODES:
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override one configuration entry; the value is read as YAML")  # fmt: skip
    # --sim
    parser.add_argument(
        "--sample", default="easy", help="--sim: easy, hard or a sim/ path"
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="--sim: the run's directory"
    )
    parser.add_argument("--oracle-start", action="store_true",
                        help="--sim: start the BAF stage from the planted clone labels")  # fmt: skip
    parser.add_argument("--pure", action="store_true",
                        help="--sim: redraw the tumour spots pure (`port.sim.fixtures.purify`) first")  # fmt: skip
    parser.add_argument("--window", default="", metavar="X0,X1,Y0,Y1",
                        help="--sim: crop to the spots in this contiguous window first (`crop`)")  # fmt: skip
    parser.add_argument("--normal-fraction", default="", metavar="F1,F2,...",
                        help="--sim with --pure: tumour clone c's spots F_c normal instead")  # fmt: skip
    parser.add_argument("--confusion-sampled", action="store_true",
                        help="--sim: print only planted rows and decoded columns of the confusion")  # fmt: skip
    # --recovery
    parser.add_argument(
        "--instance", default="dev", choices=["calicost", "critical", "dev"]
    )
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--lattice",
        action="store_true",
        help="--recovery: plant integer copies (COPY_LATTICE)",
    )
    parser.add_argument("--loh", action="store_true",
                        help="--recovery with --lattice: add mirrored LOH (port.sim.truth.LOH_STATES)")  # fmt: skip
    parser.add_argument(
        "--likelihood",
        action="store_true",
        help="--recovery: -log P(x) at the fit and the truth",
    )
    parser.add_argument("--oracle-normal", action="store_true",
                        help="--recovery: the planted normal spots as the normal candidates (upper bound)")  # fmt: skip
    parser.add_argument("--m-step-tol", type=float, default=None,
                        help="--recovery: ftol and gtol for the emission M step, which cnaster hard-codes (#30)")  # fmt: skip
    parser.add_argument("--calicost", action="store_true",
                        help="--recovery: run run_calicost on the same inputs; flags go to it (#347)")  # fmt: skip
    parser.add_argument("--diffexp", nargs=2, type=float, metavar=("FOLD", "N_GENES"),
                        help="--recovery: plant N_GENES highest-UMI genes FOLD times up in tumour spots (#440)")  # fmt: skip
    # --copy, --errors
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                        help="--errors: draws which realization carries the error bars")  # fmt: skip
    parser.add_argument(
        "--output", type=Path, default=None, help="--copy, --errors: the figure"
    )
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    return parser


def _sim(arguments: argparse.Namespace) -> None:
    from port.extensions.integer_copy import DEFAULT_MAX_TOTAL_COPY
    from port.qa.audit import audit_sample, scratch, settings
    from port.qa.scoring import confusion_table
    from port.qa.statistics import peak_gb
    from port.sim.fixtures import SAMPLES, load_simulated

    sample = load_simulated(SAMPLES.get(arguments.sample, arguments.sample))

    if arguments.window:
        from port.sim.fixtures import crop

        window = tuple(float(v) for v in arguments.window.split(","))
        if len(window) != 4:
            msg = f"--window takes X0,X1,Y0,Y1, got {arguments.window!r}"
            raise SystemExit(msg)
        cropped = crop(sample, scratch(), window)
        sample = load_simulated(cropped.name, cropped.parent)
        print(f"WINDOW {list(window)} spots={sample.barcodes.size}", flush=True)

    if arguments.pure:
        from port.sim.fixtures import purify

        normal = tuple(
            float(f) for f in arguments.normal_fraction.split(",") if f.strip()
        )
        print(f"PLANTED normal_fraction={list(normal)}", flush=True)
        pure = purify(sample, scratch(), normal=normal)
        sample = load_simulated(pure.name, pure.parent)

    flags = [f for f in arguments.flags if f != "--"]
    recovery, output = audit_sample(
        sample, flags, settings(arguments.set), arguments.root,
        oracle=arguments.oracle_start,
    )  # fmt: skip
    recovery.peak_gb = round(peak_gb(), 2)
    print(
        "SIM "
        + json.dumps({**asdict(recovery), "set": arguments.set, "output": str(output)})
    )
    print(
        confusion_table(
            recovery.confusion,
            DEFAULT_MAX_TOTAL_COPY,
            sampled=arguments.confusion_sampled,
        ),
        file=sys.stderr,
        flush=True,
    )


def _recovery(arguments: argparse.Namespace) -> None:
    from port.qa.audit import audit_truth, settings
    from port.qa.statistics import peak_gb
    from port.sim import truth as planted
    from port.sim.truth import fixture_hash

    instance = getattr(planted, f"{arguments.instance}_instance")
    truth = (
        instance(
            n_states=len(planted.COPY_LATTICE), copy_lattice=True, loh=arguments.loh
        )
        if arguments.lattice
        else instance()
    )
    flags = [f for f in arguments.flags if f != "--"]
    recovery, output = audit_truth(
        truth,
        flags,
        n_states=arguments.states,
        max_iter_outer=arguments.outer,
        max_iter=arguments.iterations,
        overrides=settings(arguments.set),
        likelihood=arguments.likelihood,
        oracle_normal=arguments.oracle_normal,
        m_step_tol=arguments.m_step_tol,
        calicost=arguments.calicost,
        diffexp=(
            None
            if arguments.diffexp is None
            else (arguments.diffexp[0], int(arguments.diffexp[1]))
        ),
    )
    recovery.peak_gb = round(peak_gb(), 2)
    print(
        "RECOVERY "
        + json.dumps(
            {
                **asdict(recovery),
                "output": str(output),
                "fixture_hash": fixture_hash(truth),
                "lattice": arguments.lattice,
                "set": arguments.set,
                "oracle_normal": arguments.oracle_normal,
                "m_step_tol": arguments.m_step_tol,
                "states": arguments.states,
                "outer": arguments.outer,
                "iterations": arguments.iterations,
                "diffexp": arguments.diffexp,
            }
        )
    )


def _copy(arguments: argparse.Namespace) -> None:
    from port.qa.audit import audit_copies, copy_summary, plot_copies
    from port.qa.provenance import PLOTS

    output = arguments.output or PLOTS / "realizations_copies.png"
    with warnings.catch_warnings(), tempfile.TemporaryDirectory() as scratch:
        warnings.simplefilter("ignore")
        scores = audit_copies(arguments.realizations, Path(scratch), arguments.jobs)

    plot_copies(scores, output)
    output.with_suffix(".json").write_text(
        json.dumps({"summary": copy_summary(scores), "realizations": scores}, indent=1)
        + "\n"
    )
    print("COPY_AUDIT " + json.dumps(copy_summary(scores)))


def _errors(arguments: argparse.Namespace) -> None:
    from port.qa.audit import audit_errors
    from port.qa.provenance import PLOTS
    from port.sim.realizations import GENOME

    audit_errors(
        arguments.realizations,
        arguments.output or PLOTS / "realizations.png",
        jobs=arguments.jobs,
        seed=GENOME["seed"] if arguments.seed is None else arguments.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mode the flags name; print its line."""
    import matplotlib as mpl

    mpl.use("Agg")
    arguments = _parser().parse_args(argv)
    mode = next(name for name in MODES if getattr(arguments, name))
    {"sim": _sim, "recovery": _recovery, "copy": _copy, "errors": _errors}[mode](
        arguments
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
