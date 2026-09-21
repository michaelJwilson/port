"""`run_cnaster`, with `port`'s replacements installed.

`run_cnaster_port config.yaml` runs `cnaster`'s own pipeline, unmodified,
with the names in `port.pipeline.SWAPS` rebound to `port`'s measured
replacements. `--no-patch` runs the same call with nothing rebound, so the
two arms of a comparison are one flag apart rather than two scripts.

**It is not a fork of the pipeline.** Nothing here reimplements a stage or
changes an order; the entry point called is `cnaster.scripts.run_cnaster`,
and if `cnaster` lands a patch upstream the row leaves `SWAPS` and this
script keeps working.

**`--figures` is on by default, so a default run does not reproduce
`cnaster` byte for byte.** It is the largest measured win here -- 47 per
cent of a run, and 8,287 MB of figure rendering down to 1,036 MB (#195) --
and a figure written at a different dpi is a different file by design. Pass
`--no-figures` for an arm that does reproduce bitwise.

That property still holds of `port.pipeline.SWAPS`, which is unchanged and
still what `install()` defaults to; only this entry point's default moved.
So what `tests/test_patched_entry_point.py` asserts is what it always
asserted -- every row of `SWAPS` reproducing `cnaster` -- and the table that
does not make that claim is still the separate one.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack

from port.pipeline import (
    FIGURE_SWAPS,
    NUMERIC_SWAPS,
    SWAPS,
    Spent,
    instrumented,
    patched,
    warm,
)
from port.tracking import tracking


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_cnaster_port",
        description="Run cnaster's pipeline with port's replacements installed.",
    )
    parser.add_argument(
        "config", nargs="?", help="the YAML configuration run_cnaster reads"
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="run the same pipeline with nothing rebound, for the baseline arm",
    )
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "install the replacements that change the output: the figure dpi "
            "and the rasterizing groups (#195). **On by default**, because it "
            "is the largest measured win port has -- 47 per cent of a run, and "
            "8,287 MB of figure rendering down to 1,036 MB. Pass --no-figures "
            "for an arm that reproduces cnaster bitwise, which every other "
            "swap does and this one does not."
        ),
    )
    parser.add_argument(
        "--approx",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "install the replacements that agree to a tolerance rather than "
            "bitwise: the vectorized negative-binomial log-pmf (#240). **Off "
            "by default**: it is 1.78x on the kernel and nothing on a whole "
            "run, and the 8.6e-13 disagreement moves one segment's integer "
            "copy number by 3 (#244). Available for measuring that, not for "
            "running production with."
        ),
    )
    parser.add_argument(
        "--warm-up",
        action="store_true",
        help=(
            "compile every kernel before the clock starts, so the timed run "
            "measures the code and not the compiler (#211). Off by default: a "
            "run whose first call really is the cost should be able to see it."
        ),
    )
    parser.add_argument(
        "--track",
        metavar="REPO",
        default=None,
        help=(
            "record every instrumented stage into an Aim run at REPO (#251): "
            "the objective, the iteration count and the clone count per visit, "
            "with the stage and solver as Aim context. **Off by default and it "
            "changes no number** -- outside a track block `record` returns on "
            "its first line. Needs the `track` extra; the run refuses rather "
            "than quietly recording nothing if `aim` is absent."
        ),
    )
    parser.add_argument(
        "--time-stages",
        action="store_true",
        help="report what the swapped names cost in this run, patched or not",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what would be rebound, and exit without running",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline, patched unless `--no-patch` is given."""
    arguments = _parser().parse_args(argv)

    # NB imported before the swaps are applied, so that the rebinding finds
    #    the entry point's own `from cnaster.omics import ...` bindings. The
    #    import is here rather than at module scope because `--list` and
    #    `--help` should not pay for it.
    import cnaster.scripts.run_cnaster as pipeline

    if arguments.list:
        for swap in SWAPS:
            print(f"{swap.module}.{swap.name} <- {swap.replacement}  (#{swap.ticket})")
        for swap in NUMERIC_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, agrees to a tolerance; --no-approx to omit)"
            )
        for swap in FIGURE_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, changes the output; --no-figures to omit)"
            )
        return 0

    if arguments.config is None:
        _parser().error("a configuration is required unless --list is given")

    with ExitStack() as stack:
        # NB `--figures` is additive rather than a third mode, and it composes
        #    with `--no-patch`: what a reader needs to know about a run is
        #    which of the two tables produced it, not which flag was typed.
        #
        #    It is **on by default** and the two tables stay separate, which
        #    is the whole point: `SWAPS` is still the set that reproduces
        #    `cnaster` bitwise, so the tests that assert that property keep
        #    asserting it, while a user who just runs the entry point gets
        #    the 47 per cent. Turning it on by merging the tables would have
        #    bought the same speed and cost the claim.
        # NB `--no-patch` means nothing rebound, so it turns the figure
        #    default off with it. Without this the baseline arm still
        #    installs `write_fig` and stops being a baseline -- measured, and
        #    not subtly: the unpatched arm ran 188.88 s at 11.35 GB before
        #    the default moved and 142.36 s at 3.67 GB after, which reads as
        #    the patch doing less good rather than the baseline getting the
        #    win for free.
        #
        #    `default=None` is what makes that possible: it separates "not
        #    asked" from "asked for off", so `--no-patch --figures` still
        #    composes and still measures the figure swap on its own.
        figures = (
            not arguments.no_patch if arguments.figures is None else arguments.figures
        )
        # NB **off** unless asked for. Measured on a whole run at
        #    4,000 x 1,980 x 5: it recovers -1.04 s and -0.051 GB -- nothing,
        #    within noise -- while moving one segment's integer copy number by
        #    3 (#244). The 1.78x kernel ratio does not survive `CountEncoder`
        #    dedup, which is what #240 warned it might not.
        approx = bool(arguments.approx)

        selected = SWAPS if not arguments.no_patch else ()
        if approx:
            selected = selected + NUMERIC_SWAPS
        if figures:
            selected = selected + FIGURE_SWAPS

        if selected:
            sites = stack.enter_context(patched(selected))
            print(
                f"run_cnaster_port: {len(selected)} replacements over "
                f"{len(sites)} bindings"
                + (", figures included" if figures else "")
                + (", approx included" if approx else ""),
                file=sys.stderr,
            )
        else:
            print("run_cnaster_port: --no-patch, nothing rebound", file=sys.stderr)

        # NB after the swaps, so a recorded run records whichever
        #    implementation is installed, and before the timer, because
        #    opening the store is not part of what is being measured. The
        #    store is stamped with what a reader needs to tell two runs
        #    apart; `aim` is imported inside `tracking`, never here.
        if arguments.track is not None:
            run = stack.enter_context(tracking(arguments.track))
            run["config"] = str(arguments.config)
            run["swaps"] = len(selected)
            run["figures"] = figures
            run["approx"] = approx
            run["patched"] = not arguments.no_patch
            print(f"run_cnaster_port: tracking to {arguments.track}", file=sys.stderr)

        # NB after the swaps and before the timer, so what is compiled is
        #    what the run will call and none of it lands in the measurement.
        if arguments.warm_up:
            warmed = warm()
            print(f"run_cnaster_port: {warmed.report()}", file=sys.stderr)

        # NB after the swaps, so the wrapper times whichever implementation
        #    the run is about to use.
        # NB the figure swap is timed whether or not it is installed, so the
        #    two arms print the same rows and `write_fig` can be compared
        #    against itself rather than inferred from the whole-run delta.
        spent = (
            stack.enter_context(instrumented(SWAPS + NUMERIC_SWAPS + FIGURE_SWAPS))
            if arguments.time_stages
            else None
        )

        started = time.perf_counter()
        pipeline.run_cnaster(arguments.config)
        wall = time.perf_counter() - started

    if spent is not None:
        _report(spent, wall, patched=not arguments.no_patch)

    print(f"run_cnaster_port: {wall:.2f}s", file=sys.stderr)
    return 0


def _report(spent: dict[str, Spent], wall: float, *, patched: bool) -> None:
    """What the swapped names cost, against the run that contained them."""
    arm = "patched" if patched else "baseline"
    width = max(len(name) for name in spent)
    total = sum(entry.seconds for entry in spent.values())

    print(f"\n  {arm}: {wall:.2f}s whole run", file=sys.stderr)
    print(
        f"  {'stage':<{width}} {'calls':>6} {'seconds':>9} {'share':>7}"
        f" {'first':>9} {'warm/call':>10}",
        file=sys.stderr,
    )

    # NB the last two columns are #204: a compiled kernel's first call is
    #    compilation, and summed into the total it reads as a kernel that is
    #    eighty times slower than it is.
    for name, entry in sorted(spent.items(), key=lambda item: -item[1].seconds):
        share = 100.0 * entry.seconds / wall
        per_warm = entry.warm / entry.warm_calls if entry.warm_calls else float("nan")
        print(
            f"  {name:<{width}} {entry.calls:6d} {entry.seconds:9.3f} {share:6.2f}%"
            f" {entry.first:9.3f} {per_warm:10.3f}",
            file=sys.stderr,
        )

    first = sum(entry.first for entry in spent.values())
    print(
        f"  {'TOTAL':<{width}} {'':>6} {total:9.3f} {100.0 * total / wall:6.2f}%"
        f" {first:9.3f}",
        file=sys.stderr,
    )
    print(
        f"  of which first calls: {first:.3f}s "
        f"({100.0 * first / total if total else 0.0:.1f}% of the swapped stages, "
        f"{100.0 * first / wall:.1f}% of the run)",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
