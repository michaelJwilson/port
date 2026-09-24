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
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from port.pipeline import (
    COPY_SWAPS,
    FIGURE_SWAPS,
    NUMERIC_SWAPS,
    REFINEMENT_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
    Spent,
    instrumented,
    patched,
    warm,
)


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
        "--shift",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "fold the per-clone logmu_shift into the fit and pin the balanced, "
            "lowest-mu state to mu = 1 afterwards (#276, #293). **On by "
            "default**: without it a clone's rates come back divided by its "
            "own normalizer. Pass --no-shift for cnaster's unshifted model."
        ),
    )
    parser.add_argument(
        "--refinement-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "keep each read-depth sub-clone inside its BAF clone, with the "
            "mask cnaster computes and drops (#348); without it the ICM floor "
            "reassigns spots across BAF clones. **On by default**, off with "
            "--no-patch."
        ),
    )
    parser.add_argument(
        "--floor-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "meet the clone-size floor smallest first, each spot to its best "
            "remaining clone, at hmrf.min_spots_per_clone (#348); cnaster "
            "empties every clone under a fixed 200 at once and reassigns its "
            "spots at random. **On by default**, off with --no-patch."
        ),
    )
    parser.add_argument(
        "--distinct-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "initialize the HMM from distinct GMM components: a component "
            "within one standard deviation of a heavier one is merged into it "
            "before the most populated K are kept (#348); cnaster keeps the K "
            "most populated, which on a mostly normal genome are slices of the "
            "normal cluster. **On by default**, off with --no-patch."
        ),
    )
    parser.add_argument(
        "--copy-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "decode integer copies by the HMM's likelihood (#362) under the "
            "cap the configuration states, int_copy_num.max_total_copy (#313); "
            "cnaster's L1 decoders read no cap and decode under A + B <= 6. "
            "**On by default**, off with --no-patch."
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
        "--sal",
        action="store_true",
        help=(
            "substitute snakes_and_ladders routines where port measured a "
            "gain (#312): alpha expansion with the Rust minimum cut for the "
            "clone labelling, a lower Potts energy on every problem measured. "
            "Off by default; no row reproduces cnaster."
        ),
    )
    parser.add_argument(
        "--copy-errors",
        action="store_true",
        help=(
            "after the run, write cnv_copy_sets.tsv beside its fit: every "
            "integer (A, B) inside each state's 95 per cent credible region, "
            "from the observed information of the fitted objective (#353). "
            "**Off by default**: it differentiates the whole objective once, "
            "and it adds a file rather than changing one. Needs the shift, "
            "whose pin sets the scale (A + B) / 2 is compared on."
        ),
    )
    parser.add_argument(
        "--clone-mixture",
        action="store_true",
        help=(
            "score spots against each clone's pure path (#380): between the "
            "HMM fit and spot assignment, fit each clone's pseudobulk as a "
            "row-stochastic K x K mixture of the model clones, and assign "
            "against the unmixed paths. Off by default; unphased HMM only."
        ),
    )
    parser.add_argument(
        "--mixture-space",
        choices=("lattice", "states"),
        default="lattice",
        help=(
            "with --clone-mixture, what the pure profiles are: `lattice`, "
            "integer (A, B) pairs (a uniform admixture cannot hide in them), "
            "or `states`, the HMM's fitted continuous states (#380)"
        ),
    )
    parser.add_argument(
        "--mixture-cap",
        type=float,
        default=None,
        help=(
            "with --clone-mixture, the largest share of a clone's pseudobulk "
            "from other clones, e.g. 0.2; default 0.5 (#380)"
        ),
    )
    parser.add_argument(
        "--mixture-anneal",
        default=None,
        help=(
            "with --clone-mixture, anneal that cap linearly over each stage's "
            "outer iterations, START,END, e.g. 0.1,0.5 (#380)"
        ),
    )
    parser.add_argument(
        "--copy-decode",
        choices=("lattice", "shared"),
        default="lattice",
        help=(
            "the integer copy decode written to the tables and figures (#371): "
            "`lattice`, the default, each clone's own path over every (A, B) "
            "with its tumour fraction fitted (#370), per bin; `shared`, one "
            "pair per continuous state for every clone (#327). Both read the "
            "captured fit, so both need the copy rows."
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
        for swap in SHIFT_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, changes the model; --no-shift to omit)"
            )
        for swap in REFINEMENT_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, changes the clones; --no-refinement-mask to omit)"
            )
        for swap in COPY_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, likelihood decode, caps from the config; --no-copy-cap to omit)"
            )
        from port.extensions.sal import SAL_ROWS

        for row in SAL_ROWS:
            print(
                f"{row.cnaster} <- {row.sal}  (#{row.ticket}, {row.axis}; --sal to add)"
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
        # NB **on** unless refused, and off with `--no-patch` for the same
        #    reason the figures are: a baseline arm that fits a different
        #    model is not a baseline. The clone assignment applies the shift
        #    through `port`'s `pipeline_clone_assignment`, which is in
        #    `SWAPS`, so `--no-patch --shift` fits shifted and assigns clones
        #    unshifted; it is allowed, and said.
        shift = not arguments.no_patch if arguments.shift is None else arguments.shift

        # NB the decode compares `(A + B) / 2` against the pinned rates; an
        #    unshifted fit's rates carry the baseline's per-clone scale, so
        #    the sets would be drawn on the wrong axis (#353).
        if arguments.copy_errors and not shift:
            _parser().error("--copy-errors needs the shift; drop --no-shift")

        kept = stack.enter_context(_kept()) if arguments.copy_errors else None

        selected = SWAPS if not arguments.no_patch else ()

        # NB `--sal` selects the clone labelling through `port`'s
        #    `pipeline_clone_assignment`, a `SWAPS` row; under `--no-patch`
        #    that one row is installed alone, so the flag still means what it
        #    says and the rest of the baseline stays `cnaster`'s.
        if arguments.sal:
            from port.extensions.sal import sal

            if arguments.no_patch:
                selected = tuple(
                    swap for swap in SWAPS if swap.name == "pipeline_clone_assignment"
                )
        if approx:
            selected = selected + NUMERIC_SWAPS
        if figures:
            selected = selected + FIGURE_SWAPS
        # NB on unless refused, and off with `--no-patch` like the figures: a
        #    baseline arm decodes under `cnaster`'s caps.
        copy_cap = (
            not arguments.no_patch if arguments.copy_cap is None else arguments.copy_cap
        )
        if copy_cap:
            selected = selected + COPY_SWAPS
        refinement_mask = (
            not arguments.no_patch
            if arguments.refinement_mask is None
            else arguments.refinement_mask
        )
        if refinement_mask:
            from port.patch.hmrf.refinement import forget

            selected = selected + REFINEMENT_SWAPS
            stack.callback(forget)
        floor = (
            not arguments.no_patch
            if arguments.floor_merge is None
            else arguments.floor_merge
        )
        if floor:
            from port.patch.icm.floor import floor_merge

            stack.enter_context(floor_merge())
        distinct = (
            not arguments.no_patch
            if arguments.distinct_init is None
            else arguments.distinct_init
        )
        if distinct:
            from port.patch.hmm_initialize.distinct import distinct_init

            stack.enter_context(distinct_init())

        # NB the copy rows decode by the HMM's likelihood only (#362), which
        #    reads each clone's counts from the fit this captures; entered
        #    before `patched`, so the shift's row installs the capturing
        #    `run_core_inference`.
        if copy_cap:
            from port.extensions.copy_likelihood import capture
            from port.patch.integer_copy import copy_decoder

            stack.enter_context(capture())
            stack.enter_context(copy_decoder(arguments.copy_decode))
        if shift:
            from port.patch.hmm_nophasing import logmu_shift

            selected = selected + SHIFT_SWAPS
            stack.enter_context(logmu_shift())

        if arguments.sal:
            stack.enter_context(sal(shift=shift))

            if arguments.no_patch:
                print(
                    "run_cnaster_port: --no-patch --shift assigns clones "
                    "unshifted; the shift reaches the HMM only",
                    file=sys.stderr,
                )

        if selected:
            sites = stack.enter_context(patched(selected))
            print(
                f"run_cnaster_port: {len(selected)} replacements over "
                f"{len(sites)} bindings"
                + (", figures included" if figures else "")
                + (", copy caps from the config" if copy_cap else "")
                + (", refinement mask" if refinement_mask else "")
                + (", floor merged smallest first" if floor else "")
                + (", distinct initial states" if distinct else "")
                + (", approx included" if approx else "")
                + (", shift included" if shift else "")
                + (", sal included" if arguments.sal else ""),
                file=sys.stderr,
            )
        else:
            print("run_cnaster_port: --no-patch, nothing rebound", file=sys.stderr)

        # NB after the swaps: it wraps whatever `pipeline_clone_assignment`
        #    they bound, `port`'s or `--sal`'s (#380).
        if arguments.clone_mixture:
            from port.extensions.clone_mixture import clone_mixture

            anneal: tuple[float, float] | None = None
            if arguments.mixture_anneal:
                low, high = (float(x) for x in arguments.mixture_anneal.split(","))
                anneal = (low, high)

            stack.enter_context(clone_mixture(cap=arguments.mixture_cap, anneal=anneal))
            print(
                "run_cnaster_port: clone mixture in the loop"
                + (f", cap {arguments.mixture_cap}" if arguments.mixture_cap else "")
                + (f", annealed {arguments.mixture_anneal}" if anneal else ""),
                file=sys.stderr,
            )

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

    if kept is not None:
        _write_copy_sets(arguments.config, kept)

    print(f"run_cnaster_port: {wall:.2f}s", file=sys.stderr)
    return 0


@contextmanager
def _kept() -> Iterator[list[Any]]:
    """Keep the last `params="smp"` fit `port`'s `run_core_inference` returns.

    Entered **before** the swaps, so `patched` finds the wrapper where it
    rebinds `run_core_inference`, and the fit kept is the pinned one.
    """
    import numpy as np

    import port.patch.hmrf as patch
    from port.extensions.copy_errors import Captured

    kept: list[Any] = []
    original = patch.run_core_inference

    def keep(
        single_x: Any, lengths: Any, base: Any, total: Any, *rest: Any, **kw: Any
    ) -> Any:
        result = original(single_x, lengths, base, total, *rest, **kw)

        if kw.get("params") == "smp":
            kept.append(
                Captured(
                    np.array(single_x, dtype=np.float64),
                    np.asarray(lengths, dtype=np.int64),
                    np.array(base, dtype=np.float64),
                    np.array(total, dtype=np.float64),
                    result,
                )
            )

        return result

    patch.run_core_inference = keep

    try:
        yield kept
    finally:
        patch.run_core_inference = original


def _write_copy_sets(config: str, kept: list[Any]) -> None:
    """Write the credible sets beside the run's final fit."""
    from pathlib import Path

    import yaml

    from port.extensions.copy_errors import write_copy_sets

    if not kept:
        print("run_cnaster_port: --copy-errors kept no fit", file=sys.stderr)
        return

    output = Path(yaml.safe_load(Path(config).read_text())["paths"]["output_dir"])
    fits = sorted(
        output.rglob("rdrbaf_final_nstates*_smp.npz"), key=lambda p: p.stat().st_mtime
    )
    run = fits[-1].parent if fits else output
    path = write_copy_sets(run, kept[-1])
    print(f"run_cnaster_port: wrote {path}", file=sys.stderr)


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
