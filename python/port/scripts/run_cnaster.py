"""`run_cnaster`, with `port`'s replacements installed.

`run_cnaster_port config.yaml` runs `cnaster`'s own pipeline, unmodified,
with the names in `port.pipeline.SWAPS` rebound to `port`'s measured
replacements. `--no-patch` runs the same call with nothing rebound, so the
two arms of a comparison are one flag apart rather than two scripts.

**It is not a fork of the pipeline.** Nothing here reimplements a stage or
changes an order; the entry point called is `cnaster.scripts.run_cnaster`,
and if `cnaster` lands a patch upstream the row leaves `SWAPS` and this
script keeps working.

**`--figure-swaps` is on by default, so a default run does not reproduce
`cnaster` byte for byte.** It is the largest measured win here -- 47 per
cent of a run, and 8,287 MB of figure rendering down to 1,036 MB (#195) --
and a figure written at a different dpi is a different file by design.
`--no-figure-swaps` alone does not give a bitwise arm: the shift and copy-cap
tables are on by default too and change the fit (#466). `--no-patch` does.

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
    PLOT_OFF_SWAPS,
    REFINEMENT_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
    Spent,
    instrumented,
    patched,
    warm,
    with_options,
)


def _layout(text: str) -> tuple[int, int]:
    """`"3,1"` as `(3, 1)`, both positive."""
    try:
        rows, columns = (int(part) for part in text.split(","))
    except ValueError as error:
        msg = f"expected ROWS,COLUMNS, got {text!r}"
        raise argparse.ArgumentTypeError(msg) from error

    if rows < 1 or columns < 1:
        msg = f"a layout needs a row and a column, got {text!r}"
        raise argparse.ArgumentTypeError(msg)

    return rows, columns


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_cnaster_port",
        description="Run cnaster's pipeline with port's replacements installed.",
    )
    parser.add_argument(
        "config", nargs="?", help="the YAML configuration run_cnaster reads"
    )
    parser.add_argument(
        "--no-outputs",
        action="store_true",
        help=(
            "skip port.extensions.outputs, which writes the fitted states, the "
            "integer segments, the bin-level posterior means and a manifest "
            "beside cnaster's files (#331); off with --no-patch"
        ),
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="run the same pipeline with nothing rebound, for the baseline arm",
    )
    parser.add_argument(
        "--figure-swaps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "install the figure replacements, which change the files: the "
            "figure dpi and rasterizing groups (#195), the genomic RDR line "
            "(#299), the spatial tiles and the copy-number profile (#309). "
            "**On by default**, because it is the largest measured win port "
            "has -- 47 per cent of a run, and 8,287 MB of figure rendering "
            "down to 1,036 MB. Off with --no-figure-swaps or --no-patch; only "
            "--no-patch also leaves out the shift and copy-cap tables, which "
            "change the fit."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help=(
            "build every figure and write none (#403): the plotting code runs, "
            "its rendering does not. For a run whose claim is not a figure. "
            "Not --no-figure-swaps, which writes cnaster's figures unswapped."
        ),
    )
    parser.add_argument(
        "--shift",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "fold the per-clone logmu_shift into the fit and pin the normal "
            "clone's dominant balanced state to mu = 1 afterwards (#276, #299). "
            "**On by "
            "default**: without it a clone's rates come back divided by its "
            "own normalizer. Pass --no-shift for cnaster's unshifted model, "
            "which also leaves out the sal emission and distinct init."
        ),
    )
    parser.add_argument(
        "--refinement-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "keep each read-depth sub-clone inside its BAF clone, with the "
            "mask cnaster computes and drops (#348), as a 100-nat penalty "
            "(#467); without it the ICM floor reassigns spots across BAF "
            "clones. Off by default, on with --sal: on #338's three-sample "
            "instance the default arm splits 2 planted clones into 6."
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
            "spots at random. Off by default, on with --sal (#467): on #338's "
            "three-sample instance the default arm splits 2 planted clones into 6."
        ),
    )
    parser.add_argument(
        "--hmm-start",
        default=None,
        metavar="START",
        help=(
            "the read-depth + BAF stage's HMM start from sal's count-pair "
            "mixture, a key of sal.search.mixture_starts (#489), conditioned "
            "on each bin's exposure and trials; 'none' keeps --distinct-init's. "
            "Off by default; kmeans++x5+em with --sal."
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
            "normal cluster. **On by default**, off with --no-patch or "
            "--no-shift, whose run_core_inference row is what reads it."
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
        "--png-copies",
        action="store_true",
        help=(
            "write a PNG without metadata beside each PDF, so two runs of the "
            "same code write the same bytes (docs/plots, #452). Needs "
            "--figure-swaps."
        ),
    )
    parser.add_argument(
        "--sample-layout",
        type=_layout,
        default=None,
        metavar="ROWS,COLUMNS",
        help=(
            "draw the clone spatial plots one panel per sample on this grid, "
            "e.g. 3,1, each sample in its own coordinates (#328). Unset, "
            "cnaster's one axis with samples offset along x. Needs --figure-swaps."
        ),
    )
    parser.add_argument(
        "--genomic-colours",
        choices=("integer", "states"),
        default=None,
        help=(
            "colour the clones_genomic bins by deduplicated integer copies "
            "(A, B), or by fitted HMM state with each state's continuous "
            "2 mu and p, so states oversampling one integer pair stay "
            "distinct. Unset, cnaster's choice per figure. Needs --figure-swaps."
        ),
    )
    parser.add_argument(
        "--rust",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "run cnaster's four forward/backward lattices from port's Rust "
            "backend, oxiport (#318): bitwise cnaster's, compiled once at "
            "build rather than by numba in every process. **On by default**, "
            "off with --no-patch; --no-patch --rust adds it alone."
        ),
    )
    parser.add_argument(
        "--sal-emission",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "score the coded NB/BB emission with sal's dense log-emission "
            "(#425): to 3.2e-12 of cnaster's kernels (3.5e-9 at the dispersion "
            "floor, where sal is the nearer the exact value), 3-9x the kernels "
            "at 100,000 codes, and no faster end to end, as CountEncoder dedup "
            "leaves the kernels small. **On by default** where the shift is, "
            "off with --no-sal-emission or --no-shift; not a --sal row."
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
        "--audit-config",
        action="store_true",
        help=(
            "print what the configuration states that cnaster does not use -- "
            "keys nothing reads, thresholds that cannot fire, floors that do "
            "not govern (#324) -- and exit without running"
        ),
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
        for swap in FIGURE_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, changes the output; --no-figure-swaps to omit)"
            )
        for swap in SHIFT_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, changes the model; --no-shift to omit)"
            )
        for swap in REFINEMENT_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, changes the clones; --refinement-mask to install)"
            )
        for swap in COPY_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, likelihood decode, caps from the config; --no-copy-cap to omit)"
            )
        from port.patch.lattice import RUST_LATTICES

        for module, cls in RUST_LATTICES:
            print(
                f"{module}.{cls}.{{forward,backward}}_lattice <- port.oxiport  "
                "(#318, bitwise; --no-rust to omit)"
            )
        from port.extensions.sal import SAL_ROWS

        for row in SAL_ROWS:
            print(
                f"{row.cnaster} <- {row.sal}  (#{row.ticket}, {row.axis}; --sal to add)"
            )
        return 0

    if arguments.config is None:
        _parser().error("a configuration is required unless --list is given")

    # NB refused before the configuration is read: an argument error, not a
    #    file error. The figure default is the one the run below computes.
    if arguments.genomic_colours is not None and not (
        not arguments.no_patch
        if arguments.figure_swaps is None
        else arguments.figure_swaps
    ):
        _parser().error("--genomic-colours needs the figure swaps")

    # NB **on** unless refused, and off with `--no-patch` for the same
    #    reason the figures are: a baseline arm that fits a different
    #    model is not a baseline. The clone assignment applies the shift
    #    through `port`'s `pipeline_clone_assignment`, which is in
    #    `SWAPS`, so `--no-patch --shift` fits shifted and assigns clones
    #    unshifted; it is allowed, and said.
    shift = not arguments.no_patch if arguments.shift is None else arguments.shift

    # NB the decode compares `(A + B) / 2` against the pinned rates; an
    #    unshifted fit's rates carry the baseline's per-clone scale, so
    #    the sets would be drawn on the wrong axis (#353). Refused before
    #    the config is read.
    if arguments.copy_errors and not shift:
        _parser().error("--copy-errors needs the shift; drop --no-shift")

    import yaml

    from port.extensions.config_audit import audit

    findings = audit(yaml.safe_load(open(arguments.config)))  # noqa: PTH123, SIM115

    if arguments.audit_config:
        for finding in findings:
            print(finding)
        return 0

    print(
        f"run_cnaster_port: config audit: {len(findings)} findings"
        + (" (--audit-config to list)" if findings else ""),
        file=sys.stderr,
    )

    with ExitStack() as stack:
        # NB `--figure-swaps` is additive rather than a third mode, and it composes
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
        #    asked" from "asked for off", so `--no-patch --figure-swaps` still
        #    composes and still measures the figure swap on its own.
        figures = (
            not arguments.no_patch
            if arguments.figure_swaps is None
            else arguments.figure_swaps
        )
        # NB bitwise, so on by default like `SWAPS`, and off with it: a
        #    baseline arm is `cnaster`'s compiled code as well as its names.
        rust = not arguments.no_patch if arguments.rust is None else arguments.rust

        if rust:
            from port.patch.lattice import rust_lattices

            stack.enter_context(rust_lattices())

        kept = stack.enter_context(_kept()) if arguments.copy_errors else None

        selected = SWAPS if not arguments.no_patch else ()

        # NB `--sal` selects the clone labelling through `port`'s
        #    `pipeline_clone_assignment`, a `SWAPS` row; under `--no-patch`
        #    that one row is installed alone, so the flag still means what it
        #    says and the rest of the baseline stays `cnaster`'s.
        # NB the sal emission and the distinct initializer are read by the
        #    `SHIFT_SWAPS` rows alone -- port's `hmm_nophasing` class and
        #    `run_core_inference` -- so without the shift they are off, and
        #    asking for either is refused rather than ignored (#466).
        for flag, asked in (
            ("--sal-emission", arguments.sal_emission),
            ("--distinct-init", arguments.distinct_init),
        ):
            if asked and not shift:
                _parser().error(f"{flag} is read by the shift rows; --no-shift")
        sal_emission_on = (
            shift if arguments.sal_emission is None else arguments.sal_emission
        )
        # NB the coded emission from sal's tables (#425), an option of the
        #    `hmm_nophasing` row, bound once the shift's table is selected.
        #    That row is in `SHIFT_SWAPS`, so `--no-patch --shift` reads it.
        model: dict[str, Any] = {"emission_kernels": "sal"} if sal_emission_on else {}

        if arguments.sal and arguments.no_patch:
            selected = tuple(
                swap for swap in SWAPS if swap.name == "pipeline_clone_assignment"
            )
        if figures:
            selected = selected + FIGURE_SWAPS

            from port.extensions.figure_style import figure_font

            stack.enter_context(figure_font())
        if arguments.png_copies:
            if not figures:
                _parser().error("--png-copies needs the figure swaps")

            selected = with_options(
                selected, "port.patch.utils:write_fig", png_copy=True
            )
        if arguments.sample_layout is not None:
            if not figures:
                _parser().error("--sample-layout needs the figure swaps")

            selected = with_options(
                selected,
                "port.patch.plotting:plot_clones_spatial",
                preferred_sample_layout=arguments.sample_layout,
            )

        if arguments.genomic_colours is not None:
            selected = with_options(
                selected,
                "port.patch.plot_genomic:plot_clones_genomic",
                preferred_colour_by=arguments.genomic_colours,
            )
        # NB on unless refused, and off with `--no-patch` like the figures: a
        #    baseline arm decodes under `cnaster`'s caps.
        copy_cap = (
            not arguments.no_patch if arguments.copy_cap is None else arguments.copy_cap
        )
        if copy_cap:
            # NB refused here rather than at the decode, hours into the run.
            for finding in findings:
                if finding.kind == "invalid" and finding.key.startswith("int_copy"):
                    _parser().error(f"--copy-cap: {finding.detail}")

            selected = selected + COPY_SWAPS
        # NB opt-in on the default arm, where each alone over-splits #338's
        #    three-sample instance (6 fitted clones against 2 planted); on with
        #    --sal, which with both recovers CalicoST hard at 0.982 against
        #    0.303 and keeps every other fixture measured (#467).
        refinement_mask = bool(
            arguments.sal
            if arguments.refinement_mask is None
            else arguments.refinement_mask
        )
        floor = bool(
            arguments.sal if arguments.floor_merge is None else arguments.floor_merge
        )
        # NB the mask and the floor are read by port's
        #    `pipeline_clone_assignment` alone; without it both would be
        #    installed and read by nothing (#466).
        if (refinement_mask or floor) and not any(
            swap.name == "pipeline_clone_assignment" for swap in selected
        ):
            _parser().error(
                "--refinement-mask and --floor-merge need port's "
                "pipeline_clone_assignment, which --no-patch leaves out"
            )
        if refinement_mask:
            selected = selected + REFINEMENT_SWAPS
        if floor:
            selected = with_options(
                selected,
                "port.patch.hmrf:pipeline_clone_assignment",
                floor_merge=True,
            )
        # NB options of `port`'s `run_core_inference`, bound once the shift's
        #    table, which holds it, is selected (#517).
        inference: dict[str, Any] = {}
        distinct = (
            shift and not arguments.no_patch
            if arguments.distinct_init is None
            else arguments.distinct_init
        )
        if distinct:
            inference["distinct_init"] = True
        hmm_start = (
            ("kmeans++x5+em" if arguments.sal else "none")
            if arguments.hmm_start is None
            else arguments.hmm_start
        )
        if hmm_start != "none":
            from port.patch.hmm_initialize.sal_mixture import checked

            inference["hmm_start"] = checked(hmm_start)

        # NB the copy rows decode by the HMM's likelihood only (#362), which
        #    reads each clone's counts from the fit this captures; entered
        #    before `patched`, so the shift's row installs the capturing
        #    `run_core_inference`.
        if copy_cap:
            from port.extensions.copy_likelihood import capture

            stack.enter_context(capture())

            for swap in COPY_SWAPS:
                selected = with_options(
                    selected, swap.replacement, decoder=arguments.copy_decode
                )
        if shift:
            selected = selected + SHIFT_SWAPS

            # NB the genomic figure draws each clone's line at `mu / Z_c` when
            #    the fit it plots was shifted (#299).
            selected = with_options(
                selected,
                "port.patch.plot_genomic:plot_clones_genomic",
                logmu_shift=True,
            )

        if model:
            selected = with_options(
                selected, "port.patch.hmm_nophasing:hmm_nophasing", **model
            )

        if inference:
            selected = with_options(
                selected, "port.patch.hmrf:run_core_inference", **inference
            )

        if arguments.sal:
            from port.extensions.sal import sal_options

            selected = with_options(
                selected,
                "port.patch.hmrf:pipeline_clone_assignment",
                **sal_options(shift=shift),
            )

            if arguments.no_patch:
                print(
                    "run_cnaster_port: --no-patch --shift assigns clones "
                    "unshifted; the shift reaches the HMM only",
                    file=sys.stderr,
                )

        if arguments.no_plots:
            # NB after every other table, so it rebinds whichever `write_fig`
            #    the figure swaps put in place.
            selected = selected + PLOT_OFF_SWAPS

        if selected:
            sites = stack.enter_context(patched(selected))
            print(
                f"run_cnaster_port: {len(selected)} replacements over "
                f"{len(sites)} bindings"
                + (", figures included" if figures else "")
                + (", copy caps from the config" if copy_cap else "")
                + (", refinement mask" if refinement_mask else "")
                + (f", sal HMM start {hmm_start}" if hmm_start != "none" else "")
                + (", floor merged smallest first" if floor else "")
                + (", distinct initial states" if distinct else "")
                + (", shift included" if shift else "")
                + (", rust lattices" if rust else "")
                + (", sal included" if arguments.sal else "")
                + (", no plots written" if arguments.no_plots else ""),
                file=sys.stderr,
            )
        else:
            print(
                "run_cnaster_port: --no-patch, nothing rebound"
                + (" but the rust lattices" if rust else ""),
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
            stack.enter_context(instrumented(SWAPS + FIGURE_SWAPS))
            if arguments.time_stages
            else None
        )

        # NB every segmentation the patched stages make, as labellings of the
        #    same genes (#438); empty under `--no-patch`, which records nothing.
        from port.extensions.segments import recording

        lineage = stack.enter_context(recording())

        started = time.perf_counter()
        pipeline.run_cnaster(arguments.config)
        wall = time.perf_counter() - started

    if spent is not None:
        _report(spent, wall, patched=not arguments.no_patch)

    # NB after the run and outside its timer, and off with `--no-patch`: a
    #    baseline arm writes what `cnaster` writes and nothing beside it.
    if not (arguments.no_outputs or arguments.no_patch):
        _write_outputs(
            arguments.config,
            {
                "figures": figures,
                "shift": shift,
                "copy_decode": f"lattice_decode ({arguments.copy_decode})"
                if copy_cap
                else "cnaster",
            },
            lineage.table(),
        )
    if kept is not None:
        _write_copy_sets(arguments.config, kept)

    print(f"run_cnaster_port: {wall:.2f}s", file=sys.stderr)
    return 0


def _write_outputs(config: str, flags: dict[str, Any], segments: Any) -> None:
    """`port.extensions.outputs` into each run directory the run wrote.

    `segments` is the run's lineage, one row per gene and one label column
    per segmentation (#438), written beside them as `gene_segments.tsv`.
    """
    from pathlib import Path

    from port.extensions.outputs import config_keys, run_directories, write_outputs

    output_dir = config_keys(Path(config)).get("output_dir")

    if output_dir is None:
        print(
            "run_cnaster_port: no output_dir in the config; outputs skipped",
            file=sys.stderr,
        )
        return

    for run in run_directories(Path(output_dir)):
        write_outputs(run, Path(config), flags)
        if len(segments):
            segments.to_csv(run / "gene_segments.tsv", sep="\t", index=False)
        print(f"run_cnaster_port: outputs written to {run}", file=sys.stderr)


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
