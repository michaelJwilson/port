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
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any, NamedTuple

from port.pipeline import (
    COPY_SWAPS,
    FIGURE_SWAPS,
    LOG_SPACE_SWAPS,
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
        description=(
            "Run cnaster's pipeline with port's replacements installed. "
            "README.md, 'Running the pipeline patched', has each option's "
            "measurements."
        ),
    )
    parser.add_argument(
        "config", nargs="?", help="the YAML configuration run_cnaster reads"
    )
    parser.add_argument(
        "--no-outputs",
        action="store_true",
        help="skip port's fitted and decoded tables beside cnaster's (#331); off with --no-patch",
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
        help="install FIGURE_SWAPS (#195, #299, #309); on, off with --no-patch",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="build every figure and write none (#403)",
    )
    parser.add_argument(
        "--shift",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "install SHIFT_SWAPS, the per-clone shift and its pin (#276, #299), "
            "and LOG_SPACE_SWAPS (#560, #561); on, off with --no-patch"
        ),
    )
    parser.add_argument(
        "--refinement-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep read-depth sub-clones inside their BAF clone (#348, #467); off, on with --sal",
    )
    parser.add_argument(
        "--floor-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="meet the clone-size floor smallest first (#348); off, on with --sal",
    )
    parser.add_argument(
        "--hmm-start",
        default=None,
        metavar="START",
        help="the read-depth HMM's copy-state start, a sal mixture start or lattice (#489, #547); none, kmeans++x5+em with --sal",
    )
    parser.add_argument(
        "--baf-start",
        default=None,
        metavar="START",
        help="the BAF-only HMM's copy-state start, a sal mixture start or lattice (#540); none keeps distinct's",
    )
    parser.add_argument(
        "--distinct-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="start the HMM from distinct GMM components (#348); on where the shift is",
    )
    parser.add_argument(
        "--copy-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="decode integer copies by likelihood under the configured cap (#313, #362); on, off with --no-patch",
    )
    parser.add_argument(
        "--png-copies",
        action="store_true",
        help="write a PNG without metadata beside each PDF (#452); needs the figure swaps",
    )
    parser.add_argument(
        "--sample-layout",
        type=_layout,
        default=None,
        metavar="ROWS,COLUMNS",
        help="clone spatial plots one panel per sample on this grid, e.g. 3,1 (#328); needs the figure swaps",
    )
    parser.add_argument(
        "--genomic-colours",
        choices=("integer", "states"),
        default=None,
        help="colour clones_genomic by integer (A, B) or by fitted state; needs the figure swaps",
    )
    parser.add_argument(
        "--rust",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cnaster's four lattices from oxiport, bitwise (#318); on, off with --no-patch",
    )
    parser.add_argument(
        "--sal-emission",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="score the coded emission with sal's kernels (#425); on where the shift is",
    )
    parser.add_argument(
        "--sal",
        action="store_true",
        help="snakes_and_ladders' labelling, and the mask, floor and start it implies (#312)",
    )
    parser.add_argument(
        "--copy-errors",
        action="store_true",
        help="write cnv_copy_sets.tsv, each state's 95 per cent credible (A, B) (#353); needs the shift",
    )
    parser.add_argument(
        "--copy-decode",
        choices=("lattice", "shared"),
        default="lattice",
        help="lattice, per clone (#370), or shared, one pair per state (#327)",
    )
    parser.add_argument(
        "--parsimony-decode",
        action="store_true",
        help="the lattice decode's prior -0.5 |A + B - 2| per bin (T- #471); off, flat",
    )
    parser.add_argument(
        "--warm-up",
        action="store_true",
        help="compile every kernel before the clock starts (#211)",
    )
    parser.add_argument(
        "--time-stages",
        action="store_true",
        help="report what the swapped names cost in this run, patched or not",
    )
    parser.add_argument(
        "--audit-config",
        action="store_true",
        help="print what the configuration states that cnaster does not use (#324), and exit",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what would be rebound, and exit without running",
    )
    return parser


class Settings(NamedTuple):
    """What each tri-state flag resolves to, read once from the arguments (#517).

    `None` from the parser means "not asked", which is what lets
    `--no-patch --figure-swaps` compose: the default follows the arm, and an
    explicit flag overrides it either way.
    """

    figures: bool
    """`FIGURE_SWAPS`: on, off with `--no-patch` -- a baseline arm draws `cnaster`'s."""
    shift: bool
    """`SHIFT_SWAPS`: on, off with `--no-patch` -- a baseline fits `cnaster`'s model."""
    rust: bool
    """The Rust lattices: bitwise, so on, and off with `--no-patch`."""
    sal_emission: bool
    """sal's coded emission: on where the shift is, whose row reads it."""
    copy_cap: bool
    """`COPY_SWAPS`: on, off with `--no-patch` -- a baseline decodes under `cnaster`'s caps."""
    refinement_mask: bool
    """`REFINEMENT_SWAPS`: off, on with `--sal` (#467)."""
    floor: bool
    """The floor merge: off, on with `--sal` (#467)."""
    distinct: bool
    """The distinct initializer: on where the shift is, off with `--no-patch`."""
    hmm_start: str
    """The read-depth HMM's copy-state start: `none`, `kmeans++x5+em` with `--sal` (#489)."""
    baf_start: str
    """The BAF-only stage's start: `none`, `distinct`'s kept (#540)."""
    parsimony: float
    """The lattice decode's prior weight: `0`, flat, `PARSIMONY` with `--parsimony-decode`."""


def _settings(arguments: argparse.Namespace) -> Settings:
    """Each flag as asked, or its default for this arm."""

    def asked(value: Any, default: Any) -> Any:
        return default if value is None else value

    from port.extensions.copy_likelihood import PARSIMONY

    patch = not arguments.no_patch
    shift = bool(asked(arguments.shift, patch))

    return Settings(
        figures=bool(asked(arguments.figure_swaps, patch)),
        shift=shift,
        rust=bool(asked(arguments.rust, patch)),
        sal_emission=bool(asked(arguments.sal_emission, shift)),
        copy_cap=bool(asked(arguments.copy_cap, patch)),
        refinement_mask=bool(asked(arguments.refinement_mask, arguments.sal)),
        floor=bool(asked(arguments.floor_merge, arguments.sal)),
        distinct=bool(asked(arguments.distinct_init, shift and patch)),
        hmm_start=str(
            asked(arguments.hmm_start, "kmeans++x5+em" if arguments.sal else "none")
        ),
        baf_start=str(asked(arguments.baf_start, "none")),
        parsimony=PARSIMONY if arguments.parsimony_decode else 0.0,
    )


def _refusals(arguments: argparse.Namespace, settings: Settings) -> list[str]:
    """Every flag asked for where nothing would read it, as argument errors.

    Refused rather than ignored (#466), and before the configuration is
    read: an argument error, not a file error.
    """
    refused = [
        f"{flag} needs the figure swaps"
        for flag, asked in (
            ("--genomic-colours", arguments.genomic_colours is not None),
            ("--png-copies", arguments.png_copies),
            ("--sample-layout", arguments.sample_layout is not None),
        )
        if asked and not settings.figures
    ]

    # NB the decode compares `(A + B) / 2` against the pinned rates; an
    #    unshifted fit's rates carry the baseline's per-clone scale (#353).
    if arguments.copy_errors and not settings.shift:
        refused.append("--copy-errors needs the shift; drop --no-shift")
    # NB the copy rows decode the fit the shift's `run_core_inference` row
    #    captures; without it the decode stopped hours in, with no captured
    #    fit (#576). Refused here, naming the way out.
    elif settings.copy_cap and not settings.shift:
        refused.append(
            "--no-shift leaves the copy decode no captured fit; add --no-copy-cap"
        )

    # NB the prior is the lattice decode's alone; `shared` and `cnaster`'s
    #    decoders take none.
    if arguments.parsimony_decode and not (
        settings.copy_cap and arguments.copy_decode == "lattice"
    ):
        refused.append("--parsimony-decode needs the lattice copy decode")

    # NB read by the `SHIFT_SWAPS` rows alone -- port's `hmm_nophasing` class
    #    and `run_core_inference`.
    refused += [
        f"{flag} is read by the shift rows; --no-shift"
        for flag, asked in (
            ("--sal-emission", arguments.sal_emission),
            ("--distinct-init", arguments.distinct_init),
            ("--baf-start", arguments.baf_start),
        )
        if asked and not settings.shift
    ]

    # NB read by port's `pipeline_clone_assignment` alone, which `--no-patch`
    #    leaves out unless `--sal` installs it.
    if (settings.refinement_mask or settings.floor) and (
        arguments.no_patch and not arguments.sal
    ):
        refused.append(
            "--refinement-mask and --floor-merge need port's "
            "pipeline_clone_assignment, which --no-patch leaves out"
        )

    return refused


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
        for swap in LOG_SPACE_SWAPS:
            print(
                f"{swap.module}.{swap.name} <- {swap.replacement}  "
                f"(#{swap.ticket}, to 1e-9 where cnaster is right; with the shift)"
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

    settings = _settings(arguments)

    for refusal in _refusals(arguments, settings):
        _parser().error(refusal)

    # NB the clone assignment applies the shift through `port`'s
    #    `pipeline_clone_assignment`, which is in `SWAPS`, so
    #    `--no-patch --shift` fits shifted and assigns clones unshifted; it
    #    is allowed, and said.
    shift = settings.shift

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
        figures = settings.figures
        # NB bitwise, so on by default like `SWAPS`, and off with it: a
        #    baseline arm is `cnaster`'s compiled code as well as its names.
        rust = settings.rust

        if rust:
            from port.patch.lattice import rust_lattices

            stack.enter_context(rust_lattices())

        # NB before the swaps, so `patched` installs the capturing wrapper and
        #    the fit kept is the pinned one.
        from port.extensions.copy_errors import captured_fits

        kept = stack.enter_context(captured_fits()) if arguments.copy_errors else None

        selected = SWAPS if not arguments.no_patch else ()

        # NB `--sal` selects the clone labelling through `port`'s
        #    `pipeline_clone_assignment`, a `SWAPS` row; under `--no-patch`
        #    that one row is installed alone, so the flag still means what it
        #    says and the rest of the baseline stays `cnaster`'s.
        # NB the sal emission and the distinct initializer are read by the
        #    `SHIFT_SWAPS` rows alone -- port's `hmm_nophasing` class and
        #    `run_core_inference` -- so without the shift they are off, and
        #    asking for either is refused rather than ignored (#466).
        sal_emission_on = settings.sal_emission
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
            selected = with_options(
                selected, "port.patch.utils:write_fig", png_copy=True
            )
        if arguments.sample_layout is not None:
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
        copy_cap = settings.copy_cap
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
        refinement_mask, floor = settings.refinement_mask, settings.floor
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
        distinct = settings.distinct
        if distinct:
            inference["distinct_init"] = True
        hmm_start = settings.hmm_start
        if hmm_start != "none":
            from port.patch.hmm_initialize.sal_mixture import checked

            inference["hmm_start"] = checked(hmm_start)
        if settings.baf_start != "none":
            from port.patch.hmm_initialize.sal_mixture import checked

            inference["baf_start"] = checked(settings.baf_start)

        # NB the copy rows decode by the HMM's likelihood only (#362), which
        #    reads each clone's counts from the fit this captures; entered
        #    before `patched`, so the shift's row installs the capturing
        #    `run_core_inference`.
        if copy_cap:
            from port.extensions.copy_likelihood import capture

            stack.enter_context(capture())

            for swap in COPY_SWAPS:
                selected = with_options(
                    selected,
                    swap.replacement,
                    decoder=arguments.copy_decode,
                    parsimony=settings.parsimony,
                )
        if shift:
            # NB the emission kernels where `cnaster`'s are wrong (#560,
            #    #561), on the arm that already departs from its fit; the
            #    field compiles its kernels in, so it takes them as an option.
            selected = with_options(
                selected + SHIFT_SWAPS + LOG_SPACE_SWAPS,
                "port.patch.hmrf:pipeline_clone_assignment",
                log_space=True,
            )

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
            from port.patch.omics.blocks import SAL_NORMAL_UMI_FLOOR

            # NB the read-depth segment floor (#551) the lattice start was
            #    tuned at (#547); a configuration's `quality` keys still win.
            selected = with_options(
                selected,
                "port.patch.omics:create_bin_ranges",
                normal_umi_floor=SAL_NORMAL_UMI_FLOOR,
            )

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
                + (
                    f", parsimony {settings.parsimony}"
                    if arguments.parsimony_decode
                    else ""
                )
                + (", refinement mask" if refinement_mask else "")
                + (f", HMM start {hmm_start}" if hmm_start != "none" else "")
                + (
                    f", BAF start {settings.baf_start}"
                    if settings.baf_start != "none"
                    else ""
                )
                + (", floor merged smallest first" if floor else "")
                + (", distinct initial states" if distinct else "")
                + (", shift included" if shift else "")
                + (", rust lattices" if rust else "")
                + (", sal included" if arguments.sal else "")
                + (
                    ", read-depth segments of 300 normal UMI unless configured"
                    if arguments.sal
                    else ""
                )
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
                "parsimony": settings.parsimony if copy_cap else None,
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
