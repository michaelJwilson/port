"""`run_cnaster` with `port.pipeline.SWAPS` installed; only `--no-patch`, which installs nothing, is bitwise `cnaster` (#195, #466)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NamedTuple

from port.pipeline import (
    COPY_SWAPS,
    DEFAULTS,
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

GENOMIC_FIGURE = "port.patch.plot_genomic:plot_clones_genomic"


def _installs(selected: tuple[Any, ...], replacement: str) -> bool:
    return any(swap.replacement == replacement for swap in selected)


def _timed(selected: tuple[Any, ...]) -> tuple[Any, ...]:
    """`SWAPS`, `FIGURE_SWAPS` and the selected rows, once per binding; classes excluded."""
    import importlib
    import inspect

    rows: dict[tuple[str, str], Any] = {}

    for swap in (*SWAPS, *FIGURE_SWAPS, *selected):
        module = importlib.import_module(swap.module)

        if not inspect.isclass(getattr(module, swap.name, None)):
            rows.setdefault((swap.module, swap.name), swap)

    return tuple(rows.values())


def _layout(text: str) -> tuple[int, int]:
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
        help="write no cnamaste.h5 (T- #817); off with --no-patch",
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
        help="keep read-depth sub-clones inside their BAF clone (#348, #467); on, off with --no-patch unless --sal",
    )
    parser.add_argument(
        "--floor-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="meet the clone-size floor smallest first (#348); on, off with --no-patch unless --sal",
    )
    parser.add_argument(
        "--min-segment-normal-umi",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="floor read-depth segments at 300 normal UMI where the config states no quality key (#551, #547); off, on with --sal, refused with --no-patch",
    )
    parser.add_argument(
        "--hmm-start",
        default=None,
        metavar="START",
        help="the read-depth HMM's copy-state start, a sal mixture start or lattice (#489, #547); none, kmeans++x5+em with --sal",
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
        help="snakes_and_ladders' labelling (#312) and the HMM start kmeans++x5+em (#489)",
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
    """Each tri-state flag resolved (#517); `None` from the parser means not asked."""

    figures: bool
    shift: bool
    rust: bool
    sal_emission: bool
    copy_cap: bool
    refinement_mask: bool
    floor: bool
    min_segment_normal_umi: bool
    """The read-depth segment floor: on with `--sal` alone (#551, T- #617; not in `DEFAULTS`)."""
    distinct: bool
    hmm_start: str


def _settings(arguments: argparse.Namespace) -> Settings:
    def asked(value: Any, default: Any) -> Any:
        return default if value is None else value

    patch = not arguments.no_patch
    shift = bool(asked(arguments.shift, patch))

    return Settings(
        figures=bool(asked(arguments.figure_swaps, patch)),
        shift=shift,
        rust=bool(asked(arguments.rust, patch)),
        sal_emission=bool(asked(arguments.sal_emission, shift)),
        copy_cap=bool(asked(arguments.copy_cap, patch)),
        refinement_mask=bool(asked(arguments.refinement_mask, patch or arguments.sal)),
        floor=bool(asked(arguments.floor_merge, patch or arguments.sal)),
        # NB `--sal`'s alone: the default arm decodes a planted (1, 2) as (1, 3) (#645, #617).
        min_segment_normal_umi=bool(
            asked(arguments.min_segment_normal_umi, patch and arguments.sal)
        ),
        distinct=bool(asked(arguments.distinct_init, shift and patch)),
        hmm_start=str(
            asked(
                arguments.hmm_start,
                "kmeans++x5+em" if arguments.sal and shift else "none",
            )
        ),
    )


def _refusals(arguments: argparse.Namespace, settings: Settings) -> list[str]:
    """Flags asked for where nothing would read them, as argument errors (#466)."""
    refused = [
        f"{flag} needs the figure swaps"
        for flag, asked in (
            ("--genomic-colours", arguments.genomic_colours is not None),
            ("--png-copies", arguments.png_copies),
            ("--sample-layout", arguments.sample_layout is not None),
        )
        if asked and not settings.figures
    ]

    # NB without the shift's captured fit the copy decode fails hours in (#576).
    if settings.copy_cap and not settings.shift:
        refused.append(
            "--no-shift leaves the copy decode no captured fit; add --no-copy-cap"
        )

    refused += [
        f"{flag} is read by the shift rows; --no-shift"
        for flag, asked in (
            ("--sal-emission", arguments.sal_emission),
            ("--distinct-init", arguments.distinct_init),
            ("--hmm-start", arguments.hmm_start),
        )
        if asked and not settings.shift
    ]

    if (settings.refinement_mask or settings.floor) and (
        arguments.no_patch and not arguments.sal
    ):
        refused.append(
            "--refinement-mask and --floor-merge need port's "
            "pipeline_clone_assignment, which --no-patch leaves out"
        )

    if settings.min_segment_normal_umi and arguments.no_patch:
        refused.append(
            "--min-segment-normal-umi needs port's create_bin_ranges, "
            "which --no-patch leaves out"
        )

    return refused


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline, patched unless `--no-patch` is given."""
    arguments = _parser().parse_args(argv)

    # NB imported before the swaps so rebinding finds its bindings; local to skip it on `--list`.
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
        for default in DEFAULTS:
            print(
                f"{default.setting}  (#{default.ticket}, port's own, on by default; "
                f"--no-{default.flag.removeprefix('--')} to omit)"
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

    opened: Any = None
    with ExitStack() as stack:
        figures = settings.figures
        rust = settings.rust

        if rust:
            from port.patch.lattice import rust_lattices

            stack.enter_context(rust_lattices())

        selected = SWAPS if not arguments.no_patch else ()

        sal_emission_on = settings.sal_emission
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
        copy_cap = settings.copy_cap
        if copy_cap:
            for finding in findings:
                if finding.kind == "invalid" and finding.key.startswith("int_copy"):
                    _parser().error(f"--copy-cap: {finding.detail}")

            selected = selected + COPY_SWAPS
        # NB on together (#617): each alone over-splits #338's instance (#467).
        refinement_mask, floor = settings.refinement_mask, settings.floor
        if refinement_mask:
            selected = selected + REFINEMENT_SWAPS
        if floor:
            selected = with_options(
                selected,
                "port.patch.hmrf:pipeline_clone_assignment",
                floor_merge=True,
            )
        inference: dict[str, Any] = {}
        distinct = settings.distinct
        if distinct:
            inference["distinct_init"] = True
        hmm_start = settings.hmm_start
        if hmm_start != "none":
            from port.patch.hmm_initialize.sal_mixture import checked

            inference["hmm_start"] = checked(hmm_start)

        # NB the copy rows decode the fit this captures (#362); entered before `patched`.
        if copy_cap:
            from port.extensions.copy_likelihood import capture

            stack.enter_context(capture())
        if shift:
            selected = selected + SHIFT_SWAPS + LOG_SPACE_SWAPS

            # NB the log-space kernels (#560, #561); absent under `--no-patch --shift` without `--sal`.
            if _installs(selected, "port.patch.hmrf:pipeline_clone_assignment"):
                selected = with_options(
                    selected,
                    "port.patch.hmrf:pipeline_clone_assignment",
                    log_space=True,
                )

            # NB the shifted line at `mu / Z_c` (#299); without figure swaps, this row alone (#617).
            if not figures:
                selected = selected + tuple(
                    swap for swap in FIGURE_SWAPS if swap.replacement == GENOMIC_FIGURE
                )
            selected = with_options(selected, GENOMIC_FIGURE, logmu_shift=True)

        if model:
            selected = with_options(
                selected, "port.patch.hmm_nophasing:hmm_nophasing", **model
            )

        if inference:
            selected = with_options(
                selected, "port.patch.hmrf:run_core_inference", **inference
            )

        if settings.min_segment_normal_umi:
            from port.patch.omics.blocks import MIN_SEGMENT_NORMAL_UMI

            # NB the segment floor (#551, #547); a configuration's `quality` keys win.
            selected = with_options(
                selected,
                "port.patch.omics:create_bin_ranges",
                min_segment_normal_umi=MIN_SEGMENT_NORMAL_UMI,
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
            # NB last, so it rebinds whichever `write_fig` is in place.
            selected = selected + PLOT_OFF_SWAPS

        # NB recorded for the pages (T- #817); entered before the swaps install.
        pages = None
        if figures and not arguments.no_plots and not arguments.no_patch:
            from port.extensions.combined_figure import recording as page_calls

            pages = stack.enter_context(page_calls())

        if selected:
            sites = stack.enter_context(patched(selected))
            print(
                f"run_cnaster_port: {len(selected)} replacements over "
                f"{len(sites)} bindings"
                + (", figures included" if figures else "")
                + (", copy caps from the config" if copy_cap else "")
                + (", refinement mask" if refinement_mask else "")
                + (f", HMM start {hmm_start}" if hmm_start != "none" else "")
                + (", floor merged smallest first" if floor else "")
                + (", distinct initial states" if distinct else "")
                + (", shift included" if shift else "")
                + (
                    ", the genomic figure's shifted line"
                    if shift and not figures
                    else ""
                )
                + (", rust lattices" if rust else "")
                + (", sal included" if arguments.sal else "")
                + (
                    ", read-depth segments of 300 normal UMI unless configured"
                    if settings.min_segment_normal_umi
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

        # NB after the swaps and before the timer, so compilation is not measured.
        if arguments.warm_up:
            warmed = warm()
            print(f"run_cnaster_port: {warmed.report()}", file=sys.stderr)

        # NB timer wraps the installed implementation; figure swap timed on both arms (#617).
        spent = (
            stack.enter_context(instrumented(_timed(selected)))
            if arguments.time_stages
            else None
        )

        # NB each segmentation as labellings of the same genes (#438), staged into `cnamaste.h5`.
        from port.extensions.segments import recording

        stack.enter_context(recording())

        if not (arguments.no_outputs or arguments.no_patch):
            opened = _open_cnamaste(arguments.config, vars(arguments))
            if opened is not None:
                import json

                from port.extensions import cnamaste
                from port.extensions.run_record import tapping

                stack.enter_context(cnamaste.writing(opened))
                stack.enter_context(
                    tapping(
                        pipeline,
                        Path(arguments.config),
                        json.dumps(vars(arguments), default=str),
                    )
                )

        since = time.time()
        started = time.perf_counter()
        pipeline.run_cnaster(arguments.config)
        wall = time.perf_counter() - started

    if spent is not None:
        _report(spent, wall, patched=not arguments.no_patch)

    # NB outside the timer; off with `--no-patch`, so a baseline writes only what `cnaster` does.
    if pages is not None:
        _write_pages(
            arguments.config, pages, since=since, png_copy=arguments.png_copies
        )
    if not (arguments.no_outputs or arguments.no_patch):
        _write_outputs(arguments.config, since=since, cnamaste_file=opened)
    if opened is not None:
        from port.extensions.run_record import release

        release()

    print(f"run_cnaster_port: {wall:.2f}s", file=sys.stderr)
    return 0


def _write_pages(
    config: str, pages: Any, *, since: float | None, png_copy: bool
) -> None:
    """`genomic.pdf`, `spatial.pdf` and `combined.pdf` into each run directory's `plots/` (T- #817)."""
    from port.extensions.combined_figure import run_slide, write_pages
    from port.extensions.outputs import config_keys, run_directories

    output_dir = config_keys(Path(config)).get("output_dir")
    if output_dir is None:
        return
    slide = run_slide(Path(config))
    for run in run_directories(Path(output_dir), since):
        for page in write_pages(pages, run / "plots", slide, png_copy=png_copy):
            print(f"run_cnaster_port: wrote {page}", file=sys.stderr)


def _open_cnamaste(config: str, flags: dict[str, Any]) -> Any:
    """`<output_dir>/cnamaste.h5`, made with its root attributes; `None` without an `output_dir`."""
    import json
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    from port.extensions import cnamaste
    from port.extensions.outputs import config_keys
    from port.extensions.repository import commit as checkout
    from port.sim.fixtures import realization_hash

    output_dir = config_keys(Path(config)).get("output_dir")
    if output_dir is None:
        return None

    def installed(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    try:
        commit = checkout()
    except Exception:  # noqa: BLE001 -- a wheel install has no checkout to name
        commit = "unknown"
    import yaml

    paths = (yaml.safe_load(Path(config).read_text()) or {}).get("paths") or {}
    sheet = Path(paths.get("sample_sheet") or Path(config).parent / "sample_sheet.tsv")
    path = Path(output_dir) / cnamaste.FILE
    cnamaste.create(
        path, commit=commit, port=installed("port"), cnaster=installed("cnaster"),
        sal=installed("snakes_and_ladders"), sample_hash=realization_hash(sheet.parent),
        flags=json.dumps(flags, default=str),
    )  # fmt: skip
    return path


def _write_outputs(
    config: str, *, since: float | None = None, cnamaste_file: Any = None
) -> None:
    """The integer stages into `cnamaste.h5` for each run directory written after `since` (T- #817, T- #617)."""
    from port.extensions.outputs import config_keys, run_directories
    from port.extensions.run_record import integer_groups

    output_dir = config_keys(Path(config)).get("output_dir")

    if output_dir is None or cnamaste_file is None:
        return

    for run in run_directories(Path(output_dir), since):
        integer_groups(cnamaste_file, run, Path(config))
        print(
            f"run_cnaster_port: integer stages written to {cnamaste_file}",
            file=sys.stderr,
        )


def _report(spent: dict[str, Spent], wall: float, *, patched: bool) -> None:
    arm = "patched" if patched else "baseline"
    width = max(len(name) for name in spent)
    total = sum(entry.seconds for entry in spent.values())

    print(f"\n  {arm}: {wall:.2f}s whole run", file=sys.stderr)
    print(
        f"  {'stage':<{width}} {'calls':>6} {'seconds':>9} {'share':>7}"
        f" {'first':>9} {'warm/call':>10}",
        file=sys.stderr,
    )

    # NB a compiled kernel's first call is compilation, so reported apart (#204).
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
