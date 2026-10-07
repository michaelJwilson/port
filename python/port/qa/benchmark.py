"""Benchmarks and the figure run: what `run_benchmark` and `run_figures` measure and draw (T- #673 G4).

- `--final` (#494): `run_cnaster_port --sal` (`port`) and CalicoST under its
  own shipped configuration (`calicost`) on one sample, each in a child
  process, scored by `port.qa.audit.score_sample`. `port` repeats on a warm
  numba cache and reports the median wall; `calicost` runs once under a cap
  (0 for none), on a kept `root` it resumes from (#532, PR- #677).
- `--patched-share` (#302): the share of `cnaster` lines an unpatched run
  executes that `run_cnaster_port` replaces, recorded for the badge.
- `run_figures`: the dev instance's figures, as a user of the entry point
  gets them (`figures`).

Moved from `tests/final_benchmark.py`, `tests/patched_share.py` and
`tests/generate_plots.py`, which repeated their argument parsing, timing and
peak-memory reads.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from port.qa import provenance
from port.qa.provenance import ROOT
from port.qa.statistics import measured, median_wall, peak_gb
from port.sim.files import located
from port.sim.truth import COPY_LATTICE, CoreInferenceTruth, dev_instance

if TYPE_CHECKING:
    from port.extensions.combined_figure import Recorded

TIMEOUT = 1800
"""CalicoST's budget per case, in seconds (#494)."""

BADGES = ROOT / ".badges"
MEASUREMENTS = BADGES / "measurements.json"
"""Where the badges' measured figures live; `tests.badges` renders them."""

STAGED = "staged.json"
"""Under a kept `root`: the staged configuration and whether it is joint, so a rerun resumes."""


def _shipped(joint: bool) -> Path:
    """CalicoST's `configuration_cna`, or `configuration_cna_multi` for several slices."""
    from port.sim.fixtures import references

    resources = references()

    if resources is None:
        msg = "CalicoST's checkout is not installed; its configuration_cna is needed"
        raise FileNotFoundError(msg)

    return resources.parent / (
        "configuration_cna_multi" if joint else "configuration_cna"
    )


def _sample(name: str) -> Any:
    from port.sim.fixtures import load_simulated

    path = Path(name)
    return (
        load_simulated(path.name, path.parent)
        if path.is_absolute()
        else load_simulated(name)
    )


def port(name: str, repeats: int) -> dict[str, Any]:
    """`run_audit --sim -- --sal` in a child, `repeats` times; the last run's scores, the median wall."""
    done, wall, walls = median_wall(
        lambda: subprocess.run(
            [
                sys.executable,
                "-m",
                "port.scripts.run_audit",
                "--sim",
                "--sample",
                name,
                "--",
                "--sal",
                "--no-plots",
            ],
            capture_output=True,
            text=True,
            check=True,
        ),
        repeats,
    )
    line = next(x for x in done.stdout.splitlines() if x.startswith("SIM "))
    row = json.loads(line[4:])

    return {
        "tool": "port --sal",
        "ari": row["ari"],
        "clones": row["n_clones"],
        "ari_integer": row["ari_integer"],
        "integer_clones": row["n_integer_clones"],
        "copy_ari": row["copy_ari"],
        "exact_altered": row["exact_altered"],
        "exact_altered_minor": row["exact_altered_minor"],
        "wall": round(wall, 1),
        "walls": [round(w, 1) for w in walls],
        "peak_gb": round(peak_gb(children=True), 2),
    }


def baf_stage(sample: Any, output: Path) -> dict[str, Any]:
    """Clone ARI (clones) of CalicoST's BAF-stage fit, for a run stopped before its end.

    The Neyman-Pearson-merged `mergedallspots` fit where CalicoST wrote it,
    else the HMRF's last round in `allspots`, before that merge.
    """
    import numpy as np

    from port.qa.scoring import shared_ari

    merged = sorted(output.glob("*/mergedallspots_nstates*_sp.npz"))
    unmerged = sorted(output.glob("*/allspots_nstates*_sp.npz"))

    if merged:
        fitted = np.load(merged[0], allow_pickle=True)["new_assignment"]
    elif unmerged:
        fit = np.load(unmerged[0], allow_pickle=True)
        fitted = fit[f"round{int(fit['num_iterations']) - 1}_assignment"]
    else:
        return {"baf_stage": None}

    meta = pd.read_csv(
        output / "parsed_inputs" / "table_meta.csv.gz", sep="\t", index_col=0
    )
    labels = pd.Series(fitted, index=meta.index)
    planted = pd.read_csv(
        located(sample.path / "truth_clone_labels.tsv"), sep="\t", index_col=0
    )["labels"]
    ari, spots = shared_ari(planted, labels)
    return {
        "baf_stage": "merged" if merged else "before the merge",
        "baf_stage_ari": round(ari, 4),
        "baf_stage_clones": int(labels.nunique()),
        "baf_stage_spots": spots,
        "written": sorted(p.name for p in output.glob("*/*.npz")),
    }


def joint_inputs(config: Path, root: Path) -> bool:
    """Stage a sheet of several slices as CalicoST's joint loader reads it; whether it had several.

    The drawn slices name their spots `<barcode>_<sample_id>` already, as the
    SNP directory does; CalicoST's `load_joint_data` appends `_<sample_id>`
    to each slice's barcodes itself, so none would match. Each slice is
    copied with the suffix removed and the configuration pointed at the copy.
    """
    import anndata

    document = yaml.safe_load(config.read_text())
    sheet = pd.read_csv(document["paths"]["sample_sheet"], sep=r"\s+")

    if len(sheet) == 1:
        return False

    slices = []

    for row in sheet.itertuples(index=False):
        suffix = f"_{row.sample_id}"
        into = root / "calicost_slices" / str(row.sample_id)
        (into / "spatial").mkdir(parents=True)
        counts = anndata.read_h5ad(
            Path(row.spaceranger_dir) / "filtered_feature_bc_matrix.h5ad"
        )
        counts.obs.index = counts.obs.index.str.removesuffix(suffix)
        counts.write_h5ad(into / "filtered_feature_bc_matrix.h5ad")
        positions = pd.read_csv(
            Path(row.spaceranger_dir) / "spatial" / "tissue_positions_list.csv",
            header=None,
        )
        positions[0] = positions[0].str.removesuffix(suffix)
        positions.to_csv(
            into / "spatial" / "tissue_positions_list.csv", header=False, index=False
        )
        slices.append(row._replace(spaceranger_dir=str(into)))

    joint = root / "calicost_sheet.tsv"
    pd.DataFrame(slices).to_csv(joint, sep="\t", index=False)
    document["paths"]["sample_sheet"] = str(joint)
    config.write_text(yaml.safe_dump(document, sort_keys=False))
    return True


def _with_clones(shipped: Path, n_clones: int | None, root: Path) -> Path:
    """The shipped file, with `n_clones` replaced where one is given."""
    if n_clones is None:
        return shipped

    lines = [
        f"n_clones : {n_clones}" if line.split(":")[0].strip() == "n_clones" else line
        for line in shipped.read_text().splitlines()
    ]
    edited = root / shipped.name
    edited.write_text("\n".join(lines) + "\n")
    return edited


def staged(sample: Any, root: Path) -> tuple[Path, bool]:
    """The sample's inputs staged under `root` for CalicoST; reused where `root` holds them already.

    CalicoST skips a stage whose checkpoint (`*.npz`) its output directory
    already holds, so a run killed part way resumes when rerun on the same
    `root` (#532: the uncapped `dev_tree` r0 run was resumed twice).
    """
    from port.qa.audit import drawn_config
    from port.sim.fixtures import write_sim_inputs

    marker = root / STAGED

    if marker.is_file():
        record = json.loads(marker.read_text())
        return Path(record["config"]), bool(record["joint"])

    root.mkdir(parents=True, exist_ok=True)
    config = Path(
        drawn_config(sample, root, {})
        if (sample.path / "snp").is_dir()
        else write_sim_inputs(sample, root, {})
    )
    joint = joint_inputs(config, root)
    marker.write_text(json.dumps({"config": str(config), "joint": joint}))
    return config, joint


def calicost(
    name: str,
    timeout: int = TIMEOUT,
    n_clones: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """CalicoST on its shipped configuration, under `timeout`, scored as port is.

    `n_clones` replaces the shipped file's clone count, its one edited value.
    `timeout` 0 runs uncapped. A kept `root` resumes a killed run from
    CalicoST's checkpoints (`staged`); `wall` is then this attempt's alone.
    """
    from port.qa.audit import score_sample

    sample = _sample(name)
    root = Path(tempfile.mkdtemp()) if root is None else root
    config, joint = staged(sample, root)
    tool = "CalicoST (shipped" + ("" if n_clones is None else f", n_clones {n_clones}")
    tool += ", uncapped)" if timeout == 0 else f", cap {timeout} s)"
    capped = [] if timeout == 0 else ["timeout", str(timeout)]
    failed: subprocess.CalledProcessError | None = None

    with measured() as cost:
        try:
            subprocess.run(
                [
                    *capped,
                    sys.executable,
                    "-c",
                    "from port.scripts.run_calicost import main; raise SystemExit(main())",
                    str(config),
                    "--shipped",
                    str(_with_clones(_shipped(joint), n_clones, root)),
                    "--no-align",
                    "--no-figures",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            failed = error

    wall = cost.wall_s

    if failed is not None:
        reached = [x for x in failed.stderr.splitlines() if " - " in x][-1:]
        return {
            "tool": tool,
            "finished": False,
            "returncode": failed.returncode,
            "wall": round(wall, 1),
            "peak_gb": round(peak_gb(children=True), 2),
            "reached": reached[0][:200] if reached else failed.stderr[-400:],
            "output": str(root / "output_calicost"),
            **baf_stage(sample, root / "output_calicost"),
        }

    print(f"calicost output: {root / 'output_calicost'}", file=sys.stderr, flush=True)

    try:
        recovery = score_sample(sample, root / "output_calicost", "calicost", wall)
    except Exception as error:  # noqa: BLE001 -- the run is kept; the scoring is reported
        return {
            "tool": tool,
            "finished": True,
            "scored": f"{type(error).__name__}: {error}",
            "wall": round(wall, 1),
            "peak_gb": round(peak_gb(children=True), 2),
            "output": str(root / "output_calicost"),
        }

    return {
        "tool": tool,
        "finished": True,
        "ari": recovery.ari,
        "clones": recovery.n_clones,
        "ari_integer": recovery.ari_integer,
        "integer_clones": recovery.n_integer_clones,
        "copy_ari": recovery.copy_ari,
        "exact_altered": recovery.exact_altered,
        "exact_altered_minor": recovery.exact_altered_minor,
        "wall": round(wall, 1),
        "peak_gb": round(peak_gb(children=True), 2),
        "output": str(root / "output_calicost"),
    }


# --- --patched-share --------------------------------------------------------


INSTANCE = "dev"
"""What the run is measured on: `port.sim.truth.dev_instance`, five states."""


def _lines(function: Any) -> tuple[str, set[int]]:
    """The file and line numbers `function`'s source occupies."""
    function = inspect.unwrap(function)
    source, start = inspect.getsourcelines(function)

    return os.path.realpath(inspect.getsourcefile(function) or ""), set(
        range(start, start + len(source))
    )


def patched_lines() -> dict[str, set[int]]:
    """`{file: lines}` of the `cnaster` code the default rows replace."""
    import importlib

    from port.pipeline import FIGURE_SWAPS, SHIFT_SWAPS, SWAPS

    spans: dict[str, set[int]] = {}

    for swap in SWAPS + FIGURE_SWAPS + SHIFT_SWAPS:
        original = getattr(importlib.import_module(swap.module), swap.name)
        module_name, _, attribute = swap.replacement.partition(":")
        replacement = getattr(importlib.import_module(module_name), attribute)

        if inspect.isclass(original):
            targets = [
                getattr(original, name)
                for name, value in vars(replacement).items()
                if callable(value) or isinstance(value, staticmethod | classmethod)
                if hasattr(original, name)
            ]
        else:
            targets = [original]

        for target in targets:
            try:
                path, lines = _lines(target)
            except (OSError, TypeError):
                continue

            spans.setdefault(path, set()).update(lines)

    return spans


def executed_lines() -> dict[str, set[int]]:
    """`{file: lines}` an unpatched `run_cnaster` executes on the dev instance.

    In a subprocess, because `numba` reads `NUMBA_DISABLE_JIT` at import and
    this process has imported it already.
    """
    import json

    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "executed.json"
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"from port.qa.benchmark import traced; traced({str(report)!r})",
            ],
            check=True,
            env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
        )
        recorded: dict[str, list[int]] = json.loads(report.read_text())

    return {path: set(lines) for path, lines in recorded.items()}


def traced(report: Path | str) -> None:
    """`executed_lines`' subprocess: one unpatched run under coverage, written as JSON to `report`."""
    import json

    import cnaster
    import coverage
    import matplotlib as mpl

    mpl.use("Agg")

    from port.sim.run_config import run_written
    from port.sim.truth import dev_instance

    # NB `__path__`, not `__file__`: this pin ships `cnaster` as a namespace
    #    package, with no `__init__.py` to name.
    root = str(Path(next(iter(cnaster.__path__))).resolve())
    tracer = coverage.Coverage(source=[root], data_file=None)

    with tempfile.TemporaryDirectory() as scratch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracer.start()
        try:
            run_written(
                dev_instance(),
                Path(scratch),
                port=False,
                max_iter_outer=1,
                max_iter=3,
                n_states=STATES,
            )
        finally:
            tracer.stop()

    data = tracer.get_data()
    Path(report).write_text(
        json.dumps(
            {
                os.path.realpath(path): sorted(data.lines(path) or ())
                for path in data.measured_files()
            }
        )
    )


def share(
    executed: dict[str, set[int]], patched: dict[str, set[int]]
) -> tuple[int, int]:
    """`(patched, executed)` line counts: executed lines inside patched spans."""
    total = sum(len(lines) for lines in executed.values())
    hit = sum(len(lines & patched.get(path, set())) for path, lines in executed.items())

    return hit, total


def record_patched_share() -> tuple[int, int]:
    """Measure the patched share and record it in `.badges/measurements.json` (#302).

    **Executed** is what an **unpatched** `run_cnaster` runs, on the dev
    instance, under coverage of `cnaster` with `numba` disabled -- so an `@njit`
    kernel's body counts as the lines it is, rather than as one call coverage
    cannot see into. **Patched** is every executed line inside a function that a
    row `run_cnaster_port` installs by default replaces: `SWAPS`,
    `FIGURE_SWAPS` and `SHIFT_SWAPS`. For a class row, only the methods the
    replacement class overrides count; what it inherits is still `cnaster`'s.

    So the figure is the fraction of what a run exercises that no longer runs as
    `cnaster` wrote it. It is not a quality claim and it asserts nothing, so the
    badge is blue.
    """
    import json

    hit, total = share(executed_lines(), patched_lines())
    commit = provenance.head()

    recorded = json.loads(MEASUREMENTS.read_text())
    recorded["patched"] = {
        "label": "patched",
        "percent": round(100.0 * hit / total, 2),
        "patched_lines": hit,
        "executed_lines": total,
        "instance": INSTANCE,
        "commit": commit,
        "note": (
            "Executed cnaster lines inside functions run_cnaster_port replaces "
            "by default (SWAPS, FIGURE_SWAPS, SHIFT_SWAPS; for a class, only "
            "overridden methods), over every cnaster line an unpatched "
            "run_cnaster executes on the dev instance with numba disabled. "
            "Measured by run_benchmark --patched-share (#302); not checked per "
            "pull request, because it needs a whole run."
        ),
    }
    MEASUREMENTS.write_text(json.dumps(recorded, indent=1, ensure_ascii=False) + "\n")

    return hit, total


# --- run_figures --------------------------------------------------------------


STATES = 8
"""What the run fits: the eight states each instance uses of those it plants.

The dev instance plants ten states and its clones use eight; the copy-lattice
instance plants nine and uses eight. At five (#90's figure, set when ten
did not fit) the five planted amplifications could not be separated and
chr7's two events were decoded as one state (#313).
"""


def _write_combined(
    recorded: Recorded, truth: CoreInferenceTruth, root: Path, output: Path
) -> None:
    """The genomic and spatial figures, and both on one page (#309, #339).

    The slide is mocked from the planted labels and read back through
    `port.patch.he.he_image`, as `run_cnaster` reads one (T- #771). It is written
    beside the run's inputs rather than into them: `load_input_data` would
    otherwise find it and refine the initial clones by it, and the figures
    would stop being the ones the dev instance's run draws.
    """
    from port.extensions.combined_figure import (
        combined_figure,
        genomic_figure,
        page_style,
        spatial_figure,
    )
    from port.patch.he import he_image
    from port.patch.utils import write_fig
    from port.pipeline import FIGURE_DPI
    from port.sim.he_slide import mock_he, write_he_slide

    # NB what the run's `FIGURE_SWAPS` row and `--png-copies` bind (#517).
    PAGE: dict[str, Any] = {
        "bbox_inches": None,
        "dpi": FIGURE_DPI,
        "group_rasters": True,
        "png_copy": True,
    }

    slide = mock_he(truth.labels, truth.lattice, seed=truth.seed)
    write_he_slide(slide, root / "slide")
    frame = he_image(str(root / "slide"), res="hires", pos=None)

    plots = next(output.rglob("clones_spatial.pdf")).parent
    # NB at its declared size, not a tight box: the page is drawn at the text
    #    width and included at 1:1, so a box that grows past it is rescaled.
    # NB written as drawn: the run has set seaborn's theme, which a page
    #    written under it would follow where a style is read at draw time.
    with page_style():
        write_fig(str(plots / "genomic.pdf"), genomic_figure(recorded), **PAGE)
        write_fig(
            str(plots / "spatial.pdf"),
            spatial_figure(recorded, frame),
            **PAGE,
        )
        write_fig(
            str(plots / "combined.pdf"),
            combined_figure(recorded, frame),
            **PAGE,
        )


def figures(out: Path, *, cnaster: bool = False) -> list[Path]:
    """The dev instance's figures, and its copy lattice's under `out/lattice`; the directories written.

    It writes the dev instance's inputs through `port.sim.run_config.run_written`, runs
    **`run_cnaster_port`** on them, and copies what it wrote into `DIR`
    (default `port.qa.provenance.PLOTS`, untracked). `--cnaster` runs plain
    `cnaster` instead, for a comparison.

    **Two sets.** `DIR` is the dev instance as planted by default; `DIR/lattice/`
    is the same genome with `copy_lattice=True`, whose states are integer allele
    copies `(A, B)` (`port.sim.truth.COPY_LATTICE`), so its copy-number figures
    can be read against a truth that is integer (#313).

    **CI runs this on every pull request and uploads the result** as a workflow
    artifact (`.github/workflows/figures.yml`), so the figures a pull request's
    code draws can be read beside it. They are not committed: `docs/` tracks
    no PNG outside two exceptions (`tests/test_ci_entry.py`).

    They are not a referee. Nothing here compares a figure against a previous
    one. **They are PNG** (#452): a matplotlib PDF carries a creation timestamp,
    so two runs differ byte for byte with nothing having changed. The run still
    writes its PDFs; `run_cnaster_port --png-copies` has `write_fig` write a PNG
    beside each, without metadata, and those are what is copied. `--cnaster`
    runs `cnaster`'s own `write_fig`, so it copies PDFs, for a local comparison.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    from port.extensions.combined_figure import recording
    from port.sim.run_config import run_written

    written = []
    sets = (
        (dev_instance(), out),
        (
            dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True),
            out / "lattice",
        ),
    )

    for truth, destination in sets:
        root = Path(tempfile.mkdtemp())

        if cnaster:
            output = run_written(
                truth, root, port=False, max_iter_outer=1, max_iter=3, n_states=STATES
            )
        else:
            # NB in process, through `port.scripts.run_cnaster.main`, with its
            #    defaults: the figures are the ones a user of the entry point gets.
            with recording() as recorded:
                output = run_written(
                    truth,
                    root,
                    port=True,
                    max_iter_outer=1,
                    max_iter=3,
                    n_states=STATES,
                    flags=("--png-copies",),
                )

            _write_combined(recorded, truth, root, output)

        destination.mkdir(parents=True, exist_ok=True)
        # NB PDFs are not kept (#452). PNGs are overwritten by name rather
        #    than globbed away: `realizations.png` beside them is written by
        #    another script.
        for stale in destination.glob("*.pdf"):
            stale.unlink()

        suffix = "*.pdf" if cnaster else "*.png"
        drawn = sorted(output.rglob(suffix))
        for figure in drawn:
            shutil.copy(figure, destination / figure.name)

        written.append(destination)
        print(f"wrote {len(drawn)} figures to {destination}")

    return written
