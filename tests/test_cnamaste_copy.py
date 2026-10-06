"""`run_cnamaste` writes what `cnaster` writes with `port`'s absorbed rows (T- #670).

`cnamaste` is `cnaster`'s forward path copied at the pin (PR1), into which
each later T- #670 PR moves `port`'s drop-ins for one stage. This pins what
the copy *does*: `cnaster`'s `run_cnaster`, with `ABSORBED` -- the `port`
rows moved in so far -- installed by `port.pipeline.patched`, and
`run_cnamaste`, run on one written fixture, each from the same global
`numpy` seed (`isolated_run`). Every file either writes -- tables, the fit's
`.npz`, the figures -- is compared byte for byte. **The tolerance is zero.**
The figures compare too because `SOURCE_DATE_EPOCH` fixes the date
matplotlib stamps into a PDF.

The referee is the installed `cnaster`, T- #670's unpatched oracle, under
`port`'s replacements: equality says each moved row computes what `port`'s
does in place, and that the copy is complete and isolated, not that either
is right.

Fixtures, each named by its hash:
- the gate instance (`planted_and_written`'s truth: 2 clones, 3 states,
  25 x 40 spots, 40 bins), `350fbd2b` by `fixture_hash`, one outer and three
  EM iterations;
- dev (`07b82e92`), the `release` tier, five states as
  `test_the_pipeline_completes_on_the_dev_instance` fits it, without
  figures. Unpatched `cnaster` never returns on it (T- #692); with row 15
  moved in (PR3) both arms do;
- CalicoST easy (`2d4ce9a9`, `realization_hash`), the `release` tier, at
  `zenodo_sim_config.yaml`'s settings with the normal-spot BAF interval
  widened to `(0.0, 1.0)`, as `run_config` widens it (#105): at the shipped
  `(0.01, 0.99)` `cnaster`'s own gene table raises `IndexError` on a removed
  bin (`run_cnaster.py:1483`), in both arms alike.
"""

from __future__ import annotations

import gc
import importlib
import logging
import sys
import warnings
from contextlib import ExitStack
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from port.extensions import samples, segments
from port.pipeline import (
    FIGURE_SWAPS,
    LOG_SPACE_SWAPS,
    PLOT_OFF_SWAPS,
    REFINEMENT_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
    Swap,
    patched,
    with_options,
)
from port.sim.fixtures import (
    EASY,
    load_simulated,
    realization_hash,
    references,
    write_sim_inputs,
)
from port.sim.run_config import isolated_run, write_for_run
from port.sim.truth import CoreInferenceTruth, dev_instance, fixture_hash

mpl.use("Agg")

ENTRIES = {"cnaster": "cnaster.scripts.run_cnaster", "cnamaste": "cnamaste.run"}
"""Each package's `run_cnaster`, by package."""

PREPROCESSING = frozenset(
    {
        "port.patch.io:load_input_data",
        "port.patch.io:get_aggregated_barcodes",
        "port.patch.io:get_sample_list",
        "port.patch.omics:form_gene_snp_table",
        "port.patch.omics:assign_initial_blocks",
        "port.patch.omics:summarize_blocks",
        "port.patch.omics:summarize_counts_for_blocks",
        "port.patch.omics:summarize_counts_for_bins",
        "port.patch.omics:create_bin_ranges",
        "port.patch.recomb:get_sitewise_transmat",
        "port.patch.normal_spot:filter_normal_diffexp",
        "port.patch.spatial:lattice_multislice_adjacency",
        "port.patch.spatial:best_equal_partition",
        "port.patch.spatial:initialize_rectangular_clones",
        "port.patch.normal_spot:normal_baf_bin_filter",
        "port.patch.normal_spot:determine_normal_candidates",
        "port.patch.pseudobulk:merge_pseudobulk_by_index_mix",
    }
)
"""PR3's `SWAPS` rows: `docs/port-forward.md` rows 6 and 8-23. Row 7,
`get_reference_genes`, needs `pyarrow`, which `cnamaste` does not declare."""

FIT_CHAIN: tuple[Swap, ...] = (
    *(swap for swap in SWAPS if swap.name == "hmm_phased"),
    *(swap for swap in SHIFT_SWAPS if swap.name == "hmm_nophasing"),
)
"""PR5's rows: `docs/port-forward.md` rows 25 (`SWAPS`) and 26 (`SHIFT_SWAPS`),
the second with the `apply_logmu_shift=True` `port` binds from PR7 on."""

CLONE_ASSIGNMENT: tuple[Swap, ...] = with_options(
    with_options(
        (
            *(
                swap
                for swap in SWAPS
                if swap.name
                in {"compute_loglike_spot_assignment", "pipeline_clone_assignment"}
            ),
            *REFINEMENT_SWAPS,
            *(
                swap
                for swap in SHIFT_SWAPS
                if swap.name in {"run_core_inference", "reindex_clones"}
            ),
        ),
        "port.patch.hmrf:pipeline_clone_assignment",
        label_solver="alpha-rust-fuse-merge",
        floor_merge=True,
        log_space=True,
    ),
    "port.patch.hmrf:run_core_inference",
    distinct_init=True,
)
"""PR7's rows: `docs/port-forward.md` rows 24 and 31-34, with the options
`run_cnaster_port --sal` binds: `--sal`'s solver, the floor merge, the
refinement mask, log space with the shift, and the distinct start."""


ABSORBED: tuple[Swap, ...] = (
    *(swap._replace(options=()) for swap in FIGURE_SWAPS),
    *(swap for swap in SWAPS if swap.replacement in PREPROCESSING),
    *LOG_SPACE_SWAPS,
    *FIT_CHAIN,
    *CLONE_ASSIGNMENT,
)
"""The `port` rows `cnamaste` holds, installed on the `cnaster` arm.

PR2: `docs/port-forward.md` rows 1-4, `FIGURE_SWAPS`, at `cnaster`'s
defaults -- `port` binds `write_fig`'s `dpi=150` and `group_rasters`, and
rows 2 and 4's `axis=Ticks()` (T- #683), which `cnamaste` takes up as
defaults only at PR9. Row 5, plot-off, is
`run_cnaster(..., plots=False)`, against `PLOT_OFF_SWAPS`. PR3:
`PREPROCESSING`, under the segment and sample recording `run_cnaster_port`
enters, as `run_cnamaste` enters its own. PR4: rows 27-30,
`LOG_SPACE_SWAPS`, which `port` installs with its shift and `cnamaste` holds
without it. PR5: `FIT_CHAIN`, rows 25-26 at `cnamaste`'s defaults, the
shift off and the analytic gradient on. PR6 adds no row: `distinct` (#348)
is an option of `gmm_init`, off, which `port` installs with row 33; the
segment floor (#551) is PR3's `create_bin_ranges`, off unless configured,
pinned against `--sal`'s binding below. PR6b adds none either: the `sal`
emission (#425) is `hmm_nophasing`'s `emission_kernels`, and the `sal` and
lattice starts (#489, #540) are `gmm_init`'s `start` and `baf_start`, each
off, which `port` installs with its shift (`test_cnamaste_sal.py`). PR7:
`CLONE_ASSIGNMENT` and the shift, the `port --sal` path for rows 24 and
31-34 without its start and `sal` emission. `cnamaste`'s fixes to #483,
which `port` does not carry, move no byte here: the merges are the same,
and `merge_by_minspots` writes `total_llf` as NaN in both arms
(`hmrf.py:1012`). `test_cnamaste_clones.py` pins them.
"""

NARROWED = "(0.4, 0.6)"
"""A normal-spot BAF interval at which the gate instance loses bins (#105).

At the shipped `(0.01, 0.99)` the filter removes none of the gate instance's
40 bins; at this one it removes 18, and the gene-SNP table logs 80 null
`bin_id`s.
"""

GATE_HASH = "350fbd2b"
DEV_HASH = "07b82e92"
EASY_HASH = "2d4ce9a9"


def _run(package: str, config: Path, root: Path, *, plots: bool = True) -> Path:
    """`package`'s `run_cnaster` on `config`, written under `root / package`.

    `plots=False` is `cnamaste`'s own option, and `PLOT_OFF_SWAPS` over
    `ABSORBED` on the `cnaster` arm.

    Seeded and restored as `isolated_run` does, and `package`'s own global
    configuration restored after, so neither arm reads what the other left.
    The figures it leaves open are closed and collected: on CalicoST easy an
    arm peaks at 11.8 GB and holds 11.8 GB after it returns, 1.2 GB once
    closed, so the second arm would otherwise start where the first peaked.
    """
    document = yaml.safe_load(config.read_text())
    output = root / package
    document["paths"]["output_dir"] = str(output)
    document["paths"]["perf_path"] = str(root / f"{package}.perf")
    own = root / f"{package}.yaml"
    own.write_text(yaml.safe_dump(document))

    entry = importlib.import_module(ENTRIES[package])
    globals_ = importlib.import_module(f"{package}.config")
    saved = globals_._global_config

    swaps = ABSORBED + (() if plots else PLOT_OFF_SWAPS)
    keywords = {"plots": False} if package == "cnamaste" and not plots else {}

    with ExitStack() as stack:
        stack.enter_context(isolated_run())
        stack.enter_context(warnings.catch_warnings())
        if package == "cnaster":
            stack.enter_context(patched(swaps))
            stack.enter_context(segments.recording())
            stack.enter_context(samples.recording())
        warnings.simplefilter("ignore")
        try:
            entry.run_cnaster(str(own), **keywords)
        finally:
            globals_.set_global_config(saved)
            plt.close("all")
            gc.collect()

    return output


def _differ(left: Path, right: Path) -> tuple[set[str], list[str]]:
    """Files only one side wrote, and files both wrote that differ in a byte."""
    files = {
        side: {
            p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()
        }
        for side, root in (("left", left), ("right", right))
    }
    lone = set(files["left"]) ^ set(files["right"])
    differ = sorted(
        name
        for name in set(files["left"]) & set(files["right"])
        if files["left"][name].read_bytes() != files["right"][name].read_bytes()
    )
    return lone, differ


def _equal_runs(config: Path, root: Path, *, plots: bool = True) -> tuple[int, int]:
    """Both arms on `config`; the number of files each wrote, and of `.npz`."""
    cnaster = _run("cnaster", config, root, plots=plots)
    cnamaste = _run("cnamaste", config, root, plots=plots)

    lone, differ = _differ(cnaster, cnamaste)
    assert not lone, f"written by one arm only: {sorted(lone)}"
    assert not differ, f"differ: {differ}"

    written = [p for p in cnaster.rglob("*") if p.is_file()]
    return len(written), sum(p.suffix == ".npz" for p in written)


@pytest.fixture
def _fixed_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_writes_absorbed_cnasters_bytes_on_the_gate_instance(
    planted_instance: tuple[CoreInferenceTruth, object, object, Path],
    tmp_path: Path,
) -> None:
    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    _, config = write_for_run(truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3)

    files, fits = _equal_runs(config, tmp_path)

    # NB the tables, the fits and the 19 figures, so equality is not vacuous
    assert files >= 25
    assert fits >= 1


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_writes_absorbed_cnasters_bytes_on_the_dev_instance(
    tmp_path: Path,
) -> None:
    """T- #692: both arms return, row 15's initializer redrawing where
    `cnaster`'s loops. Without figures: at `cnaster`'s 300 dpi the first
    genomic figure takes either arm past this host's 14 GB (PR3)."""
    truth = dev_instance()
    assert fixture_hash(truth) == DEV_HASH
    _, config = write_for_run(
        truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3, n_states=5
    )

    files, fits = _equal_runs(config, tmp_path, plots=False)

    assert files >= 6
    assert fits >= 1


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_writes_absorbed_cnasters_bytes_on_calicost_easy(
    tmp_path: Path,
) -> None:
    """CalicoST's shipped sample at `zenodo_sim_config.yaml`'s settings."""
    if references() is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    sample = load_simulated(EASY)
    assert realization_hash(sample.path) == EASY_HASH

    # NB widened as `run_config` widens it (#105); see the module docstring.
    widened = {"quality.normal_allele_specific_confidence": "(0.0, 1.0)"}
    config = write_sim_inputs(sample, tmp_path / "inputs", widened)

    files, fits = _equal_runs(config, tmp_path)

    assert files >= 25
    assert fits >= 1


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_without_plots_writes_what_ports_plot_off_writes(
    planted_instance: tuple[CoreInferenceTruth, object, object, Path],
    tmp_path: Path,
) -> None:
    """Row 5: `plots=False` against `port`'s `PLOT_OFF_SWAPS`, byte for byte.

    The tables and the fit only: no figure is written on either side.
    """
    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    _, config = write_for_run(truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3)

    files, fits = _equal_runs(config, tmp_path, plots=False)

    written = {p.suffix for p in (tmp_path / "cnamaste").rglob("*") if p.is_file()}
    assert ".pdf" not in written
    assert files >= 6
    assert fits >= 1


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_a_bin_the_normal_baf_filter_removes_leaves_its_genes_out(
    planted_instance: tuple[CoreInferenceTruth, object, object, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#105: `cnaster` indexes with a removed bin's null `bin_id`; `cnamaste` drops it.

    The gate instance at `NARROWED`: 18 of 40 bins removed. `cnaster` raises
    `IndexError` at the gene-level table (`run_cnaster.py:1483`). `cnamaste`
    completes: its gene-level table is the widened run's 113 genes less 51,
    each with integer copies, where `cnaster` wrote none. (Less 45 at PR3:
    PR4's log-space kernels move this run's normal-spot fit.)
    """
    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    _, config = write_for_run(truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3)
    document = yaml.safe_load(config.read_text())
    document["quality"]["normal_allele_specific_confidence"] = NARROWED
    narrowed = tmp_path / "narrowed.yaml"
    narrowed.write_text(yaml.safe_dump(document))

    for side in ("narrowed", "widened"):
        (tmp_path / side).mkdir()

    # NB `cnaster` alone: from PR3 `ABSORBED` carries `port`'s fix too.
    with monkeypatch.context() as unpatched:
        unpatched.setattr(sys.modules[__name__], "ABSORBED", ())
        with pytest.raises(IndexError, match="out of bounds"):
            _run("cnaster", narrowed, tmp_path / "narrowed")

    def genes(run: Path, root: Path) -> pd.DataFrame:
        (table,) = _run("cnamaste", run, root).rglob("cnv_genelevel.tsv")
        return pd.read_csv(table, sep="\t", index_col=0)

    lost = genes(narrowed, tmp_path / "narrowed")
    kept = genes(config, tmp_path / "widened")

    assert set(lost.index) < set(kept.index)
    assert (len(kept), len(lost)) == (113, 62)
    assert not lost.isna().to_numpy().any()
    assert all(dtype.kind == "i" for dtype in lost.dtypes)


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_completes_calicost_easy_at_the_shipped_interval(
    tmp_path: Path,
) -> None:
    """#105 on CalicoST's own sample: `cnaster` raises `IndexError` at the
    shipped normal-spot BAF interval, `(0.01, 0.99)` (PR- #689); `cnamaste`
    writes its gene-level table, with no gene of a removed bin in it."""
    if references() is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    sample = load_simulated(EASY)
    assert realization_hash(sample.path) == EASY_HASH
    config = write_sim_inputs(sample, tmp_path / "inputs")
    document = yaml.safe_load(config.read_text())
    assert document["quality"]["normal_allele_specific_confidence"] == "(0.01, 0.99)"

    (table,) = _run("cnamaste", config, tmp_path).rglob("cnv_genelevel.tsv")
    genes = pd.read_csv(table, sep="\t", index_col=0)

    assert len(genes) > 0
    assert not genes.isna().to_numpy().any()


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_the_log_space_kernels_keep_devs_clones_and_move_its_fit_by_under_1e_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR4's kernels alone, on dev: `cnaster` with every other absorbed row.

    Every spot keeps its clone; each fitted state parameter moves by under
    1e-8 relative (measured 1.2e-9, `new_log_mu`). At PR4, under `cnaster`'s
    finite-difference M step, the same swap moved it by 1.6e-3; PR5's
    analytic gradient is the difference (#244).
    """
    truth = dev_instance()
    assert fixture_hash(truth) == DEV_HASH
    _, config = write_for_run(
        truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3, n_states=5
    )
    previous = tuple(swap for swap in ABSORBED if swap not in LOG_SPACE_SWAPS)

    with monkeypatch.context() as cnasters_kernels:
        cnasters_kernels.setattr(sys.modules[__name__], "ABSORBED", previous)
        before = _run("cnaster", config, tmp_path, plots=False)
    after = _run("cnamaste", config, tmp_path, plots=False)

    def read(root: Path, pattern: str) -> Path:
        (found,) = root.rglob(pattern)
        return found

    labels = [
        pd.read_csv(read(r, "clone_labels.tsv"), sep="\t") for r in (before, after)
    ]
    pd.testing.assert_frame_equal(*labels)

    fits = [np.load(read(r, "*.npz")) for r in (before, after)]
    for name in ("new_log_mu", "new_alphas", "new_p_binom", "new_taus"):
        np.testing.assert_allclose(fits[1][name], fits[0][name], rtol=1e-8, atol=0)


@pytest.mark.infra
@pytest.mark.parametrize(
    "keywords",
    [{"flags": ["--sal"]}, {"flags": [], "likelihood": True}],
    ids=["flags", "likelihood"],
)
def test_the_audits_cnamaste_arm_refuses_what_only_port_reads(
    keywords: dict[str, object],
) -> None:
    """`run_audit --recovery --cnamaste` runs `cnamaste`'s `run_cnaster`, which
    takes no flags; `--likelihood` reads port's fit. Both are refused before
    any input is written."""
    from port.qa.audit import audit_truth
    from port.sim.truth import critical_instance

    with pytest.raises(ValueError, match="run_cnamaste takes no flags"):
        audit_truth(critical_instance(), cnamaste=True, **keywords)  # type: ignore[arg-type]


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_the_segment_floor_by_configuration_writes_what_sals_binding_writes(
    planted_instance: tuple[CoreInferenceTruth, object, object, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 300 normal-UMI segment floor (#551), `--sal`'s alone (T- #667).

    `run_cnaster_port --sal` binds `min_segment_normal_umi=300` into
    `create_bin_ranges`; `cnamaste` has no flags, and its floor is the
    configuration's `quality.min_segment_normal_umi: true`. On the gate
    instance, without figures, the two write the same bytes, and the floor
    takes the segments from 40 to 38 (T- #670 PR6).
    """
    from port.patch.omics.blocks import MIN_SEGMENT_NORMAL_UMI

    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    _, config = write_for_run(truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3)
    document = yaml.safe_load(config.read_text())
    assert "min_segment_normal_umi" not in document["quality"]
    document["quality"]["min_segment_normal_umi"] = True
    floored = tmp_path / "floored.yaml"
    floored.write_text(yaml.safe_dump(document))

    bound = with_options(
        ABSORBED,
        "port.patch.omics:create_bin_ranges",
        min_segment_normal_umi=MIN_SEGMENT_NORMAL_UMI,
    )
    with monkeypatch.context() as sal:
        sal.setattr(sys.modules[__name__], "ABSORBED", bound)
        cnaster = _run("cnaster", config, tmp_path, plots=False)
    cnamaste = _run("cnamaste", floored, tmp_path, plots=False)

    lone, differ = _differ(cnaster, cnamaste)
    assert not lone, f"written by one arm only: {sorted(lone)}"
    assert not differ, f"differ: {differ}"

    (table,) = cnamaste.rglob("cnv_seglevel.tsv")
    assert len(pd.read_csv(table, sep="\t")) == 38


class _Kept(logging.Handler):
    """Keeps every record it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_the_integer_copy_log_says_log_mu_is_not_normalized(
    planted_instance: tuple[CoreInferenceTruth, object, object, Path],
    tmp_path: Path,
) -> None:
    """#136: `cnaster` logs that it normalized `log_mu` before the integer copy
    decoder, and prints one array as both "new" and "given"; the normalization
    is commented out. `cnamaste` logs that the decoder reads it unnormalized.
    The gate instance, without figures: one message per clone per ploidy."""
    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    _, config = write_for_run(truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3)

    def messages(package: str) -> list[str]:
        # NB imported first: `get_logger` clears the handlers of the logger
        #    it configures, so one added before the import is dropped.
        logger = logging.getLogger(importlib.import_module(ENTRIES[package]).__name__)
        handler = _Kept()
        records = handler.records
        logger.addHandler(handler)
        try:
            _run(package, config, tmp_path, plots=False)
        finally:
            logger.removeHandler(handler)
        return [
            record.getMessage()
            for record in records
            if record.getMessage().startswith("For clone ")
            and "mu" in record.getMessage()
        ]

    claimed = messages("cnaster")
    stated = messages("cnamaste")

    assert claimed
    for message in claimed:
        assert "normalized log mu to sum_bin lambda * np.exp(log_mu) = 1." in message
        new, given = message.split("yielding new mu=\n")[1].split("\ngiven mu=\n")
        assert new == given.removesuffix(".")

    assert len(stated) == len(claimed)
    assert all("not normalized" in message for message in stated)
