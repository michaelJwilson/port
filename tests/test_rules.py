"""`port`'s repository rules: one scan of the tree, one row per rule (#852).

A `Rule` names what it reads, a check returning what it finds, and the
exceptions it declares. It fails on a finding it does not declare, and on a
declared exception it no longer finds, so a declaration only shrinks. A rule
reading `COLLECTION` gets the whole suite's items and skips where the
collection is narrowed. Sources are read and parsed once a session
(`tests.source_graph.text`, `parse`).
"""

from __future__ import annotations

import ast
import configparser
import datetime
import hashlib
import importlib
import importlib.util
import inspect
import json
import pkgutil
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import cnaster
import port.patch
import port.scripts.run_cnaster as entry
import pytest
import sal
from _pytest.mark.expression import Expression
from port.extensions import cnamaste
from port.extensions.figure_style import DEFAULT, stated
from port.pipeline import (
    COPY_SWAPS,
    DEFAULTS,
    FIGURE_SWAPS,
    LOG_SPACE_SWAPS,
    REFINEMENT_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
    Swap,
    swap_sites,
)
from port.qa import ledger
from port.qa.ledger import CONVERTED, METRICS, TEST, TIMESTAMP, check_note
from port.qa.vocabulary import TERMS, replaced_by
from port.scripts.run_cnaster import Settings
from port.sim import truth as sim_truth
from port.sim.draw import extended
from port.sim.fixtures import R0_HASH, SIM_ROOT
from port.studies.paper_figures import KEY_STUDIES, QUESTIONS
from port.studies.paper_figures import OUT as PAPER

from scripts import ci, port_forward
from scripts.badges import BADGES, MEASUREMENTS, UNMEASURED, badges
from tests import ROOT
from tests.metrics import fixture_hash
from tests.source_graph import (
    COUNTING,
    PACKAGE,
    SANDBOX_TESTS,
    TESTS,
    counting_mentions,
    modules,
    parse,
    reached,
    row_modules,
    state_writes,
    tables,
    text,
)
from tests.test_sim_r0_hash import HASHED, _stated

Found = Iterable[str] | Mapping[str, object]
COLLECTION = "the collected suite"
"""What a rule over test items reads; it skips on a narrowed collection."""


@dataclass(frozen=True)
class Rule:
    """One repository rule: what it reads, its check, and its declared exceptions."""

    name: str
    reads: str
    check: Callable[..., Found]
    declared: Collection[str] = frozenset()


RULES: list[Rule] = []


def rule(
    name: str, reads: str, declared: Collection[str] = frozenset()
) -> Callable[[Callable[..., Found]], Callable[..., Found]]:
    """Register the decorated check as the rule `name`."""

    def register(check: Callable[..., Found]) -> Callable[..., Found]:
        RULES.append(Rule(name, reads, check, declared))
        return check

    return register


def files(root: Path, pattern: str = "*.py", skip: Collection[str] = ()) -> list[Path]:
    """`root`'s files matching `pattern`, none with a part under `root` in `skip`."""
    return [
        path
        for path in sorted(root.rglob(pattern))
        if not set(skip) & set(path.relative_to(root).parts)
    ]


def installed(package: Any) -> Path:
    """Where a (namespace) package is installed."""
    return Path(next(iter(package.__path__)))


@cache
def pyproject() -> dict[str, Any]:
    return tomllib.loads(text(ROOT / "pyproject.toml"))


@cache
def oracle_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(ROOT / ".coveragerc-oracle")
    return parser


def git_files(*arguments: str) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", *arguments], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [name for name in listed.split("\0") if name]


def markers(item: pytest.Item) -> set[str]:
    return {mark.name for mark in item.iter_markers()}


# --- module roles and the import graph (#517 E3, T- #673) ---------------------

Role = Literal[
    "row", "row-helper", "extension", "oracle", "tool", "sim", "script", "pipeline", "set aside"
]  # fmt: skip

ROLE_MODULES: dict[Role, str] = {
    "pipeline": "pipeline",
    "script": """
        scripts.run_calicost scripts.run_cnaster scripts.run_plots
        qa.scripts.run_audit qa.scripts.run_benchmark qa.scripts.run_calibrate
        qa.scripts.run_figures qa.scripts.run_ledger qa.scripts.run_study
    """,
    "extension": """
        extensions.adjacency extensions.cnamaste extensions.combined_figure
        extensions.config_audit extensions.copy_likelihood extensions.copy_starts
        extensions.figure_record extensions.figure_style extensions.genomic_axis
        extensions.integer_copy extensions.label_solver extensions.multisample
        extensions.outputs extensions.repository extensions.run_record extensions.sal
        extensions.samples extensions.segments extensions.spatial_page
    """,
    "oracle": "qa.emission_family qa.kronecker_posteriors",
    # NB qa: run measurement and records; studies: run by hand from `run_study` (T- #673)
    "tool": """
        qa.audit qa.benchmark qa.cnaster_arm qa.errors qa.jax_hmm qa.jax_setup
        qa.ledger qa.parameter_errors qa.provenance qa.realization_plot qa.records
        qa.scoring qa.stage qa.statistics qa.stream qa.vocabulary
        studies.benchmark_table studies.calicost_figures studies.clone_label_arms
        studies.clone_label_notebook studies.clone_labels studies.clone_starts
        studies.cna_lengths studies.copy_start_arms studies.copy_start_notebook
        studies.copy_starts studies.copy_state_plot studies.copy_state_stream
        studies.field_strength studies.figures studies.metrics_history
        studies.notebook studies.paper_figures studies.population
        studies.population_report studies.potts_plot studies.potts_stream
    """,
    "row": """
        patch.count_encoder patch.he patch.hmm_nophasing.bb_logpmf
        patch.hmm_nophasing.nb_logpmf patch.hmm_nophasing.shifted_emission
        patch.hmm_phased.coded_emission patch.hmrf.clone_assignment
        patch.hmrf.core_inference patch.hmrf.field patch.hmrf.refinement
        patch.integer_copy patch.io patch.normal_spot patch.omics.blocks
        patch.omics.summaries patch.plot_copy_number_profile patch.plot_genomic
        patch.plotting.spatial patch.pseudobulk patch.recomb patch.reference
        patch.spatial patch.utils
    """,
    "row-helper": """
        patch._clone_paths patch._signature patch.emission
        patch.hmm_initialize.distinct patch.hmm_initialize.sal_mixture
        patch.hmm_nophasing.dense_emission patch.hmm_nophasing.gradient
        patch.hmm_nophasing.logmu_shift patch.hmrf.adjacency patch.hmrf.fused_field
        patch.hmrf.invariants patch.hmrf.reindex patch.hmrf.tabulated_field
        patch.icm.alpha_expansion patch.icm.floor patch.icm.interface patch.lattice
    """,
    "sim": """
        sim.analysis sim.draw sim.entries sim.files sim.fixtures sim.he_slide
        sim.inputs sim.kernels sim.laws sim.normal_fit sim.realizations
        sim.run_config sim.truth sim.truth_figure sim.unsegment
    """,
    "set aside": """
        sandbox.admixture.clone_mixture sandbox.admixture.probes.sim_probe
        sandbox.admixture.variants sandbox.clone_starts.problem
        sandbox.clone_starts.starts sandbox.copy_audit sandbox.extensions.copy_errors
        sandbox.extensions.copy_starts sandbox.extensions.hmm_init_trials
        sandbox.extensions.hmm_objective sandbox.extensions.label_solvers
        sandbox.extensions.segment_sets sandbox.extensions.shared_decode
        sandbox.integer_decoding.calicost_decoders sandbox.integer_decoding.rdr_summary
        sandbox.integer_decoding.schemes sandbox.normal_candidates sandbox.np_merge
        sandbox.np_merge.__main__ sandbox.np_merge.merge sandbox.patch.emission
        sandbox.patch.hmm_initialize.backends sandbox.patch.hmm_initialize.filtering
        sandbox.patch.plotting.genomic sandbox.patch.plotting.loh_density
        sandbox.population_sets sandbox.sal_hmm_init sandbox.sim_from_run
        sandbox.wolff_init sandbox.wolff_umi_init
    """,
}
"""Every module under `port.` by role; a package `__init__` counts once it defines something."""

ROLES: dict[str, Role] = {
    f"port.{name}": role
    for role, names in ROLE_MODULES.items()
    for name in names.split()
}

WHERE: dict[Role, tuple[str, ...]] = {
    "row": ("port.patch.",),
    "row-helper": ("port.patch.",),
    "extension": ("port.extensions.",),
    "oracle": ("port.extensions.", "port.qa."),
    "tool": ("port.extensions.", "port.qa.", "port.studies."),
    "sim": ("port.sim.",),
    "script": ("port.scripts.", "port.qa.scripts."),
    "pipeline": ("port.pipeline",),
    "set aside": ("port.sandbox.",),
}

PIPELINE = frozenset({"port.scripts.run_calicost", "port.scripts.run_cnaster"})
"""The pipeline entry points; every other `script` is a QA tool (T- #673)."""

SANDBOX_HEADERLESS: frozenset[str] = frozenset()
"""Sandbox modules without the header: none since #517 step 8."""

HEADER = ("Ticket:", "Measurement:", "Exit:")
"""What a sandbox module's docstring states, one line each."""


def by(role: str) -> set[str]:
    return {name for name, declared in ROLES.items() if declared == role}


@rule("module-roles", "python/port", declared=ROLES)
def _module_roles() -> Found:
    """Modules with a role: any module, or an `__init__` defining something."""
    return {
        name
        for name, path in modules().items()
        if path.name != "__init__.py"
        or any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            for node in parse(path).body
        )
    }


@rule("module-placement", "ROLES")
def _module_placement() -> Found:
    return {
        name: f"{role} lives under {WHERE[role]}"
        for name, role in ROLES.items()
        if not name.startswith(WHERE[role])
    }


@rule("module-rows", "the swap tables", declared=frozenset(by("row")))
def _module_rows() -> Found:
    """The modules the swap tables install from are the rows, both ways (#517 E)."""
    return row_modules()


@rule("module-live", "the import graph")
def _module_live() -> Iterator[str]:
    """`extension` and `row-helper` are reached from live roots; others are not."""
    live = reached({"port.pipeline"} | PIPELINE | row_modules())
    yield from (f"entry point not a script: {name}" for name in PIPELINE - by("script"))
    for role in ("extension", "row-helper"):
        yield from (f"{role} not live: {name}" for name in by(role) - live)
    # NB an oracle the run reaches is not independent of it (#749 WP8)
    for role in ("tool", "oracle", "set aside"):
        yield from (f"{role} live: {name}" for name in by(role) & live)


@rule("oracle-referees", "tests/test_*.py")
def _oracle_referees() -> Found:
    """Every oracle module is named by an end-to-end or oracle test file."""
    referred: set[str] = set()
    # NB a test of set-aside code referees nothing live (#851)
    for path in files(TESTS, "test_*.py", skip={SANDBOX_TESTS.name}):
        if any(f"mark.{marker}" in text(path) for marker in COUNTING):
            referred |= {name for name in by("oracle") if name in text(path)}
    return by("oracle") - referred


@rule("sandbox-header", "python/port/sandbox", declared=SANDBOX_HEADERLESS)
def _sandbox_header() -> Iterator[str]:
    """Sandbox modules whose docstring misses a `HEADER` line."""
    for name in by("set aside"):
        lines = (ast.get_docstring(parse(modules()[name])) or "").splitlines()
        if not all(any(line.startswith(key) for line in lines) for key in HEADER):
            yield name


@rule("sandbox-imports", "python/port")
def _sandbox_imports() -> Iterator[str]:
    """No module outside `sandbox/` and `studies/` imports the sandbox (T- #617, #673)."""
    for path in files(PACKAGE, skip={"sandbox", "studies"}):
        for node in ast.walk(parse(path)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            yield from (
                f"{path.relative_to(PACKAGE)}: {name}"
                for name in names
                if name == "port.sandbox" or name.startswith("port.sandbox.")
            )


@rule("module-jobs", "python/port")
def _module_jobs() -> Iterator[str]:
    """Only the job directories under `port/`, and no stray top-level module (T- #673)."""
    allowed = {"patch", "extensions", "sim", "scripts", "sandbox", "qa", "studies"}
    # NB `pipeline.py` is the swap table and its installer
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name not in {"__init__.py", "pipeline.py"}:
            yield f"{path.name} claims none of the four jobs"
    for child in sorted(p.name for p in PACKAGE.iterdir() if p.is_dir()):
        if not child.startswith("__") and child not in allowed:
            yield f"port/{child}/ is not one of {sorted(allowed)}"


ADMITTED: dict[str, str] = {
    "port.patch.hmm_nophasing.nb_logpmf:_nb_logpmf_1d": (
        "cnaster's own name, which the row replaces"
    ),
}
"""Admitted cross-module private imports, each with why (`docs/port-forward.md`); only shrinks."""


@rule("private-imports", "python/port", declared=ADMITTED)
def _private_imports() -> Found:
    """`port` modules importing another's `_name`, `sandbox/` included (#763)."""
    found: dict[str, list[str]] = {}
    for path in files(PACKAGE):
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "port"
            ):
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        where = str(path.relative_to(PACKAGE.parent))
                        found.setdefault(f"{node.module}:{alias.name}", []).append(
                            where
                        )
    return found


READS: dict[str, str] = {
    "python/port/extensions/adjacency.py": "PORT_ADJACENCY and PORT_SQUARE_NEIGHBOURHOOD select the lattice construction for a subprocess arm (#417)",
    "python/port/extensions/label_solver.py": "PORT_LABEL_SOLVER overrides the bound solver for a benchmark arm (#246)",
    "python/port/qa/benchmark.py": "the patched share's child inherits the environment with NUMBA_DISABLE_JIT set (moved from tests/, T- #673 G4)",
    "python/port/qa/audit.py": "PORT_SIM_CACHE keeps a cropped or purified sample between runs (moved from tests/, T- #673 G3)",
    "python/port/sim/draw.py": "PORT_CACHE locates the simulator's map cache",
    "python/port/studies/population.py": "pins each member's thread pools to one thread (moved from tests/, T- #673 G5)",
    "python/port/sim/fixtures.py": "PORT_GRCH38 locates CalicoST's GRCh38 resources for the committed samples (moved from tests/, T- #673 G6)",
}  # fmt: skip
"""Each file that reads the environment outside `sandbox/`, and why (T- #617, rule 1)."""


@rule("environment-reads", "python/port", declared=READS)
def _environment_reads() -> Found:
    pattern = re.compile(r"\bos\.environ\b|\bgetenv\(")
    return {
        str(path.relative_to(ROOT))
        for path in files(PACKAGE, skip={"sandbox"})
        if pattern.search(text(path))
    }


Kind = Literal["switch", "run", "cache", "rebind"]

STATE: dict[str, Kind] = {
    # NB studies run by hand, reached from no entry point (T- #673 G5).
    "port.extensions.label_solver.sweep_for": "rebind",
    "port.patch.hmrf.core_inference.UPSTREAM": "rebind",
    "port.patch.normal_spot.determine_normal_candidates": "rebind",
    "port.studies.clone_label_arms.HELD": "run",
    "port.studies.clone_label_arms.potts_graph": "rebind",
    "port.studies.copy_start_arms._CALLS": "run",
    "port.studies.copy_state_stream._WARM": "cache",
    "port.studies.potts_stream._GRAPHS": "cache",
    # NB `audit_truth` restores the name in its `finally` (T- #673 G3).
    "cnaster.scripts.run_cnaster.determine_normal_candidates": "rebind",
    "cnaster.hmm_initialize.GaussianMixture": "rebind",
    # NB #735: `port.qa.stage` wraps one `run_core_inference` call and restores it.
    "cnaster.hmrf.pipeline_clone_assignment": "rebind",
    "port.extensions.copy_likelihood._FITS": "run",
    "port.extensions.cnamaste._ACTIVE": "run",
    "port.extensions.run_record._HELD": "run",
    "port.extensions.samples._CURRENT": "run",
    "port.extensions.segments._CURRENT": "run",
    "port.patch.hmm_nophasing.shifted_emission.hmm_nophasing._row_shift": "run",
    "port.patch.hmrf.clone_assignment._BOUNDARY": "cache",
    "port.patch.hmrf.core_inference._NORMAL": "run",
    "port.patch.hmrf.core_inference._PROPAGATED": "run",
    "port.patch.hmrf.refinement._KEPT": "run",
    "port.patch.hmrf.run_core_inference": "rebind",
    "port.patch.integer_copy._RECORDERS": "run",
    "port.patch.integer_copy._SHARED": "cache",
    "port.patch.io.NORMAL_SPOTS": "run",
}
"""Every name `port` writes after import, by kind; no `switch` since #517 steps 1-2.
`tests/test_run_state.py` holds a whole run to it (#517 E2)."""

OUTLIVES: frozenset[str] = frozenset()
"""What a whole run leaves changed: nothing since #517 step 1."""


@rule("state-writes", "python/port", declared=STATE)
def _state_writes() -> Found:
    return state_writes()


@rule("state-outlives", "STATE")
def _state_outlives() -> Found:
    return OUTLIVES - set(STATE)


COUNTS: dict[str, int] = {
    # NB 22 since T- #831 set aside four flags no `--sal` run takes
    "run_cnaster_port flags": 22,
    "classes": 91,
    # NB dataclasses carry mutable state, machinery or `__post_init__` (#517 D)
    "dataclasses": 33,
    "NamedTuples": 40,
    # NB labels by overlap (`qa.scoring.matched`); states by responsibility distance
    "Hungarian matcher": 2,
    # NB in-memory instance vs sample on disk (T- #673 G3)
    "audit arm": 2,
    "clone_path": 1,
    # NB `combined_figure`, `segments`, `samples` (T- #418)
    "recording": 3,
    # NB 1 each since T- #673 G1
    "commit reader": 1,
    "peak memory reader": 1,
}
"""Flags and classes in `python/port` outside `sandbox/`, then definitions of one concept
across `python/port` and `tests/`, with the reason where not 1 (#517 E4)."""


def referenced(node: ast.AST) -> set[str]:
    return {
        sub.id if isinstance(sub, ast.Name) else sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name | ast.Attribute)
    }


def literal_arguments(node: ast.AST) -> set[str]:
    """String literals passed positionally to calls in `node`, alone or in a list."""
    return {
        item.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        for arg in sub.args
        for item in (arg.elts if isinstance(arg, ast.List | ast.Tuple) else [arg])
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


@rule("one-implementation", "python/port, tests")
def _one_implementation() -> Found:
    """Each count equals its declared value; a change moves it with its reason in the PR."""
    port_files = [p for p in sorted(PACKAGE.rglob("*.py")) if "sandbox" not in p.parts]
    classes = [
        node
        for path in port_files
        for node in ast.walk(parse(path))
        if isinstance(node, ast.ClassDef)
    ]
    functions = [
        node
        for path in [*port_files, *sorted(TESTS.rglob("*.py"))]
        for node in ast.walk(parse(path))
        if isinstance(node, ast.FunctionDef)
    ]
    entry_point = parse(PACKAGE / "scripts" / "run_cnaster.py")
    measured = {
        "run_cnaster_port flags": sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for node in ast.walk(entry_point)
        ),
        "classes": len(classes),
        "dataclasses": sum(
            any("dataclass" in ast.unparse(d) for d in c.decorator_list)
            for c in classes
        ),
        "NamedTuples": sum(
            any(ast.unparse(b).endswith("NamedTuple") for b in c.bases) for c in classes
        ),
        "Hungarian matcher": sum(
            "linear_sum_assignment" in referenced(f) for f in functions
        ),
        "audit arm": sum(f.name in {"audit_sample", "audit_truth"} for f in functions),
        "clone_path": sum(f.name == "clone_path" for f in functions),
        "recording": sum(f.name == "recording" for f in functions),
        "commit reader": sum("rev-parse" in literal_arguments(f) for f in functions),
        "peak memory reader": sum("ru_maxrss" in referenced(f) for f in functions),
    }
    return {
        name: f"{COUNTS[name]} -> {measured[name]}"
        for name in COUNTS
        if measured[name] != COUNTS[name]
    }


@rule("aim-extra", "pyproject.toml, python/port")
def _aim_extra() -> Iterator[str]:
    """`aim` is a declared extra and no `python/port/` module imports it at module scope (#251)."""
    project = pyproject()["project"]
    if not any(
        name.startswith("aim") for name in project["optional-dependencies"]["track"]
    ):
        yield "the `track` extra must pin aim"
    if any(name.startswith("aim") for name in project["dependencies"]):
        yield "aim is an extra, not a dependency"
    for path in PACKAGE.rglob("*.py"):
        for number, line in enumerate(text(path).splitlines(), start=1):
            if line.startswith(("import aim", "from aim")):
                yield f"module-scope aim import: {path.relative_to(ROOT)}:{number}"


# --- API vocabulary and signatures (`CLAUDE.md`, #401) -------------------------

KNOWN: dict[str, str] = {
    # --- F1, F8, F10: the label-solver seam ---------------------------------
    "port.patch.icm.interface:IcmResult field cost": "F1",
    "port.patch.icm.interface:IcmResult field niter": "F10",
    # NB its keyword-only `backend` is the sal option the rows bind
    #    (`Backend.RUST`); the seam's other arguments match `icm_sweep`.
    "port.patch.icm.alpha_expansion:alpha_expansion_sweep sibling": "F8",
    # --- F2, F3, F5: count-kernel and data names ----------------------------
    "port.patch.normal_spot:cumulative_and_mass arg alpha": "F3",
    "port.patch.normal_spot:cumulative_and_mass arg beta": "F2",
    "port.patch.normal_spot:removal_indicator arg alpha": "F3",
    "port.patch.normal_spot:removal_indicator arg beta": "F2",
    "port.extensions.integer_copy:success_probability_variance arg alpha": "F3",
    "port.extensions.integer_copy:success_probability_variance arg beta": "F2",
    "port.qa.parameter_errors:shift_weights arg log_mus": "F5",
    "port.patch.hmm_nophasing.logmu_shift:shifts arg log_mus": "F5",
    "port.patch.hmrf.core_inference:pin_neutral arg result": "F6",
    # --- T- #673 G6: the simulation machinery, moved from `tests/` ----------
    # NB renaming `CoreInferenceTruth.seed` moves every recorded dev fixture (`07b82e92`).
    "port.sim.truth:CoreInferenceTruth field seed": "G6",
    "port.sim.truth:core_inference_truth arg seed": "G6",
    "port.sim.he_slide:mock_he arg seed": "G6",
    "port.sim.fixtures:purify arg seed": "G6",
    "port.sim.run_config:run_cnaster_config arg max_iter": "G6",
    # NB G3: audits and realizations; `max_iter` is `run_cnaster_config`'s.
    "port.qa.audit:audit_truth arg max_iter": "G3",
    "port.qa.audit:audit_errors arg seed": "G3",
    "port.sim.realizations:realize arg seed": "G3",
    "port.sim.realizations:chosen arg seed": "G3",
    # NB G5: the studies; a job's `seed` is a field of pickled records.
    "port.studies.copy_state_stream:solve_start arg seed": "G5",
    "port.studies.potts_stream:solve_labelling arg seed": "G5",
    "port.studies.clone_label_arms:Job field seed": "G5",
    "port.studies.copy_start_arms:Job field seed": "G5",
    "port.studies.population:run_member arg seed": "G5",
    "port.studies.population:draw_member arg seed": "G5",
    "port.studies.population:stay_member arg seed": "G5",
    "port.studies.population_report:summarize arg seed": "G5",
}
"""Departures found by #401's audit, keyed `module:name kind word`; only shrinks."""

SIBLINGS: dict[str, tuple[str, ...]] = {
    "port.patch.icm.interface:icm_sweep": (
        "port.patch.icm.alpha_expansion:alpha_expansion_sweep",
        "port.extensions.label_solver:sal_icm_sweep",
        "port.extensions.label_solver:fusion_then_merge",
    ),
}
"""Entry points one setting chooses between: same arguments, order and result as the first."""

ITERATION_FIELDS = {"niter", "iterations", "n_iter", "passes", "cycles", "converged"}
"""A result carrying one of these reports an iterative run."""

SKIPPED = {"sandbox", "deprecated", "tests", "__pycache__"}


Entry = tuple[str, ast.FunctionDef | ast.ClassDef]


@cache
def defined() -> tuple[Entry, ...]:
    """Every public top-level callable and class under `python/port`, keyed `module:name`."""
    out = []
    for path in files(PACKAGE, skip=SKIPPED):
        module = ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts)
        for node in parse(path).body:
            if isinstance(
                node, ast.FunctionDef | ast.ClassDef
            ) and not node.name.startswith("_"):
                out.append((f"{module}:{node.name}", node))
    return tuple(out)


@cache
def owned() -> tuple[Entry, ...]:
    """What `port` owns: every public name bar the drop-ins, which `cnaster` defines too."""
    upstream = {
        node.name
        for path in files(installed(cnaster), skip=SKIPPED)
        for node in ast.walk(parse(path))
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    return tuple((key, node) for key, node in defined() if node.name not in upstream)


def arguments(node: ast.FunctionDef) -> list[ast.arg]:
    every = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    return [a for a in every if a.arg not in {"self", "cls"}]


def fields(node: ast.ClassDef) -> list[str]:
    return [
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    ]


def words() -> Iterator[tuple[str, str]]:
    """`(key, word)` for every argument and field name `port` owns."""
    for where, node in owned():
        if isinstance(node, ast.FunctionDef):
            yield from ((f"{where} arg {a.arg}", a.arg) for a in arguments(node))
            continue
        yield from ((f"{where} field {field}", field) for field in fields(node))
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and not method.name.startswith("_"):
                for a in arguments(method):
                    yield f"{where}.{method.name} arg {a.arg}", a.arg


def shape(node: ast.FunctionDef) -> tuple[tuple[str, ...], str | None]:
    """Arguments in order and the result annotation, what siblings share."""
    names = tuple(a.arg for a in arguments(node))
    variadic = tuple(
        f"*{a.arg}" for a in (node.args.vararg, node.args.kwarg) if a is not None
    )
    return names + variadic, ast.unparse(node.returns) if node.returns else None


@rule("vocabulary-terms", "port.qa.vocabulary")
def _vocabulary_terms() -> Iterator[str]:
    """A term's name is not another's retired word, and a word retires once."""
    names = [term.name for term in TERMS]
    retired = [old for term in TERMS for old in term.replaces]
    if len(set(names)) != len(names):
        yield "a term is defined twice"
    if len(set(retired)) != len(retired):
        yield "a word retires into two terms"
    yield from (
        f"a term's name is also retired: {n}" for n in set(names) & set(retired)
    )


def _source_words(package: Any) -> Found:
    """A term taken from `package` is a word that package uses."""
    source = "\n".join(text(p) for p in files(installed(package), skip=SKIPPED))
    return {
        term.name
        for term in TERMS
        if term.source == package.__name__
        and not re.search(rf"\b{term.name}\b", source)
    }


for _package in (cnaster, sal):
    RULES.append(
        Rule(
            f"vocabulary-in-{_package.__name__}",
            _package.__name__,
            lambda p=_package: _source_words(p),
        )
    )


@rule("api-vocabulary", "python/port, cnaster", declared=KNOWN)
def _api_vocabulary() -> Found:
    """Retired words, a missing `termination`, a sibling that differs; each with what replaces it."""
    retired = replaced_by()
    found = {key: retired[word] for key, word in words() if word in retired}
    for where, node in owned():
        if isinstance(node, ast.ClassDef):
            names = set(fields(node))
            if names & ITERATION_FIELDS and "termination" not in names:
                found[f"{where} termination"] = "termination"
    by_key = dict(defined())
    for reference, others in SIBLINGS.items():
        first = by_key[reference]
        assert isinstance(first, ast.FunctionDef)
        for other in others:
            node = by_key[other]
            assert isinstance(node, ast.FunctionDef)
            if shape(node) != shape(first):
                found[f"{other} sibling"] = reference
    return found


@rule("api-tensors", "python/port")
def _api_tensors() -> Iterator[str]:
    """A `torch` or `jax` annotation is in a module whose name says so."""
    tensor = re.compile(r"\b(torch|jnp|jax)\.")
    for where, node in owned():
        if re.search(r"torch|jax", where.partition(":")[0]):
            continue
        for sub in ast.walk(node):
            annotation = getattr(sub, "annotation", None) or getattr(
                sub, "returns", None
            )
            if annotation is not None and tensor.search(ast.unparse(annotation)):
                yield f"tensor in a public signature: {where}"


def resolve(target: str) -> Any:
    module_name, _, attribute = target.partition(":")
    __import__(module_name)
    return getattr(sys.modules[module_name], attribute)


def accepts(function: Any) -> list[tuple[str, Any, Any]]:
    """A signature as (name, kind, default), annotations dropped."""
    return [
        (p.name, p.kind, p.default)
        for p in inspect.signature(function).parameters.values()
    ]


def departure_of(swap: Swap) -> str | None:
    """How a replacement's signature differs from cnaster's, or `None`.

    Defaults must match; extra parameters only keyword-only with a default (#186).
    """
    upstream = accepts(resolve(f"{swap.module}:{swap.name}"))
    replacement = accepts(resolve(swap.replacement))
    shared = replacement[: len(upstream)]
    if shared != upstream:
        return f"takes {shared}, not {upstream}"
    for name, kind, default in replacement[len(upstream) :]:
        if kind is not inspect.Parameter.KEYWORD_ONLY:
            return f"{name} is positional and new"
        if default is inspect.Parameter.empty:
            return f"{name} is new and required"
    return None


DEPARTURES: dict[str, str] = {}
"""Rows, `table:name`, that do not yet accept what they replace; may only shrink (#517 step 1)."""


@rule("swap-signatures", "the swap tables", declared=DEPARTURES)
def _swap_signatures() -> Found:
    """Every replacement accepts its original's call (#517 E1); a declared departure is a row."""
    rows = {
        f"{table}:{swap.name}": swap
        for table, swaps in tables().items()
        for swap in swaps
    }
    found: dict[str, object] = {
        key: departure
        for key, swap in rows.items()
        if (departure := departure_of(swap))
    }
    return found | {
        f"{key} no longer a row": "" for key in set(DEPARTURES) - rows.keys()
    }


@rule("swap-sites", "port.pipeline.swap_sites")
def _swap_sites() -> Iterator[str]:
    """Each `from cnaster.omics import ...` binding in `run_cnaster` is rebound."""
    sites = swap_sites()
    bound = {
        site.name for site in sites if site.module == "cnaster.scripts.run_cnaster"
    }
    if len(sites) <= len(SWAPS):
        yield "no name was found bound anywhere but where it is defined"
    expected = {"load_input_data", "assign_initial_blocks", "summarize_counts_for_bins"}
    yield from (f"not rebound in run_cnaster: {name}" for name in expected - bound)


@rule("swap-tickets", "SWAPS")
def _swap_tickets() -> Iterator[str]:
    """Every row cites its measurement, and names its replacement `module:name`."""
    if not SWAPS:
        yield "the table is empty"
    for swap in SWAPS:
        if swap.ticket <= 0 or ":" not in swap.replacement:
            yield f"{swap.module}.{swap.name}: ticket {swap.ticket}, {swap.replacement}"


# --- drop-in correspondence and coverage scope (#128, #250, #281, #517) -------

UNINSTALLED: dict[str, str] = {}
"""Drop-ins deliberately in no swap table, each with what would put it in one."""


def dropins() -> set[str]:
    """Every drop-in module by import path, package roots excluded."""
    return {
        ".".join(path.relative_to(ROOT / "python").with_suffix("").parts)
        for path in (PACKAGE / "patch").rglob("*.py")
        if path.name != "__init__.py"
    }


@rule("dropin-refereed", "tests/test_*.py")
def _dropin_refereed() -> Found:
    """Each drop-in is imported by a `patch`- or `cnaster`-marked test, or it belongs under `extensions/`."""
    imported: set[str] = set()
    for path in TESTS.glob("test_*.py"):
        # NB `patch` and `cnaster` together are guard 4's correspondence selection.
        if any(
            f"pytest.mark.{marker}" in text(path) for marker in ("patch", "cnaster")
        ):
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
    for name in [n for n in imported if n.startswith("port.patch")]:
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - a test importing nothing real
            continue
        for attribute in getattr(module, "__all__", ()):
            origin = getattr(getattr(module, attribute, None), "__module__", None)
            if origin is not None:
                imported.add(origin)
    return {m for m in dropins() if not any(name.startswith(m) for name in imported)}


@rule("dropin-uninstalled", "the swap tables")
def _dropin_uninstalled() -> Iterator[str]:
    """No declared drop-in is installed by a swap table, and each names a drop-in (#281)."""
    reached: set[str] = set()
    for swap in (
        SWAPS
        + FIGURE_SWAPS
        + SHIFT_SWAPS
        + COPY_SWAPS
        + REFINEMENT_SWAPS
        + LOG_SPACE_SWAPS
    ):
        module_name, _, attribute = swap.replacement.partition(":")
        reached.add(module_name)
        origin = getattr(
            getattr(importlib.import_module(module_name), attribute, None),
            "__module__",
            None,
        )
        if origin is not None:
            reached.add(origin)
    yield from (
        f"installed, still declared uninstalled: {m}"
        for m in reached & set(UNINSTALLED)
    )
    yield from (f"declared, not a drop-in: {m}" for m in set(UNINSTALLED) - dropins())


UNIFIERS = ("emission", "lattice", "plotting")
"""Patches replacing several `cnaster` modules, named so adding one is explicit."""


@rule("patch-names", "port.patch")
def _patch_names() -> Iterator[str]:
    """Each non-unifier `port.patch.<name>` imports as `cnaster.<name>` (#250)."""
    for info in pkgutil.iter_modules(port.patch.__path__):
        if not info.name.startswith("_") and info.name not in UNIFIERS:
            try:
                importlib.import_module(f"cnaster.{info.name}")
            except ImportError as error:
                yield f"{info.name}: {error}"


@rule("patch-unifiers", "port.patch")
def _patch_unifiers() -> Iterator[str]:
    """Each unifier's `MIRRORS` names more than one importable target (T- #776)."""
    for name in UNIFIERS:
        mirrors = getattr(
            importlib.import_module(f"port.patch.{name}"), "MIRRORS", None
        )
        if not isinstance(mirrors, tuple) or len(mirrors) <= 1:
            yield f"{name} mirrors {mirrors}: one target is a rename, not an exception"
            continue
        for target in mirrors:
            try:
                importlib.import_module(target)
            except ImportError as error:
                yield f"{name}: {error}"


@rule("swap-targets", "the swap tables")
def _swap_targets() -> Iterator[str]:
    """Every swap row replaces `cnaster.X` from `port.patch.X`."""
    for swap in SWAPS + FIGURE_SWAPS + SHIFT_SWAPS + COPY_SWAPS:
        target = swap.replacement.partition(":")[0]
        expected = f"port.patch.{swap.module.rpartition('.')[2]}"
        if target != expected:
            yield f"{swap.module}.{swap.name} is replaced from {target}, not {expected}"
        try:
            importlib.import_module(target)
        except ImportError as error:
            yield f"{target}: {error}"


PRIVATE_SURFACE = frozenset(
    {
        "cnaster.hmm_nophasing:_bb_logpmf_1d",
        "cnaster.hmm_nophasing:_nb_logpmf_1d",
        "cnaster.hmm_phased:_switch_betabinom_1d",
        # The four `plot_genomic` layout helpers the sandbox replacement imports (#278)
        "cnaster.plot_genomic:_annotate_clone_stats",
        "cnaster.plot_genomic:_create_clone_gridspec",
        "cnaster.plot_genomic:_draw_chromosome_boundaries",
        "cnaster.plot_genomic:_format_track_axis",
    }
)
"""Every underscore-prefixed `cnaster` name `port` imports, reviewed."""


@rule("cnaster-private-surface", "python/port", declared=PRIVATE_SURFACE)
def _cnaster_private_surface() -> Found:
    return {
        f"{node.module}:{alias.name}"
        for path in PACKAGE.rglob("*.py")
        for node in ast.walk(parse(path))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("cnaster")
        for alias in node.names
        if alias.name.startswith("_")
    }


@rule("coverage-source", "pyproject.toml")
def _coverage_source() -> Iterator[str]:
    """The configured path is the package `import cnaster` resolves to."""
    configured = [
        ROOT / source
        for source in pyproject()["tool"]["coverage"]["run"]["source"]
        if "cnaster" in source
    ]
    if len(configured) != 1:
        yield f"expected exactly one cnaster source entry: {configured}"
    # NB cnaster is a namespace package: `__file__` is None.
    elif configured[0].resolve() != installed(cnaster).resolve():
        yield (
            f"coverage measures {configured[0]}, but cnaster is installed at "
            f"{installed(cnaster).resolve()}; the gate reports a fraction of the wrong denominator"
        )


def oracle_globs() -> list[str]:
    return [glob.strip() for glob in oracle_config()["report"]["include"].split()]


@rule("oracle-installed", ".coveragerc-oracle")
def _oracle_installed() -> Iterator[str]:
    """Each declared oracle module resolves via `find_spec` in the installed package."""
    for glob in oracle_globs():
        name = (
            glob.removeprefix("*/")
            .removesuffix("/*")
            .removesuffix(".py")
            .replace("/", ".")
        )
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            yield f"{name} resolves to no file"
        elif not Path(spec.origin).resolve().is_relative_to(installed(sal).resolve()):
            yield f"{name} resolves outside the installed package"


@rule("oracle-imports", "tests, .coveragerc-oracle")
def _oracle_imports() -> Iterator[str]:
    """Every upstream module a test imports is in the declared oracle surface."""
    declared = {
        g.removeprefix("*/").removesuffix("/*").removesuffix(".py").replace("/", ".")
        for g in oracle_globs()
    }
    # NB `tests/sandbox/` included: a set-aside test referees against `sal` too (#851)
    for path in files(TESTS):
        used: set[str] = set()
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "sal"
            ):
                used.add(node.module or "")
            elif isinstance(node, ast.Import):
                used.update(a.name for a in node.names if a.name.startswith("sal"))
        yield from (
            f"{path.relative_to(TESTS)}: {name}"
            for name in used
            if name != "sal"
            and not any(name == d or name.startswith(d + ".") for d in declared)
        )


CORRESPONDENCE: dict[str, tuple[str, ...]] = {
    "sal.emissions": ("cnaster.hmm_nophasing", "cnaster.hmm_emission"),
    "sal.opt.hmm": ("cnaster.hmm", "cnaster.hmm_nophasing"),
    "sal.opt.fit": ("cnaster.integer_copy",),
    "sal.ragged": ("cnaster.hmrf_utils",),
    "sal.likelihood.ragged": ("cnaster.hmm_nophasing",),
    "sal.likelihood.forward_backward": ("cnaster.hmm_nophasing",),
    "sal.likelihood.spatio_sequential": ("cnaster.hmrf",),
    "sal.search.spatio_sequential": ("cnaster.hmrf",),
    "sal.search.alpha_expansion": ("cnaster.icm",),
    "sal.search.icm": ("cnaster.icm",),
    "sal.search.trws": ("cnaster.icm",),
    "sal.enumeration": ("cnaster.icm",),
    "sal.sim.potts": ("cnaster.icm",),
    "sal.backend": ("cnaster.icm",),
    "sal.sim.count_pairs": ("cnaster.hmm_nophasing",),
    "sal.sim.spatio_sequential": ("cnaster.hmrf",),
    "sal.sim.graph": ("cnaster.hmrf_utils",),
    "sal.sim.hmm": ("cnaster.hmm_nophasing",),
    "sal.opt.emission_mixture": ("cnaster.hmm_initialize",),
    "sal.opt.mixture": ("cnaster.hmm_initialize",),
    "sal.search.mixture_starts": ("cnaster.hmm_initialize",),
    "sal.opt.objective": ("cnaster.hmm_nophasing",),
    "sal.opt.termination": ("cnaster.integer_copy",),
}
"""Each declared upstream module and the `cnaster` code whose claim rests on it (#128)."""

CNASTER_EMISSION_KERNELS = frozenset({"nb", "bb"})
"""The emission families `cnaster` implements, derived from its kernels (#232)."""

UNMATCHED_FAMILIES = frozenset(
    {
        "CategoricalEmission",
        "GaussianEmission",
        "PoissonEmission",
        "BinomialEmission",
        "RateConcentrationBetaBinomialEmission",
        "RateConcentrationCountPairEmission",
    }
)
"""Upstream families with no `cnaster` counterpart, excluded from the denominator (T- #707)."""


@rule("oracle-correspondence", ".coveragerc-oracle", declared=CORRESPONDENCE)
def _oracle_correspondence() -> Found:
    """Declared oracle modules equal `CORRESPONDENCE`'s keys, both ways."""
    tails = [
        g.split("sal/", 1)[1].removesuffix("/*").removesuffix(".py")
        for g in oracle_globs()
    ]
    return {"sal." + tail.replace("/", ".") for tail in tails}


@rule("oracle-counterparts", "cnaster")
def _oracle_counterparts() -> Iterator[str]:
    """Every `cnaster` counterpart resolves via `find_spec`, without importing it."""
    for upstream, counterparts in CORRESPONDENCE.items():
        yield from (
            f"{upstream}: {c}"
            for c in counterparts
            if importlib.util.find_spec(c) is None
        )


def spec_tree(name: str) -> ast.Module:
    spec = importlib.util.find_spec(name)
    assert spec is not None
    assert spec.origin is not None
    return parse(Path(spec.origin))


@rule(
    "cnaster-emission-kernels",
    "cnaster.hmm_nophasing",
    declared=CNASTER_EMISSION_KERNELS,
)
def _cnaster_emission_kernels() -> Found:
    """`hmm_nophasing`'s `_*_logpmf_1d` kernels are exactly `nb` and `bb` (#128)."""
    return {
        node.name.removeprefix("_").removesuffix("_logpmf_1d")
        for node in spec_tree("cnaster.hmm_nophasing").body
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_logpmf_1d")
    }


@rule("unmatched-families", "sal.emissions", declared=UNMATCHED_FAMILIES)
def _unmatched_families() -> Found:
    """Upstream's emission families with no `cnaster` counterpart."""
    origin = importlib.util.find_spec("sal.emissions")
    assert origin is not None
    assert origin.origin is not None
    matched = {
        "NegativeBinomialEmission",
        "BetaBinomialEmission",
        # NB matched on #232: referees `cnaster`'s kernels to 1.4e-13 (#69)
        "CountPairEmission",
    }
    base = {"CountEmissionFamily", "EmissionFamily"}
    # NB a package since e0aeb19 (#410): families span its modules
    return (
        {
            node.name
            for path in sorted(Path(origin.origin).parent.glob("*.py"))
            for node in parse(path).body
            if isinstance(node, ast.ClassDef) and node.name.endswith("Emission")
        }
        - matched
        - base
    )


@rule("excluded-families", ".coveragerc-oracle", declared=UNMATCHED_FAMILIES)
def _excluded_families() -> Found:
    """`exclude_also` names exactly the unmatched families."""
    return {
        line.strip().removeprefix("^class ")
        for line in oracle_config()["report"]["exclude_also"].splitlines()
        if line.strip()
    }


VALIDATED: dict[str, str] = {
    "hmm_emission:compute_bb_ab": "tests/test_m_step.py::test_m_step_agrees_with_upstream",
    "hmm_nophasing:nbinom_logpmf_numba": "tests/test_numba_kernels.py::test_negative_binomial_kernel_matches_scipy",
    "hmm_nophasing:betabinom_logpmf_numba": "tests/test_numba_kernels.py::test_beta_binomial_kernel_matches_scipy",
    "hmm_nophasing:numba_logsumexp": "tests/test_numba_kernels.py::test_numba_logsumexp_matches_scipy",
    "hmm_nophasing:_nb_logpmf_1d": "tests/test_numba_kernels.py::test_dense_and_single_observation_kernels_agree",
    "hmm_nophasing:_dense_nb_logpmf": "tests/test_numba_kernels.py::test_dense_and_single_observation_kernels_agree",
    "hmm_nophasing:_bb_logpmf_1d": "tests/test_core_inference_fixture.py::test_cnaster_scores_the_fixture_as_upstream_does",
    "hmm_nophasing:_dense_bb_logpmf": "tests/test_core_inference_fixture.py::test_the_field_recovers_the_planted_clone_assignment",
    "hmm_nophasing:compute_logmu_shifts": "tests/test_logmu_shift.py::test_it_reproduces_cnasters_loop",
    "hmm_nophasing:forward_lattice": "tests/test_hmm_single_chain.py::test_total_log_likelihood_matches_upstream",
    "hmm_nophasing:backward_lattice": "tests/test_lattice_invariants.py::test_forward_and_backward_agree_at_every_position",
    "hmm_phased:update_combined_transmat": "tests/test_hmm_phased.py::test_combined_transition_matches_cnaster_construction",
    "hmm_phased:forward_lattice": "tests/test_hmm_phased.py::test_total_log_likelihood_matches_upstream",
    "hmrf:compute_loglike_spot_assignment": "tests/test_core_inference_fixture.py::test_the_field_recovers_the_planted_clone_assignment",
}  # fmt: skip
"""Each compiled `cnaster` kernel -> the test that checks its numbers against something outside it.
Coverage cannot see `numba` kernels, so the registry carries the claim."""

UNVALIDATED: dict[str, str] = {
    "hmm_emission:collapse_exog": "the design matrix the BB M step folds; #94",
    "hmm_emission:betabinom_logpmf": "the non-`_numba` beta-binomial in `hmm_emission`, distinct from the one `hmm_nophasing` ships and validated above; #94",
    "hmm_nophasing:np_sum_ax_squeeze": "an inlined reduction helper; #94",
    "hmm_phased:backward_lattice": "the phased forward is refereed against upstream and the backward is not, which is the asymmetry `test_lattice_invariants` closes for the unphased pair and nothing closes for this one; #94",
    "hmm_phased:_switch_betabinom_1d": "the phased channel is unreachable through its own classmethod; #9",
    "hmrf:logsumexp": "a third copy of the reduction, module-local; #94",
    "hmrf:pool_spatio_genomic_counts": "#94, and the pooling cost is #13",
    "icm:calc_cluster_assignment_cost": "#64 reduces this interface; #94",
    "icm:calc_assignment_cost": "#64 reduces this interface; #94",
    "icm:logsumexp": "a fourth copy of the reduction, module-local; #94",
    "icm:icm_sweep": "the array solver, not the `_deque` one the live path takes; #64, #94",
    "normal_spot:compute_local_normal_mask": "#89",
    "normal_spot:_compute_baseline_core": "#89",
    "utils:top_hat_sum": "#94",
    "wolff:_wolff_annealing_core": "the module raises on import; #71",
    "scripts/run_cnaster:set_numba_seed": "under `scripts/`, which has no `__init__.py` and so is outside the coverage denominator as well; #94",
}  # fmt: skip
"""Each compiled kernel -> why it has no validation yet, and the ticket that owns the gap."""


@cache
def kernels() -> dict[str, str]:
    """Every `@njit` function `cnaster` ships outside `deprecated/` and `sandbox/`, by `module:name`."""
    root = installed(cnaster)
    found = {}
    for path in files(root, skip={"deprecated", "sandbox"}):
        module = path.relative_to(root).with_suffix("").as_posix()
        for node in ast.walk(parse(path)):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    source = ast.unparse(decorator)
                    if source == "njit" or source.startswith("njit("):
                        found[f"{module}:{node.name}"] = source
    return found


@rule("numba-classified", "cnaster", declared=VALIDATED.keys() | UNVALIDATED.keys())
def _numba_classified() -> Found:
    """Every compiled kernel is validated or names the ticket that owns the gap."""
    return kernels()


@rule("numba-validation-tests", COLLECTION)
def _numba_validation_tests(items: list[pytest.Item]) -> Found:
    collected = {item.nodeid.split("[")[0] for item in items}
    return {kernel: node for kernel, node in VALIDATED.items() if node not in collected}


@rule("numba-gaps-ticketed", "UNVALIDATED")
def _numba_gaps_ticketed() -> Found:
    """An excuse without a ticket is a decision nobody will revisit."""
    return {
        k: reason for k, reason in UNVALIDATED.items() if not re.search(r"#\d+", reason)
    }


@rule("numba-scan", "cnaster")
def _numba_scan() -> Iterator[str]:
    if len(kernels()) < 25:
        yield f"the scan found {len(kernels())} kernels"
    if len(VALIDATED) < 10:
        yield f"{len(VALIDATED)} validated"
    yield from (k for k, source in kernels().items() if not source.startswith("njit"))


UNCOUNTED = frozenset(
    {
        "FIGURE_SWAPS:plot_clones_genomic",
        "FIGURE_SWAPS:plot_clones_spatial",
        "FIGURE_SWAPS:plot_copy_number_profile",
        "FIGURE_SWAPS:write_fig",
        "PLOT_OFF_SWAPS:write_fig",
        "SWAPS:assign_initial_blocks",
        "SWAPS:create_bin_ranges",
        "SWAPS:get_aggregated_barcodes",
        "SWAPS:get_reference_genes",
        "SWAPS:initialize_rectangular_clones",
        "SWAPS:summarize_blocks",
    }
)
"""11 of 31 rows without a counting referee. The four figure rows have no truth to count against."""


@rule("counting-referees", "the swap tables, tests", declared=UNCOUNTED)
def _counting_referees() -> Iterator[str]:
    """Rows no end-to-end or oracle test names, by row or by replacement (#517 E5)."""
    mentions = counting_mentions()
    for table, swaps in tables().items():
        for swap in swaps:
            attribute = swap.replacement.rpartition(".")[2].rpartition(":")[2]
            if not (mentions.get(swap.name) or mentions.get(attribute)):
                yield f"{table}:{swap.name}"


# --- docs and config agree with what generates them ---------------------------


def ledger_defines(test: str) -> bool:
    path, _, name = test.partition("::")
    source = Path(ROOT, path)
    return source.is_file() and any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in parse(source).body
    )


@rule("metrics-runs", "docs/metrics/runs.tsv")
def _metrics_runs() -> Iterator[str]:
    """Every run parses, names a test that exists, and runs are in timestamp order (#409)."""
    every = ledger.runs()
    if not every:
        yield "docs/metrics/runs.tsv has no runs"
    ids = [run["run_id"] for run in every]
    if len(ids) != len(set(ids)):
        yield "a run_id appears twice"
    for run in every:
        if not re.fullmatch(r"[0-9a-f]{7}\+?", run["commit"]):
            yield f"commit: {run}"
        if not run["run_id"].startswith(f"{run['commit']}-"):
            yield f"run_id: {run}"
        try:
            datetime.datetime.strptime(run["timestamp"], TIMESTAMP).replace(
                tzinfo=datetime.UTC
            )
            check_note(run["note"].split(CONVERTED)[0])
        except ValueError as error:
            yield f"{error}: {run}"
        if not ledger_defines(run["test"]):
            yield f"no such path::function: {run['test']}"
    if not ledger_defines(TEST):
        yield f"no such path::function: {TEST}"
    stamps = [run["timestamp"] for run in every]
    if stamps != sorted(stamps):
        yield "runs are not in timestamp order"


@rule("metrics-ledger", "docs/metrics/ledger.tsv")
def _metrics_ledger() -> Iterator[str]:
    """Each line names a run and a definition, a hash of 8 hex digits and no hash in its name, and a number."""
    ids = {run["run_id"] for run in ledger.runs()}
    defined = {(d["metric"], d["definition"]) for d in ledger.definitions()}
    lines = ledger.ledger()
    if not lines:
        yield "docs/metrics/ledger.tsv has no lines"
    for line in lines:
        if line["run_id"] not in ids:
            yield f"no such run: {line}"
        if (line["metric"], line["definition"]) not in defined:
            yield f"no such definition: {line}"
        if not re.fullmatch(r"[0-9a-f]{8}", line["fixture_hash"]):
            yield f"fixture_hash: {line}"
        # NB the hash is the `fixture_hash` column, not a suffix of the name (#739)
        if re.search(r"_[0-9a-f]{8}$", line["fixture"]) is not None:
            yield f"fixture carries its hash: {line}"
        try:
            float(line["value"])
        except ValueError:
            yield f"value: {line}"
    keys = [(line["run_id"], line["metric"]) for line in lines]
    if len(keys) != len(set(keys)):
        yield "a run measures a metric twice"


@rule("metrics-definitions", "docs/metrics/definitions.tsv")
def _metrics_definitions() -> Iterator[str]:
    """Definition versions run 1, 2, ... per metric, and every `METRICS` key has one."""
    numbers: dict[str, list[int]] = {}
    for d in ledger.definitions():
        numbers.setdefault(d["metric"], []).append(int(d["definition"]))
        if not (d["since"] and d["scorer"] and d["meaning"]):
            yield f"incomplete: {d}"
    yield from (f"undefined: {m}" for m in set(METRICS) - set(numbers))
    for metric, found in numbers.items():
        if found != list(range(1, len(found) + 1)):
            yield f"numbered {found}: {metric}"


ADDED = frozenset({"clone_ari_int_99"})
"""Metrics added after the conversion (T- #817): no converted run measured them."""

CONVERTED_ROWS = 73
CONVERTED_SHA256 = "0bcb7c57193ac696ed08cca106200ab817914289c3900ba35187373810b44e10"
"""SHA-256 of the 73 data lines of `docs/metrics.md` at ef2261d, read from git (#620)."""


@rule("metrics-converted", "docs/metrics")
def _metrics_converted() -> Iterator[str]:
    """`--render`'s first 73 rows, in the old table's form, hash to `CONVERTED_SHA256`."""
    rows = ledger.parse(ledger.render())[:CONVERTED_ROWS]
    kept = {"note": lambda v: v.split(CONVERTED)[0]}
    # NB the converted table's columns: a metric added since (`ADDED`) has none
    lines = [
        "| "
        + " | ".join(
            kept.get(c, lambda v: v)(row[c])
            for c in ledger.COLUMNS
            if c != "benchmark" and c not in ADDED
        )
        + " |"
        for row in rows
    ]
    if len(lines) != CONVERTED_ROWS:
        yield f"{len(lines)} converted rows"
    if hashlib.sha256("\n".join(lines).encode()).hexdigest() != CONVERTED_SHA256:
        yield "the converted rows' hash moved"


@rule("metrics-dev-fixture", "docs/metrics, port.sim.truth")
def _metrics_dev_fixture() -> Iterator[str]:
    """The latest `dev` run hashes the fixture `dev_instance` builds now (#620)."""
    recorded = [row for row in ledger.read() if row["fixture"] == "dev"]
    if not recorded:
        yield "no dev run: run_ledger --record"
    elif (built := fixture_hash(sim_truth.dev_instance())) != recorded[-1][
        "fixture_hash"
    ]:
        yield (
            f"dev_instance now builds {built}, the latest dev run is "
            f"{recorded[-1]['fixture_hash']}: record a run on the new data"
        )


@rule("metrics-last-benchmark", "docs/metrics")
def _metrics_last_benchmark() -> Iterator[str]:
    """`benchmark` is a boolean; the latest sweep's runs share a commit and each names a dataset."""
    if not {r["benchmark"] for r in ledger.runs()} <= {"true", "false"}:
        yield "benchmark is not a boolean"
    sweep = ledger.last_benchmark()
    if not sweep:
        yield "no benchmark sweep"
    if len({r["commit"] for r in sweep}) > 1:
        yield "the last sweep spans commits"
    keys = [
        next(
            (line["fixture"], line["fixture_hash"])
            for line in ledger.ledger()
            if line["run_id"] == r["run_id"]
        )
        for r in sweep
    ]
    if len(keys) != len(set(keys)):
        yield "the last sweep measures a dataset twice"


@rule("metrics-cnaster-ledger", "docs/metrics/cnaster")
def _metrics_cnaster_ledger() -> Iterator[str]:
    """Its lines name its runs and a current definition, its runs are the arm's, and no run is port's (T- #833)."""
    runs = ledger.runs(ledger.CNASTER_DIR)
    lines = ledger.ledger(ledger.CNASTER_DIR)
    current = ledger.latest()
    ids = {r["run_id"] for r in runs}
    yield from (f"no such run: {line}" for line in lines if line["run_id"] not in ids)
    yield from (
        f"not current: {line}"
        for line in lines
        if current[line["metric"]] != line["definition"]
    )
    yield from (
        f"not the cnaster arm: {r}"
        for r in runs
        if not r["arm"].startswith("-- --no-patch")
    )
    yield from (f"also port's: {i}" for i in ids & {r["run_id"] for r in ledger.runs()})


README = ROOT / "README.md"
ENDPOINT = re.compile(
    r"!\[[^\]]*\]\(https://img\.shields\.io/endpoint\?url=[^)]*?/\.badges/([a-z-]+)\.json\)"
)
"""Each badge the README renders, by file name."""


@rule("badge-files", ".badges")
def _badge_files() -> Iterator[str]:
    """Each committed badge is the generator's payload, colour included, and no other file is."""
    for badge in badges():
        path = BADGES / f"{badge.name}.json"
        if not path.exists():
            yield f"{path.name} is missing; run `python -m scripts.badges`"
        elif json.loads(path.read_text()) != badge.payload():
            yield f"{path.name} is stale; run `python -m scripts.badges`"
    names = {f"{badge.name}.json" for badge in badges()}
    yield from (
        f"{path.name} is generated by nothing; delete it"
        for path in BADGES.glob("*.json")
        if path.name != MEASUREMENTS.name and path.name not in names
    )


@rule("badge-readme", "README.md")
def _badge_readme() -> Iterator[str]:
    """The README renders exactly the generated badges."""
    rendered = set(ENDPOINT.findall(text(README)))
    generated = {badge.name for badge in badges()}
    yield from (f"rendered, not generated: {n}" for n in rendered - generated)
    yield from (f"generated, not rendered: {n}" for n in generated - rendered)


@rule("badge-conditions", ".badges/measurements.json")
def _badge_conditions() -> Iterator[str]:
    """Every recorded measurement states the conditions that decided it, and its ratio is its counts'."""
    recorded = json.loads(MEASUREMENTS.read_text())
    for name, guard in recorded["coverage"].items():
        for key in ("label", "percent", "floor", "selection", "denominator", "commit"):
            if key not in guard:
                yield f"coverage guard {name} does not state {key}"
        if guard.get("percent") is None and not guard.get("note"):
            yield f"coverage guard {name} is unmeasured and does not say why"
    stated_keys = {
        "patched": ("percent", "patched_lines", "executed_lines", "instance", "commit"),
        "recovery": ("instance", "configuration", "commit", "arms"),
        "speed": ("sample", "commit", "calicost_seconds", "port_seconds", "port_cores", "assumption", "ratio"),
        "whole_run": ("instance", "commit", "arms"),
    }  # fmt: skip
    for record, keys in stated_keys.items():
        if record == "whole_run" or recorded.get(record) is not None:
            yield from (
                f"{record} does not state {k}"
                for k in keys
                if k not in recorded[record]
            )
    if (patched := recorded.get("patched")) is not None:
        if patched["patched_lines"] > patched["executed_lines"]:
            yield "the patched share exceeds its executed lines"
        if patched["percent"] != pytest.approx(
            100.0 * patched["patched_lines"] / patched["executed_lines"], abs=0.005
        ):
            yield "the percent is not the recorded counts' ratio"
    for arm, values in (recorded.get("recovery") or {"arms": {}})["arms"].items():
        if not {"ari", "ari_integer", "copy_ari", "flags"} <= set(values):
            yield f"recovery arm {arm} does not state its flags and its indices"
    speed = recorded.get("speed")
    if speed is not None and speed["ratio"] != pytest.approx(
        speed["calicost_seconds"] / speed["port_seconds"], rel=1e-3
    ):
        yield "the speed ratio is not the recorded walls' ratio"
    run = recorded["whole_run"]
    if run.get("ratio") is None:
        if not run.get("note"):
            yield "a comparison with no ratio must say what happened"
    else:
        # NB presence alone passes a null; a ratio must name its instance and two
        #    completed arms, since CI cannot re-measure a whole run
        if set(run["ratio"]) != {"runtime", "memory"}:
            yield f"the whole-run ratio is {set(run['ratio'])}"
        if not run["instance"]:
            yield "a ratio must name the instance it was read at"
        if len(run["arms"]) != 2:
            yield f"a ratio is two arms; {len(run['arms'])} recorded"
        if not all(arm["returncode"] == 0 for arm in run["arms"]):
            yield "a ratio from an arm that did not complete is not a ratio"


@rule("badge-ratio-instance", ".badges")
def _badge_ratio_instance() -> Iterator[str]:
    """A rendered ratio requires the `instance` badge to be measured, and is in ratio units."""
    if json.loads(MEASUREMENTS.read_text())["whole_run"].get("ratio") is None:
        pytest.skip("no ratio was produced; the badges carry the reason instead")
    rendered = {badge.name: badge for badge in badges()}
    if rendered["instance"].message == UNMEASURED:
        yield "a ratio is rendered but `instance` reads unmeasured, so the numbers name no size"
    yield from (
        f"{name} is not in ratio units"
        for name, badge in rendered.items()
        if name.startswith("run-") and not badge.message.endswith("X")
    )


@rule("gitattributes-badges", ".gitattributes")
def _gitattributes_badges() -> Iterator[str]:
    """`.gitattributes` routes what `--badges` regenerates through the badges driver."""
    lines = text(ROOT / ".gitattributes").splitlines()
    for pattern in (".badges/*.json",):
        if not any(
            line.startswith(pattern) and "merge=badges" in line for line in lines
        ):
            yield pattern


PAPER_FIGURES = "docs/plots/paper/"
"""The one tree under `docs/` allowed to track a PNG: T- #624's paper set."""


@rule("docs-png", "git ls-files docs")
def _docs_png() -> Found:
    """`docs/` tracks no PNG outside `docs/plots/paper/` (T- #624, #743)."""
    return [
        path
        for path in git_files("--", "docs")
        if path.lower().endswith(".png") and not path.startswith(PAPER_FIGURES)
    ]


@rule("paper-readme", "docs/plots/paper")
def _paper_readme() -> Iterator[str]:
    """The paper README lists every tracked file once, matching `QUESTIONS | KEY_STUDIES` (#624)."""
    tree = PAPER.relative_to(ROOT).as_posix()
    committed = sorted(p.removeprefix(tree + "/") for p in git_files("--", tree))
    committed.remove("README.md")
    listed = re.findall(r"^\| `([^`]+)` \|", text(PAPER / "README.md"), re.M)
    if len(listed) != len(set(listed)):
        yield "the README lists a file twice"
    if sorted(listed) != committed:
        yield f"listed {sorted(listed)}, committed {committed}"
    if sorted(QUESTIONS | KEY_STUDIES) != committed:
        yield f"QUESTIONS | KEY_STUDIES {sorted(QUESTIONS | KEY_STUDIES)}, committed {committed}"


@rule("cnamaste-schema", "docs/cnamaste-h5.md")
def _cnamaste_schema() -> Iterator[str]:
    """`docs/cnamaste-h5.md`'s tables are `render()` of both schemas, verbatim (#817)."""
    tables = re.findall(
        r"(\| Group \|.*?)\n\n", text(ROOT / "docs" / "cnamaste-h5.md"), flags=re.DOTALL
    )
    if tables != [
        cnamaste.render(cnamaste.GROUPS),
        cnamaste.render(cnamaste.TRUTH_GROUPS),
    ]:
        yield "docs/cnamaste-h5.md is not the schema"


@rule("readme-options", "README.md")
def _readme_options() -> Iterator[str]:
    """Every option of `run_cnaster_port` in the README's table, and nothing else (T- #617)."""
    flags = {
        option
        for action in entry._parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    lines = text(README).splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("| Option | Default")
    )
    documented: set[str] = set()
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        documented |= set(re.findall(r"`(--[a-z0-9-]+)`", line.split("|")[1]))
    yield from (
        f"undocumented: {flag}"
        for flag in flags - documented
        if not (flag.startswith("--no-") and f"--{flag[5:]}" in documented)
    )
    yield from (f"documented, not a flag: {flag}" for flag in documented - flags)


@rule("defaults-table", "port.pipeline.DEFAULTS")
def _defaults_table() -> Iterator[str]:
    """Every `DEFAULTS` row is a `Settings` field with a flag that turns it off (T- #617 rule 8)."""
    for default in DEFAULTS:
        if default.setting not in Settings._fields:
            yield f"not a setting: {default.setting}"
            continue
        off = f"--no-{default.flag.removeprefix('--')}"
        settings = entry._settings(entry._parser().parse_args(["config.yaml", off]))
        if getattr(settings, default.setting) is not False:
            yield f"{off} does not turn off {default.setting}"


@rule("figure-style", "pyproject.toml")
def _figure_style() -> Iterator[str]:
    """`DEFAULT`, used where no `pyproject.toml` is found, equals `[tool.port.figures]`."""
    if stated() != DEFAULT:
        yield f"{stated()} != {DEFAULT}"


@rule("port-forward", "docs/port-forward.md")
def _port_forward() -> Iterator[str]:
    """The committed table is the generated one, one `cnaster` function per row in a stage (T- #617)."""
    if text(port_forward.OUT) != port_forward.render():
        yield "run python -m scripts.port_forward"
    found = port_forward.rows()
    if len(found) != sum(len(swaps) for swaps in tables().values()):
        yield f"{len(found)} rows"
    yield from ({row[1] for row in found} - {name for name, _ in port_forward.STAGES})


MANIFESTS = SIM_ROOT / "manifests"

REVERSIBLE = frozenset(
    {
        "baseline/dev_tree.toml",
        "calicost_grch38.toml",
        "dev_shared_unique.toml",
        "dev_tree.toml",
        "dev_tree_1s.toml",
        "dev_tree_1s_easy.toml",
        "dev_tree_1s_hard.toml",
    }
)
"""The manifests that predate T- #698 and keep the reversible default."""


@rule("manifest-loh", "sim/manifests")
def _manifest_loh() -> Iterator[str]:
    """A manifest outside `REVERSIBLE` resolves `[cna] loh` to `"irreversible"`; each listed one exists and does not."""
    loh = {
        p.relative_to(MANIFESTS).as_posix(): str(
            extended(p).get("cna", {}).get("loh", "reversible")
        )
        for p in sorted(MANIFESTS.rglob("*.toml"))
    }
    if not loh.keys() - REVERSIBLE:
        yield "no new manifest: dev_tree_1s_dense.toml at least"
    yield from (
        f"{n}: {r}"
        for n, r in loh.items()
        if n not in REVERSIBLE and r != "irreversible"
    )
    yield from (f"listed, missing: {n}" for n in REVERSIBLE - loh.keys())
    yield from (
        f"listed, {loh[n]}: {n}"
        for n in REVERSIBLE & loh.keys()
        if loh[n] != "reversible"
    )


@rule("manifest-r0-hash", "sim/manifests")
def _manifest_r0_hash() -> Iterator[str]:
    """Each `dev_tree` and study manifest states its own 8-hex `r0_hash`; the baseline's is `R0_HASH` (#583, #619)."""
    if len(HASHED) != 10:
        yield f"{len(HASHED)} hashed manifests"
    for path in HASHED:
        r0 = _stated(path)
        if not isinstance(r0, str) or not re.fullmatch(r"[0-9a-f]{8}", r0):
            yield f"{path.relative_to(MANIFESTS)}: {r0}"
    if _stated(MANIFESTS / "baseline" / "dev_tree.toml") != R0_HASH:
        yield "the baseline's r0_hash is not R0_HASH"


@rule("file-sizes", "git ls-files")
def _file_sizes() -> Iterator[str]:
    """No tracked or addable file exceeds `[tool.port] max_file_bytes`."""
    limit = pyproject()["tool"]["port"]["max_file_bytes"]
    listed = [
        n
        for n in git_files("--cached", "--others", "--exclude-standard")
        if (ROOT / n).is_file()
    ]
    if not listed:
        yield "git listed no files"
    for name in listed:
        if (size := (ROOT / name).stat().st_size) > limit:
            yield f"{name}: {size:_} bytes, over {limit:_}"


TREES = {
    "run": ("patch", "extensions", "scripts", "pipeline.py"),
    "qa": ("qa", "studies", "sim"),
    "sandbox": ("sandbox",),
    "tests": (),
    "tests_sandbox": (),
}
"""Each budgeted tree's paths under `python/port`; `tests` is the repository's `tests/`
outside `tests/sandbox/`, and `tests_sandbox` is `tests/sandbox/` (#851)."""

BUDGET = {
    "run": 14842,
    "qa": 17339,
    "sandbox": 7843,
    "tests": 26512,
    "tests_sandbox": 2499,
}
"""Non-blank lines per tree (T- #831), comments and docstrings included, lowered as packages
land; a move between trees transfers its lines. Only falls."""

SLACK = 0.02
"""How far below its budget a tree may fall before the budget must be lowered to it."""

TARGET = 1.0
"""The run path's ceiling as a multiple of cnaster's non-blank lines; the aim is 0.5."""


def lines(paths: list[Path], excluded: Path | None = None) -> int:
    """Non-blank lines in every `.py` file under `paths`, none under `excluded`."""
    every = [f for p in paths for f in ([p] if p.is_file() else p.rglob("*.py"))]
    kept = [f for f in every if excluded is None or not f.is_relative_to(excluded)]
    return sum(1 for f in kept for line in f.read_text().splitlines() if line.strip())


def counted(tree: str) -> int:
    if tree == "tests":
        return lines([TESTS], excluded=SANDBOX_TESTS)
    if tree == "tests_sandbox":
        return lines([SANDBOX_TESTS])
    return lines([PACKAGE / p for p in TREES[tree]])


def _budget(tree: str) -> Iterator[str]:
    """At most `BUDGET[tree]` lines, and no more than `SLACK` below it."""
    found = counted(tree)
    if found > BUDGET[tree]:
        yield f"{tree}: {found} lines over its budget {BUDGET[tree]}"
    if found < (1 - SLACK) * BUDGET[tree]:
        yield f"{tree}: {found} lines; lower BUDGET to it"


for _tree in BUDGET:
    RULES.append(
        Rule(f"line-budget-{_tree}", f"lines of {_tree}", lambda t=_tree: _budget(t))
    )


@rule("line-budget-tests-ratio", "tests, python/port")
def _line_budget_tests_ratio() -> Iterator[str]:
    """Tests at most 2x port's own lines, `tests/sandbox/` and `python/port/sandbox/` excluded."""
    if counted("tests") > 2 * (counted("run") + counted("qa")):
        yield f"tests {counted('tests')} > 2 x {counted('run') + counted('qa')}"


@rule("line-budget-run-path", "python/port, cnaster")
def _line_budget_run_path() -> Iterator[str]:
    """The run path at most `TARGET` x cnaster's non-blank lines, its `sandbox/` and `deprecated/` excluded."""
    upstream = [
        p
        for p in installed(cnaster).rglob("*.py")
        if not {"sandbox", "deprecated"} & set(p.parts)
    ]
    if counted("run") > TARGET * lines(upstream):
        yield f"run {counted('run')} > {TARGET} x {lines(upstream)}"


# --- marker discipline and CI steps (#157, #403) -------------------------------

MARKERS = frozenset(
    {"end2end", "oracle", "analytic", "patch", "backend", "bug", "warning", "snapshot", "smoke", "infra"}
)  # fmt: skip
"""What a test is checked against: exactly one per test. `--strict-markers` refuses only
unregistered markers; these rules refuse missing ones."""

EXTERNAL = frozenset({"end2end", "oracle", "analytic", "patch", "backend"})
"""Referees outside the code under test; what may gate early."""

SCALE = frozenset({"merge", "release", "deprecate", "benchmark"})
"""Markers that say a test is not fast; disqualifying for the early gate."""

TIERS = frozenset({"critical", "merge", "release", "deprecate"})
"""When a test runs (#403): at most one; none is the gate."""


@rule("marker-one-referee", COLLECTION)
def _marker_one_referee(items: list[pytest.Item]) -> Iterator[str]:
    """Every non-benchmark test carries exactly one of `MARKERS`; no benchmark carries one."""
    for item in items:
        referees = sorted(markers(item) & MARKERS)
        if len(referees) != 1 if "benchmark" not in markers(item) else referees:
            yield f"{item.nodeid}: {referees}"


@rule("marker-infra-few", COLLECTION)
def _marker_infra_few(items: list[pytest.Item]) -> Iterator[str]:
    """`infra` stays at most a fifth of marked tests."""
    infra = sum("infra" in markers(item) for item in items)
    marked = sum(bool(markers(item) & MARKERS) for item in items)
    if infra > marked // 5:
        yield f"{infra} of {marked} marked tests are infra"


@rule("marker-early-gate", COLLECTION)
def _marker_early_gate(items: list[pytest.Item]) -> Iterator[str]:
    """Every `critical` test carries an `EXTERNAL` referee and no `SCALE` marker, and there are 20."""
    critical = [item for item in items if "critical" in markers(item)]
    yield from (
        f"{i.nodeid}: no external referee"
        for i in critical
        if not markers(i) & EXTERNAL
    )
    yield from (
        f"{i.nodeid}: {sorted(markers(i) & SCALE)}"
        for i in critical
        if markers(i) & SCALE
    )
    if len(critical) < 20:
        yield f"the early gate holds {len(critical)} tests"


@rule("marker-none-empty", COLLECTION)
def _marker_none_empty(items: list[pytest.Item]) -> Found:
    """Every one of `MARKERS` is carried by some test (#157)."""
    return MARKERS - {name for item in items for name in markers(item)}


@rule("marker-one-tier", COLLECTION)
def _marker_one_tier(items: list[pytest.Item]) -> Found:
    return {
        i.nodeid: sorted(markers(i) & TIERS)
        for i in items
        if len(markers(i) & TIERS) > 1
    }


def selects(expression: str, names: set[str]) -> bool:
    return Expression.compile(expression).evaluate(lambda name, /, **_: name in names)


@rule("ci-steps", COLLECTION)
def _ci_steps(items: list[pytest.Item]) -> Iterator[str]:
    """`scripts.ci`'s steps select every test, and outside the two coverage guards, once."""
    every = (
        ci.GATE,
        ci.JUDGED,
        ci.DROPIN,
        ci.MERGE_REST,
        "benchmark and not release",
        ci.RELEASE,
    )
    exclusive = (ci.GATE, ci.MERGE_REST, "benchmark and not release", ci.RELEASE)
    for item in items:
        if not any(selects(step, markers(item)) for step in every):
            yield f"no step runs {item.nodeid}"
        if sum(selects(step, markers(item)) for step in exclusive) > 1:
            yield f"two steps run {item.nodeid}"


@pytest.mark.infra
@pytest.mark.parametrize("checked", RULES, ids=[r.name for r in RULES])
def test_rule(checked: Rule, request: pytest.FixtureRequest) -> None:
    """`checked` finds exactly the exceptions it declares."""
    if checked.reads == COLLECTION:
        result = checked.check(request.getfixturevalue("collected_items"))
    else:
        result = checked.check()
    found = dict(result) if isinstance(result, Mapping) else dict.fromkeys(result, "")
    new = {key: value for key, value in found.items() if key not in checked.declared}
    stale = sorted(set(checked.declared) - found.keys())

    assert not new, f"{checked.name}, reading {checked.reads}: {new}"
    assert not stale, f"{checked.name}: declared, no longer found, remove: {stale}"
