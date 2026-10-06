"""Every `port` module has one role, and lives where its role says (#517 E3).

`CLAUDE.md`: `patch/` replaces a named `cnaster` function, `extensions/` adds
what has no counterpart, and `sandbox/` holds what is set aside. The roles
below are that rule made checkable, each against the import graph
`tests.source_graph` reads from the source:

| role | where | checked |
| --- | --- | --- |
| `row` | `patch/` | defines what a swap row installs, and every such module is one |
| `row-helper` | `patch/` | reached from a row, a pipeline entry point or `pipeline` |
| `extension` | `extensions/` | reached from a row, a pipeline entry point or `pipeline` |
| `oracle` | `extensions/` | imported by an `end2end` or `oracle` test |
| `tool` | `extensions/`, `qa/`, `studies/` | reached from no row or pipeline entry point: a figure, record or measurement tool |
| `sim`, `script`, `pipeline` | `sim/`, `scripts/`, `port.pipeline` | where they are |
| `set aside` | `sandbox/` | installed by nothing |

**Live** is what the pipeline entry points reach: `PIPELINE` (`run_cnaster_port`,
`run_calicost`), `port.pipeline` and the rows. The QA entry points
(`run_ledger`, `run_audit`, `run_benchmark`, `run_figures`, `run_study`, T- #673) are `script`s too, but tools: they may reach `tool`
modules, and no pipeline entry point may.

`cnamaste` (T- #670) ships on its own and has its own scope at the end of
this file: `entry` and `copy` roles, reached from `run_cnamaste`, importing
neither `cnaster` nor `port`.

A module reached by nothing lives in `sandbox/`, mirroring the tree it left
(`sandbox/patch/...`, `sandbox/extensions/...`), so graduating is a move
back; #517 step 8 moved the last of them. Every sandbox module states its
ticket, measurement and exit, and a new one arrives with them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Literal

import pytest

from tests.source_graph import PACKAGES, ROOT, modules, reached, row_modules

Role = Literal[
    "row",
    "row-helper",
    "extension",
    "oracle",
    "tool",
    "sim",
    "script",
    "pipeline",
    "set aside",
]

ROLES: dict[str, Role] = {
    "port.pipeline": "pipeline",
    "port.scripts.run_calicost": "script",
    "port.scripts.run_cnaster": "script",
    "port.scripts.run_audit": "script",
    "port.scripts.run_benchmark": "script",
    "port.scripts.run_figures": "script",
    "port.scripts.run_ledger": "script",
    "port.scripts.run_study": "script",
    # extensions
    "port.extensions.adjacency": "extension",
    "port.extensions.combined_figure": "tool",
    "port.extensions.config_audit": "extension",
    "port.extensions.copy_errors": "extension",
    "port.extensions.copy_starts": "extension",
    "port.extensions.copy_likelihood": "extension",
    "port.extensions.emission_family": "oracle",
    "port.extensions.figure_style": "extension",
    "port.extensions.integer_copy": "oracle",
    "port.extensions.jax_hmm": "oracle",
    "port.extensions.jax_setup": "extension",
    "port.extensions.kronecker_posteriors": "oracle",
    "port.extensions.label_solver": "extension",
    "port.extensions.multisample": "extension",
    "port.extensions.outputs": "extension",
    "port.extensions.parameter_errors": "oracle",
    "port.extensions.realization_plot": "tool",
    "port.extensions.sal": "extension",
    "port.extensions.samples": "extension",
    "port.extensions.segments": "extension",
    "port.extensions.vocabulary": "tool",
    # qa: what measures and records a run (T- #673), reached from no row or script
    "port.qa.audit": "tool",
    "port.qa.benchmark": "tool",
    "port.qa.ledger": "tool",
    "port.qa.provenance": "tool",
    "port.qa.scoring": "tool",
    "port.qa.statistics": "tool",
    # studies: measurements run by hand (T- #673 G5), reached from `run_study`
    "port.studies.calicost_figures": "tool",
    "port.studies.clone_label_arms": "tool",
    "port.studies.clone_label_notebook": "tool",
    "port.studies.clone_labels": "tool",
    "port.studies.clone_starts": "tool",
    "port.studies.cna_lengths": "tool",
    "port.studies.copy_start_arms": "tool",
    "port.studies.copy_start_notebook": "tool",
    "port.studies.copy_starts": "tool",
    "port.studies.copy_state_plot": "tool",
    "port.studies.copy_state_stream": "tool",
    "port.studies.field_strength": "tool",
    "port.studies.figures": "tool",
    "port.studies.hmm_starts": "tool",
    "port.studies.metrics_history": "tool",
    "port.studies.paper_figures": "tool",
    "port.studies.population": "tool",
    "port.studies.population_report": "tool",
    "port.studies.potts_plot": "tool",
    "port.studies.potts_solvers": "tool",
    "port.studies.potts_stream": "tool",
    "port.studies.stream": "tool",
    # patch: rows
    "port.patch.hmm_nophasing.bb_logpmf": "row",
    "port.patch.hmm_nophasing.nb_logpmf": "row",
    "port.patch.hmm_nophasing.shifted_emission": "row",
    "port.patch.hmm_phased.coded_emission": "row",
    "port.patch.hmrf.clone_assignment": "row",
    "port.patch.hmrf.core_inference": "row",
    "port.patch.hmrf.field": "row",
    "port.patch.hmrf.refinement": "row",
    "port.patch.integer_copy": "row",
    "port.patch.io": "row",
    "port.patch.normal_spot": "row",
    "port.patch.omics.blocks": "row",
    "port.patch.omics.summaries": "row",
    "port.patch.plot_copy_number_profile": "row",
    "port.patch.plot_genomic": "row",
    "port.patch.plotting.spatial": "row",
    "port.patch.pseudobulk": "row",
    "port.patch.recomb": "row",
    "port.patch.reference": "row",
    "port.patch.spatial": "row",
    "port.patch.utils": "row",
    # patch: helpers
    "port.patch._signature": "row-helper",
    "port.patch.hmm_initialize.distinct": "row-helper",
    "port.patch.hmm_initialize.sal_mixture": "row-helper",
    "port.patch.hmm_nophasing.dense_emission": "row-helper",
    "port.patch.hmm_nophasing.gradient": "row-helper",
    "port.patch.hmm_nophasing.logmu_shift": "row-helper",
    "port.patch.hmrf.adjacency": "row-helper",
    "port.patch.hmrf.fused_field": "row-helper",
    "port.patch.hmrf.tabulated_field": "row-helper",
    "port.patch.icm.alpha_expansion": "row-helper",
    "port.patch.icm.floor": "row-helper",
    "port.patch.icm.interface": "row-helper",
    "port.patch.lattice": "row-helper",
    "port.patch.plotting.clone_paths": "row-helper",
    "port.patch.hmrf.invariants": "row-helper",
    "port.patch.hmrf.reindex": "row-helper",
    # sim
    "port.sim.analysis": "sim",
    "port.sim.fixtures": "sim",
    "port.sim.he_slide": "sim",
    "port.sim.inputs": "sim",
    "port.sim.realizations": "sim",
    "port.sim.run_config": "sim",
    "port.sim.truth": "sim",
    "port.sim.unsegment": "sim",
    "port.sim.draw": "sim",
    "port.sim.entries": "sim",
    "port.sim.files": "sim",
    "port.sim.kernels": "sim",
    "port.sim.laws": "sim",
    "port.sim.truth_figure": "sim",
    "port.sim.normal_fit": "sim",
    # sandbox
    "port.sandbox.admixture.clone_mixture": "set aside",
    "port.sandbox.known_field.cluster": "set aside",
    "port.sandbox.known_field.color_merge": "set aside",
    "port.sandbox.known_field.field": "set aside",
    "port.sandbox.admixture.probes.sim_probe": "set aside",
    "port.sandbox.admixture.variants": "set aside",
    "port.sandbox.clone_starts.problem": "set aside",
    "port.sandbox.clone_starts.starts": "set aside",
    "port.sandbox.extensions.copy_starts": "set aside",
    "port.sandbox.extensions.hmm_init_trials": "set aside",
    "port.sandbox.integer_decoding.calicost_decoders": "set aside",
    "port.sandbox.integer_decoding.rdr_summary": "set aside",
    "port.sandbox.integer_decoding.schemes": "set aside",
    "port.sandbox.known_copy.hmm": "set aside",
    "port.sandbox.known_copy.hmm_objective": "set aside",
    "port.sandbox.known_copy.problem": "set aside",
    "port.sandbox.normal_candidates": "set aside",
    "port.sandbox.np_merge.__main__": "set aside",
    "port.sandbox.np_merge.merge": "set aside",
    "port.sandbox.patch.emission": "set aside",
    "port.sandbox.patch.hmm_initialize.backends": "set aside",
    "port.sandbox.patch.hmm_initialize.filtering": "set aside",
    "port.sandbox.patch.plotting.genomic": "set aside",
    "port.sandbox.patch.plotting.loh_density": "set aside",
    "port.sandbox.sal_hmm_init": "set aside",
    "port.sandbox.sim_from_run": "set aside",
    "port.sandbox.wolff_init": "set aside",
    "port.sandbox.wolff_umi_init": "set aside",
}
"""Every module that is not a package `__init__`, by role."""

WHERE: dict[Role, tuple[str, ...]] = {
    "row": ("port.patch.",),
    "row-helper": ("port.patch.",),
    "extension": ("port.extensions.",),
    "oracle": ("port.extensions.",),
    "tool": ("port.extensions.", "port.qa.", "port.studies."),
    "sim": ("port.sim.",),
    "script": ("port.scripts.",),
    "pipeline": ("port.pipeline",),
    "set aside": ("port.sandbox.",),
}

SANDBOX_HEADERLESS: frozenset[str] = frozenset()
"""Sandbox modules without the header: none since #517 step 8."""

HEADER = ("Ticket:", "Measurement:", "Exit:")
"""What a sandbox module's docstring states, one line each."""


def _by(role: Role) -> set[str]:
    return {name for name, declared in ROLES.items() if declared == role}


PIPELINE = frozenset({"port.scripts.run_calicost", "port.scripts.run_cnaster"})
"""The pipeline entry points: what a user runs to infer copy numbers. Every
other `script` is a QA tool entry point (T- #673)."""


def _live() -> frozenset[str]:
    roots = {"port.pipeline"} | PIPELINE | row_modules()
    return reached(roots)


@pytest.mark.infra
def test_every_module_has_a_role() -> None:
    found = {name for name, path in modules().items() if path.name != "__init__.py"}

    assert found == set(ROLES), (
        f"no role: {sorted(found - set(ROLES))}; gone: {sorted(set(ROLES) - found)}"
    )


@pytest.mark.infra
def test_every_module_lives_where_its_role_says() -> None:
    misplaced = {
        name: f"{role} lives under {WHERE[role]}"
        for name, role in ROLES.items()
        if not name.startswith(WHERE[role])
    }

    assert not misplaced, misplaced


@pytest.mark.infra
def test_the_rows_are_the_modules_the_tables_install_from() -> None:
    """Replaces the three hard-coded names #517 E found: every row, both ways.

    An extension installed over a `cnaster` name is a patch that has not
    admitted to being one; a `row` no table installs from replaces nothing.
    """
    assert row_modules() == _by("row"), (
        f"installed, not a row: {sorted(row_modules() - _by('row'))}; "
        f"a row nothing installs: {sorted(_by('row') - row_modules())}"
    )


@pytest.mark.infra
def test_live_and_set_aside_are_what_the_graph_says() -> None:
    """`extension` and `row-helper` are reached; `tool` and `set aside` are not.

    A `tool` reached from a script is an extension, and a `set aside` module
    reached from a run has graduated without saying so.
    """
    live = _live()

    assert _by("script") >= PIPELINE, sorted(PIPELINE - _by("script"))
    assert _by("extension") <= live, sorted(_by("extension") - live)
    assert _by("row-helper") <= live, sorted(_by("row-helper") - live)
    assert not (_by("tool") & live), sorted(_by("tool") & live)
    assert not (_by("set aside") & live), sorted(_by("set aside") & live)


@pytest.mark.infra
def test_every_oracle_referees_a_counting_test() -> None:
    from tests.source_graph import COUNTING, TESTS

    referred: set[str] = set()

    for path in TESTS.rglob("test_*.py"):
        source = path.read_text()

        if not any(f"mark.{marker}" in source for marker in COUNTING):
            continue

        referred |= {name for name in _by("oracle") if name in source}

    assert _by("oracle") <= referred, sorted(_by("oracle") - referred)


@pytest.mark.infra
def test_every_sandbox_module_states_its_ticket_measurement_and_exit() -> None:
    """Headerless is a declared list: a new module arrives with its header."""
    headed = set()

    for name in _by("set aside"):
        docstring = ast.get_docstring(ast.parse(modules()[name].read_text())) or ""
        lines = docstring.splitlines()

        if all(any(line.startswith(key) for line in lines) for key in HEADER):
            headed.add(name)

    headerless = _by("set aside") - headed

    assert headerless == set(SANDBOX_HEADERLESS), (
        f"missing a header: {sorted(headerless - SANDBOX_HEADERLESS)}; "
        f"headed, remove from SANDBOX_HEADERLESS: {sorted(SANDBOX_HEADERLESS - headerless)}"
    )


@pytest.mark.infra
def test_no_live_module_imports_the_sandbox() -> None:
    """`sandbox/` is installed by nothing, so nothing outside it imports it (T- #617).

    A study is the exception: `studies/` measures what `sandbox/` set aside,
    which is each sandbox module's stated measurement, and is reached from
    no pipeline entry point (T- #673 G5), so its imports install nothing.
    """
    package = Path(__file__).resolve().parents[1] / "python" / "port"
    found = []

    for path in sorted(package.rglob("*.py")):
        if {"sandbox", "studies"} & set(path.relative_to(package).parts):
            continue

        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            found += [
                f"{path.relative_to(package)}: {name}"
                for name in names
                if name == "port.sandbox" or name.startswith("port.sandbox.")
            ]

    assert found == []


# --- `cnamaste`: its own scope (T- #670) -----------------------------------
#
# `cnamaste` ships on its own, so it is not a `port` module and takes none of
# `port`'s roles. Two of its own: `entry`, what its build declares as a
# console script; `copy`, a `cnaster` module copied from the locked pin with
# only its live `cnaster` imports rewritten. A later T- #670 PR that changes a
# module gives it a role saying how it departs.

CnamasteRole = Literal["entry", "copy"]

CNAMASTE_ROLES: dict[str, CnamasteRole] = {
    "cnamaste.run": "entry",
    "cnamaste.annotation": "copy",
    "cnamaste.cna_hmrf_result": "copy",
    "cnamaste.config": "copy",
    "cnamaste.count_encoder": "copy",
    "cnamaste.filter": "copy",
    "cnamaste.he": "copy",
    "cnamaste.hmm": "copy",
    "cnamaste.hmm_emission": "copy",
    "cnamaste.hmm_initialize": "copy",
    "cnamaste.hmm_nophasing": "copy",
    "cnamaste.hmm_phased": "copy",
    "cnamaste.hmm_utils": "copy",
    "cnamaste.hmrf": "copy",
    "cnamaste.hmrf_utils": "copy",
    "cnamaste.icm": "copy",
    "cnamaste.integer_copy": "copy",
    "cnamaste.io": "copy",
    "cnamaste.logger": "copy",
    "cnamaste.normal_spot": "copy",
    "cnamaste.omics": "copy",
    "cnamaste.palette": "copy",
    "cnamaste.phasing": "copy",
    "cnamaste.plot_copy_number_profile": "copy",
    "cnamaste.plot_genomic": "copy",
    "cnamaste.plotting": "copy",
    "cnamaste.pseudobulk": "copy",
    "cnamaste.recomb": "copy",
    "cnamaste.reference": "copy",
    "cnamaste.spatial": "copy",
    "cnamaste.spatio_genomic_counts": "copy",
    "cnamaste.utils": "copy",
}
"""Every `cnamaste` module that is not its package `__init__`, by role."""

CNAMASTE_ENTRIES = {"run_cnamaste": "cnamaste.run:main"}
"""The console scripts `python/cnamaste/pyproject.toml` declares."""

PIN = "4adad4d"
"""The `cnaster` commit the copy was taken from: `uv.lock`'s pin."""

_LIVE_CNASTER_IMPORT = re.compile(r"^(\s*)(from|import) cnaster(\.|\s)", re.MULTILINE)


def _cnamaste_by(role: CnamasteRole) -> set[str]:
    return {name for name, declared in CNAMASTE_ROLES.items() if declared == role}


def _imported(path: Path) -> set[str]:
    """Every module name `path` imports, at any depth of its body."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.add(node.module or "")
    return names


@pytest.mark.infra
def test_every_cnamaste_module_has_a_role() -> None:
    found = {
        name for name, path in modules("cnamaste").items() if path.name != "__init__.py"
    }

    assert found == set(CNAMASTE_ROLES), (
        f"no role: {sorted(found - set(CNAMASTE_ROLES))}; "
        f"gone: {sorted(set(CNAMASTE_ROLES) - found)}"
    )


@pytest.mark.infra
def test_the_cnamaste_entries_are_what_its_build_declares() -> None:
    """`run_cnamaste` is declared by `cnamaste`'s own build, and nowhere else."""
    import tomllib

    own = tomllib.loads((PACKAGES["cnamaste"] / "pyproject.toml").read_text())
    port = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert own["project"]["scripts"] == CNAMASTE_ENTRIES
    assert not set(CNAMASTE_ENTRIES) & set(port["project"]["scripts"])
    assert {v.partition(":")[0] for v in CNAMASTE_ENTRIES.values()} == _cnamaste_by(
        "entry"
    )


@pytest.mark.infra
def test_every_cnamaste_module_is_reached_from_its_entry() -> None:
    """The copy is the forward path: nothing in it that `run_cnamaste` cannot reach."""
    live = reached(_cnamaste_by("entry"))

    assert set(CNAMASTE_ROLES) <= live, sorted(set(CNAMASTE_ROLES) - live)


@pytest.mark.infra
def test_cnamaste_imports_neither_cnaster_nor_port() -> None:
    """It ships on its own: an import of either would make one a dependency."""
    found = sorted(
        f"{name}: {imported}"
        for name, path in modules("cnamaste").items()
        for imported in _imported(path)
        if imported.partition(".")[0] in {"cnaster", "port"}
    )

    assert found == []


@pytest.mark.infra
def test_every_copy_is_the_pinned_cnaster_module_with_its_imports_rewritten() -> None:
    """A `copy` reads, byte for byte, as the installed `cnaster` module at the
    pin with each live `from cnaster.` / `import cnaster.` naming `cnamaste`.
    The entry is `cnaster/scripts/run_cnaster.py`'s copy."""
    import json
    from importlib.metadata import distribution

    import cnaster

    pinned = json.loads(distribution("cnaster").read_text("direct_url.json") or "{}")
    assert pinned["vcs_info"]["commit_id"].startswith(PIN)

    installed = Path(next(iter(cnaster.__path__)))
    differ = []
    for name in {*_cnamaste_by("copy"), *_cnamaste_by("entry")}:
        stem = name.removeprefix("cnamaste.")
        source = installed / (
            "scripts/run_cnaster.py" if stem == "run" else f"{stem}.py"
        )
        rewritten = _LIVE_CNASTER_IMPORT.sub(r"\1\2 cnamaste\3", source.read_text())
        if rewritten != modules("cnamaste")[name].read_text():
            differ.append(name)

    assert differ == []
