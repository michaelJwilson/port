"""Every `port` module has one declared role and lives where it says (#517 E3).

Roles are checked against the import graph `tests.source_graph` reads; live is
what `PIPELINE`, `port.pipeline` and the patch rows reach (T- #673).
"""

from __future__ import annotations

import ast
from typing import Literal

import pytest

from tests import ROOT
from tests.source_graph import modules, reached, row_modules

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
    "port.qa.scripts.run_audit": "script",
    "port.qa.scripts.run_benchmark": "script",
    "port.qa.scripts.run_calibrate": "script",
    "port.qa.scripts.run_figures": "script",
    "port.scripts.run_plots": "script",
    "port.qa.scripts.run_ledger": "script",
    "port.qa.scripts.run_study": "script",
    # extensions
    "port.extensions.adjacency": "extension",
    "port.qa.combined_figure": "tool",
    "port.extensions.cnamaste": "extension",
    "port.extensions.config_audit": "extension",
    "port.extensions.figure_record": "extension",
    "port.sandbox.extensions.copy_errors": "set aside",
    "port.extensions.copy_starts": "extension",
    "port.extensions.copy_likelihood": "extension",
    "port.qa.emission_family": "oracle",
    "port.extensions.figure_style": "extension",
    "port.qa.spatial_page": "tool",
    "port.extensions.repository": "extension",
    "port.extensions.run_record": "extension",
    "port.extensions.genomic_axis": "extension",
    "port.extensions.integer_copy": "extension",
    "port.qa.jax_hmm": "tool",
    "port.qa.jax_setup": "tool",
    "port.qa.kronecker_posteriors": "oracle",
    "port.extensions.label_solver": "extension",
    "port.extensions.multisample": "extension",
    "port.extensions.outputs": "extension",
    "port.qa.parameter_errors": "tool",
    "port.qa.realization_plot": "tool",
    "port.extensions.sal": "extension",
    "port.extensions.samples": "extension",
    "port.extensions.segments": "extension",
    "port.qa.vocabulary": "tool",
    # qa: run measurement and records (T- #673)
    "port.qa.audit": "tool",
    "port.qa.errors": "tool",
    "port.qa.benchmark": "tool",
    "port.qa.ledger": "tool",
    "port.qa.provenance": "tool",
    "port.qa.scoring": "tool",
    "port.qa.statistics": "tool",
    # studies: run by hand from `run_study` (T- #673 G5)
    "port.studies.benchmark_table": "tool",
    "port.studies.calicost_figures": "tool",
    "port.studies.clone_label_arms": "tool",
    "port.studies.clone_label_notebook": "tool",
    "port.studies.clone_labels": "tool",
    "port.studies.clone_starts": "tool",
    "port.studies.cna_lengths": "tool",
    "port.studies.copy_start_arms": "tool",
    "port.studies.copy_start_notebook": "tool",
    "port.studies.notebook": "tool",
    "port.studies.copy_starts": "tool",
    "port.studies.copy_state_plot": "tool",
    "port.studies.copy_state_stream": "tool",
    "port.studies.field_strength": "tool",
    "port.studies.figures": "tool",
    "port.studies.metrics_history": "tool",
    "port.studies.paper_figures": "tool",
    "port.studies.population": "tool",
    "port.studies.population_report": "tool",
    "port.studies.potts_plot": "tool",
    "port.studies.potts_stream": "tool",
    "port.qa.records": "tool",
    "port.qa.stage": "tool",
    "port.qa.stream": "tool",
    # patch: rows
    "port.patch.hmm_nophasing.bb_logpmf": "row",
    "port.patch.hmm_nophasing.nb_logpmf": "row",
    "port.patch.hmm_nophasing.shifted_emission": "row",
    "port.patch.hmm_phased.coded_emission": "row",
    "port.patch.he": "row",
    "port.patch.count_encoder": "row",
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
    "port.patch.emission": "row-helper",
    "port.patch._clone_paths": "row-helper",
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
    "port.sandbox.extensions.label_solvers": "set aside",
    "port.sandbox.extensions.segment_sets": "set aside",
    "port.studies.color_merge": "tool",
    "port.sandbox.admixture.probes.sim_probe": "set aside",
    "port.sandbox.admixture.variants": "set aside",
    "port.sandbox.clone_starts.problem": "set aside",
    "port.sandbox.clone_starts.starts": "set aside",
    "port.sandbox.extensions.copy_starts": "set aside",
    "port.sandbox.extensions.hmm_init_trials": "set aside",
    "port.sandbox.integer_decoding.calicost_decoders": "set aside",
    "port.sandbox.integer_decoding.rdr_summary": "set aside",
    "port.sandbox.integer_decoding.schemes": "set aside",
    "port.sandbox.extensions.hmm_objective": "set aside",
    "port.sandbox.normal_candidates": "set aside",
    "port.sandbox.np_merge": "set aside",
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
    "port.sandbox.copy_audit": "set aside",
    "port.sandbox.population_sets": "set aside",
    "port.sandbox.extensions.shared_decode": "set aside",
    "port.sandbox.wolff_umi_init": "set aside",
}
"""Every module by role; a package `__init__` counts once it defines something."""

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

SANDBOX_HEADERLESS: frozenset[str] = frozenset()
"""Sandbox modules without the header: none since #517 step 8."""

HEADER = ("Ticket:", "Measurement:", "Exit:")
"""What a sandbox module's docstring states, one line each."""


def _by(role: Role) -> set[str]:
    return {name for name, declared in ROLES.items() if declared == role}


PIPELINE = frozenset({"port.scripts.run_calicost", "port.scripts.run_cnaster"})
"""The pipeline entry points; every other `script` is a QA tool (T- #673)."""


def _live() -> frozenset[str]:
    roots = {"port.pipeline"} | PIPELINE | row_modules()
    return reached(roots)


def _has_role(name: str, source: str) -> bool:
    """Whether a file has a role: any module, or an `__init__` defining something."""
    return name != "__init__.py" or any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for node in ast.parse(source).body
    )


@pytest.mark.infra
def test_every_module_has_a_role() -> None:
    found = {
        name
        for name, path in modules().items()
        if _has_role(path.name, path.read_text())
    }

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
    """Rows equal the modules the swap tables install from, both ways (#517 E)."""
    assert row_modules() == _by("row"), (
        f"installed, not a row: {sorted(row_modules() - _by('row'))}; "
        f"a row nothing installs: {sorted(_by('row') - row_modules())}"
    )


@pytest.mark.infra
def test_live_and_set_aside_are_what_the_graph_says() -> None:
    """`extension` and `row-helper` are reached from live roots; others are not."""
    live = _live()

    assert _by("script") >= PIPELINE, sorted(PIPELINE - _by("script"))
    assert _by("extension") <= live, sorted(_by("extension") - live)
    assert _by("row-helper") <= live, sorted(_by("row-helper") - live)
    assert not (_by("tool") & live), sorted(_by("tool") & live)
    # NB an oracle the run reaches is not independent of it (#749 WP8)
    assert not (_by("oracle") & live), sorted(_by("oracle") & live)
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
    """No module outside `sandbox/` and `studies/` imports the sandbox (T- #617, #673)."""
    package = ROOT / "python" / "port"
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
