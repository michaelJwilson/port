"""Every `port` module has one role, and lives where its role says (#517 E3).

`CLAUDE.md`: `patch/` replaces a named `cnaster` function, `extensions/` adds
what has no counterpart, and `sandbox/` holds what is set aside. The roles
below are that rule made checkable, each against the import graph
`tests.source_graph` reads from the source:

| role | where | checked |
| --- | --- | --- |
| `row` | `patch/` | defines what a swap row installs, and every such module is one |
| `row-helper` | `patch/` | reached from a row, a script or `pipeline` |
| `extension` | `extensions/` | reached from a row, a script or `pipeline` |
| `oracle` | `extensions/` | imported by an `end2end` or `oracle` test |
| `tool` | `extensions/` | reached from no row or script: a figure or record tool |
| `sim`, `script`, `pipeline` | `sim/`, `scripts/`, `port.pipeline` | where they are |
| `set aside` | `sandbox/` | installed by nothing |

A module reached by nothing lives in `sandbox/`, mirroring the tree it left
(`sandbox/patch/...`, `sandbox/extensions/...`), so graduating is a move
back; #517 step 8 moved the last of them. Every sandbox module states its
ticket, measurement and exit, and a new one arrives with them.
"""

from __future__ import annotations

import ast
from typing import Literal

import pytest

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
    # extensions
    "port.extensions.adjacency": "extension",
    "port.extensions.combined_figure": "tool",
    "port.extensions.config_audit": "extension",
    "port.extensions.copy_errors": "extension",
    "port.extensions.copy_starts": "tool",
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
    "port.extensions.segments": "extension",
    "port.extensions.vocabulary": "tool",
    # patch: rows
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
    "port.patch.hmm_nophasing.nb_logpmf": "row-helper",
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
    "port.sim.draw": "sim",
    "port.sim.entries": "sim",
    "port.sim.files": "sim",
    "port.sim.kernels": "sim",
    "port.sim.laws": "sim",
    "port.sim.truth_figure": "sim",
    "port.sim.normal_fit": "sim",
    # sandbox
    "port.sandbox.admixture.clone_mixture": "set aside",
    "port.sandbox.admixture.probes.sim_probe": "set aside",
    "port.sandbox.admixture.variants": "set aside",
    "port.sandbox.extensions.hmm_init_trials": "set aside",
    "port.sandbox.integer_decoding.calicost_decoders": "set aside",
    "port.sandbox.integer_decoding.rdr_summary": "set aside",
    "port.sandbox.integer_decoding.schemes": "set aside",
    "port.sandbox.known_copy.hmm": "set aside",
    "port.sandbox.known_copy.problem": "set aside",
    "port.sandbox.normal_candidates": "set aside",
    "port.sandbox.np_merge.__main__": "set aside",
    "port.sandbox.np_merge.merge": "set aside",
    "port.sandbox.patch.emission": "set aside",
    "port.sandbox.patch.hmm_nophasing.nb_logpmf": "set aside",
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
    "tool": ("port.extensions.",),
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


def _live() -> frozenset[str]:
    roots = {"port.pipeline"} | _by("script") | row_modules()
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
