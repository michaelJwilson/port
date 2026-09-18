"""Every compiled kernel in `cnaster` is named, and says who validates it.

**Coverage cannot make this claim.** `numba` reports no line information, so
an `@njit` function reads as uncovered however hard it is exercised and as
untested when it is not — the report says the same thing either way, which is
why `CLAUDE.md` requires these tested against an independent reference
regardless of what the figure does.

So the claim is made here instead, and it is a human one: the registry says
which test validates each kernel. What this module enforces is that the claim
is *complete* and *current* — every kernel the package ships is classified, a
new one fails until someone classifies it, every named test exists, and every
kernel without one names the ticket that owns the gap.

It cannot enforce that a named test is a good test. Nothing can; that is what
the marker on the test itself is for.
"""

import ast
import pathlib

import pytest

SUBJECT = "cnaster"

OUT_OF_SCOPE = frozenset({"deprecated", "sandbox"})
"""Trees the dependency has set aside, which `pyproject.toml` also omits."""

VALIDATED: dict[str, str] = {
    "hmm_emission:compute_bb_ab": (
        "tests/test_m_step.py::test_m_step_agrees_with_upstream"
    ),
    "hmm_nophasing:nbinom_logpmf_numba": (
        "tests/test_numba_kernels.py::test_negative_binomial_kernel_matches_scipy"
    ),
    "hmm_nophasing:betabinom_logpmf_numba": (
        "tests/test_numba_kernels.py::test_beta_binomial_kernel_matches_scipy"
    ),
    "hmm_nophasing:numba_logsumexp": (
        "tests/test_numba_kernels.py::test_numba_logsumexp_matches_scipy"
    ),
    "hmm_nophasing:_nb_logpmf_1d": (
        "tests/test_numba_kernels.py::test_dense_and_single_observation_kernels_agree"
    ),
    "hmm_nophasing:_dense_nb_logpmf": (
        "tests/test_numba_kernels.py::test_dense_and_single_observation_kernels_agree"
    ),
    "hmm_nophasing:_bb_logpmf_1d": (
        "tests/test_core_inference_fixture.py"
        "::test_cnaster_scores_the_fixture_as_upstream_does"
    ),
    "hmm_nophasing:_dense_bb_logpmf": (
        "tests/test_core_inference_fixture.py"
        "::test_the_field_recovers_the_planted_clone_assignment"
    ),
    "hmm_nophasing:compute_logmu_shifts": (
        "tests/test_logmu_shifts.py::test_matches_the_vectorized_reference"
    ),
    "hmm_nophasing:forward_lattice": (
        "tests/test_hmm_single_chain.py::test_total_log_likelihood_matches_upstream"
    ),
    "hmm_nophasing:backward_lattice": (
        "tests/test_lattice_invariants.py::test_forward_and_backward_agree_at_every_position"
    ),
    "hmm_phased:update_combined_transmat": (
        "tests/test_hmm_phased.py::test_combined_transition_matches_cnaster_construction"
    ),
    "hmm_phased:forward_lattice": (
        "tests/test_hmm_phased.py::test_total_log_likelihood_matches_upstream"
    ),
    "hmrf:compute_loglike_spot_assignment": (
        "tests/test_core_inference_fixture.py"
        "::test_the_field_recovers_the_planted_clone_assignment"
    ),
}
"""Kernel -> the test that checks its numbers against something outside it."""

UNVALIDATED: dict[str, str] = {
    "hmm_emission:collapse_exog": "the design matrix the BB M step folds; #94",
    "hmm_emission:betabinom_logpmf": (
        "the non-`_numba` beta-binomial in `hmm_emission`, distinct from the "
        "one `hmm_nophasing` ships and validated above; #94"
    ),
    "hmm_nophasing:np_sum_ax_squeeze": "an inlined reduction helper; #94",
    "hmm_phased:backward_lattice": (
        "the phased forward is refereed against upstream and the backward is "
        "not, which is the asymmetry `test_lattice_invariants` closes for the "
        "unphased pair and nothing closes for this one; #94"
    ),
    "hmm_phased:_switch_betabinom_1d": (
        "the phased channel is unreachable through its own classmethod; #9"
    ),
    "hmrf:logsumexp": "a third copy of the reduction, module-local; #94",
    "hmrf:pool_spatio_genomic_counts": "#94, and the pooling cost is #13",
    "icm:calc_cluster_assignment_cost": "#64 reduces this interface; #94",
    "icm:calc_assignment_cost": "#64 reduces this interface; #94",
    "icm:logsumexp": "a fourth copy of the reduction, module-local; #94",
    "icm:icm_sweep": (
        "the array solver, not the `_deque` one the live path takes; #64, #94"
    ),
    "normal_spot:compute_local_normal_mask": "#89",
    "normal_spot:_compute_baseline_core": "#89",
    "utils:top_hat_sum": "#94",
    "wolff:_wolff_annealing_core": "the module raises on import; #71",
    "scripts/run_cnaster:set_numba_seed": (
        "under `scripts/`, which has no `__init__.py` and so is outside the "
        "coverage denominator as well; #94"
    ),
}
"""Kernel -> why it has no validation yet, and the ticket that owns the gap."""


def _installed_kernels() -> dict[str, str]:
    """Every `@njit` function the installed package ships, by `module:name`.

    Read from the source with `ast` rather than by importing: a dispatcher
    object hides the decorator that made it, and importing every module to
    find out would execute the package to audit it.
    """
    import cnaster

    root = pathlib.Path(next(iter(cnaster.__path__)))
    kernels: dict[str, str] = {}

    for path in sorted(root.rglob("*.py")):
        if OUT_OF_SCOPE.intersection(path.relative_to(root).parts):
            continue
        module = path.relative_to(root).with_suffix("").as_posix()

        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                source = ast.unparse(decorator)
                if source == "njit" or source.startswith("njit("):
                    kernels[f"{module}:{node.name}"] = source

    return kernels


@pytest.mark.infra
@pytest.mark.analytic
def test_every_compiled_kernel_is_classified() -> None:
    """A kernel is validated or it names the ticket that owns the gap.

    The failure this exists for is a *new* `@njit` landing in a dependency
    upgrade: it arrives invisible to coverage, and without this it arrives
    invisible to review as well.
    """
    installed = set(_installed_kernels())
    classified = set(VALIDATED) | set(UNVALIDATED)

    assert not installed - classified, (
        f"compiled kernels neither validated nor excused: "
        f"{sorted(installed - classified)}"
    )
    assert not classified - installed, (
        f"the registry names kernels the package no longer ships: "
        f"{sorted(classified - installed)}"
    )


@pytest.mark.infra
@pytest.mark.analytic
def test_every_named_validation_test_exists(
    collected_items: list[pytest.Item],
) -> None:
    """The registry cannot point at a test that was renamed or deleted.

    Without this the entries decay into documentation: a test moves, the claim
    stays, and the registry says a kernel is validated by something that is not
    there. Checked against the collection, so a parametrized test matches by
    its base id.
    """
    collected = {item.nodeid.split("[")[0] for item in collected_items}

    missing = {
        kernel: node for kernel, node in VALIDATED.items() if node not in collected
    }

    assert not missing, f"registry points at tests that do not exist: {missing}"


@pytest.mark.infra
@pytest.mark.analytic
def test_every_gap_names_a_ticket() -> None:
    """An excuse without a ticket is a decision nobody will revisit."""
    import re

    unticketed = {
        kernel: reason
        for kernel, reason in UNVALIDATED.items()
        if not re.search(r"#\d+", reason)
    }

    assert not unticketed, f"gaps with no ticket: {unticketed}"


@pytest.mark.infra
@pytest.mark.analytic
def test_the_registry_is_not_empty_and_the_scan_found_kernels() -> None:
    """Each assertion above passes trivially over an empty scan.

    `_installed_kernels` reads a path that carries a Python version, so it can
    come back empty the same way the coverage source can, and every guard here
    would go green over nothing.
    """
    installed = _installed_kernels()

    assert len(installed) >= 25, f"the scan found {len(installed)} kernels"
    assert len(VALIDATED) >= 10
    assert all(source.startswith("njit") for source in installed.values())
