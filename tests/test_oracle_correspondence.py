"""`.coveragerc-oracle`'s surface is matched one way, from `cnaster` (#128).

A `sal` module is declared only for the `cnaster` code it referees; checked
against the config, `cnaster`'s source and `sal.emissions`.
"""

import ast
import configparser
import importlib.util
from pathlib import Path

import pytest

from tests import ROOT

ORACLE_CONFIG = ROOT / ".coveragerc-oracle"

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
"""Each declared upstream module and the `cnaster` code whose claim rests on it."""

CNASTER_EMISSION_KERNELS = {"nb", "bb"}
"""The emission families `cnaster` implements, derived from its kernels (#232)."""

UNMATCHED_FAMILIES = {
    "CategoricalEmission",
    "GaussianEmission",
    "PoissonEmission",
    "BinomialEmission",
    "RateConcentrationBetaBinomialEmission",
    "RateConcentrationCountPairEmission",
}
"""Upstream families with no `cnaster` counterpart, excluded from the denominator (T- #707)."""


def _declared_modules() -> set[str]:
    """Return dotted names from `.coveragerc-oracle`'s `[report] include` globs."""
    parser = configparser.ConfigParser()
    parser.read(ORACLE_CONFIG)
    declared = set()
    for line in parser["report"]["include"].splitlines():
        glob = line.strip()
        if not glob:
            continue
        tail = glob.split("sal/", 1)[1].removesuffix("/*").removesuffix(".py")
        declared.add("sal." + tail.replace("/", "."))
    return declared


def _upstream_emissions_source() -> str:
    spec = importlib.util.find_spec("sal.emissions")
    assert spec is not None
    assert spec.origin is not None
    # NB a package since e0aeb19 (#410): families span its modules
    return "\n".join(
        p.read_text() for p in sorted(Path(spec.origin).parent.glob("*.py"))
    )


@pytest.mark.infra
def test_every_declared_module_names_the_cnaster_code_it_referees() -> None:
    """Declared modules equal `CORRESPONDENCE`'s keys, both ways."""
    declared = _declared_modules()
    undeclared = set(CORRESPONDENCE) - declared
    unmatched = declared - set(CORRESPONDENCE)

    assert not unmatched, f"declared with no stated cnaster counterpart: {unmatched}"
    assert not undeclared, f"matched but no longer declared: {undeclared}"


@pytest.mark.infra
def test_every_cnaster_counterpart_still_exists() -> None:
    """Every `cnaster` counterpart resolves via `find_spec`, without importing it."""
    missing = {
        upstream: counterpart
        for upstream, counterparts in CORRESPONDENCE.items()
        for counterpart in counterparts
        if importlib.util.find_spec(counterpart) is None
    }

    assert not missing, f"cnaster counterparts that no longer resolve: {missing}"


@pytest.mark.infra
def test_cnaster_implements_exactly_two_emission_families() -> None:
    """`hmm_nophasing`'s `_*_logpmf_1d` kernels are exactly `nb` and `bb` (#128)."""
    spec = importlib.util.find_spec("cnaster.hmm_nophasing")
    assert spec is not None
    assert spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text())

    kernels = {
        node.name.removeprefix("_").removesuffix("_logpmf_1d")
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_logpmf_1d")
    }

    assert kernels == CNASTER_EMISSION_KERNELS, f"cnaster's kernels are {kernels}"


@pytest.mark.infra
def test_the_unmatched_emission_families_are_the_ones_named() -> None:
    """Upstream's unmatched emission families equal `UNMATCHED_FAMILIES`."""
    tree = ast.parse(_upstream_emissions_source())
    families = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Emission")
    }
    matched = {
        "NegativeBinomialEmission",
        "BetaBinomialEmission",
        # NB matched on #232: referees `cnaster`'s kernels to 1.4e-13 (#69)
        "CountPairEmission",
    }
    base = {"CountEmissionFamily", "EmissionFamily"}

    assert families - matched - base == UNMATCHED_FAMILIES, (
        f"upstream's families are {sorted(families)}"
    )


@pytest.mark.infra
def test_every_unmatched_family_is_excluded_from_the_denominator() -> None:
    """`exclude_also` names exactly `UNMATCHED_FAMILIES`."""
    parser = configparser.ConfigParser()
    parser.read(ORACLE_CONFIG)
    excluded = {
        line.strip().removeprefix("^class ")
        for line in parser["report"]["exclude_also"].splitlines()
        if line.strip()
    }

    assert excluded == UNMATCHED_FAMILIES, (
        f"exclude_also names {sorted(excluded)}, "
        f"unmatched are {sorted(UNMATCHED_FAMILIES)}"
    )
