"""The oracle surface is matched **one way, from `cnaster`** (#128).

`.coveragerc-oracle` says which `snakes_and_ladders` modules the claims rest
on. What it cannot say by itself is *why* each one is there, and the direction
matters: a module enters because `cnaster` does something it referees, never
because upstream happens to ship it. Presence in `snakes_and_ladders` is not
entry.

Read the other way the surface would grow without bound and the figure would
fall for every module upstream adds -- a metric that moves on someone else's
commits. Read this way it moves only when `cnaster` gains a path or `port`
builds a rung, which is what `CLAUDE.md` asks a coverage number to mean.

`tests/test_coverage_scope.py` guards the other half: that no test referees
against a module the config does not declare. Together they are a biconditional
-- declared if and only if refereed, refereed only if `cnaster` matches.
"""

import ast
import configparser
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORACLE_CONFIG = PROJECT_ROOT / ".coveragerc-oracle"

CORRESPONDENCE: dict[str, tuple[str, ...]] = {
    "snakes_and_ladders.emissions": ("cnaster.hmm_nophasing", "cnaster.hmm_emission"),
    "snakes_and_ladders.opt.hmm": ("cnaster.hmm", "cnaster.hmm_nophasing"),
    "snakes_and_ladders.opt.fit": ("cnaster.integer_copy",),
    "snakes_and_ladders.ragged": ("cnaster.hmrf_utils",),
    "snakes_and_ladders.likelihood.ragged_rust": ("cnaster.hmm_nophasing",),
    "snakes_and_ladders.likelihood.forward_backward": ("cnaster.hmm_nophasing",),
    "snakes_and_ladders.likelihood.spatio_sequential": ("cnaster.hmrf",),
    "snakes_and_ladders.search.spatio_sequential": ("cnaster.hmrf",),
    "snakes_and_ladders.search.alpha_expansion": ("cnaster.icm",),
    "snakes_and_ladders.sim.count_pairs": ("cnaster.hmm_nophasing",),
    "snakes_and_ladders.sim.spatio_sequential": ("cnaster.hmrf",),
    "snakes_and_ladders.sim.graph": ("cnaster.hmrf_utils",),
    "snakes_and_ladders.sim.hmm": ("cnaster.hmm_nophasing",),
}
"""Each declared upstream module, and the `cnaster` code whose claim rests on it.

The value is the counterpart, not the test: a rung is built or not, and this
records what it would referee. Two modules are at zero coverage today
(`likelihood.spatio_sequential`, `search.spatio_sequential`) and are listed
because the correspondence exists and the rung does not -- which is #128's
largest opportunity, and the thing this table exists to keep visible.
"""

CNASTER_EMISSION_KERNELS = {"nb", "bb"}
"""The families `cnaster` implements, derived below rather than asserted.

`hmm_nophasing` defines `_nb_logpmf_1d` and `_bb_logpmf_1d` and nothing else
of the kind. Every other family upstream ships -- Categorical, Gaussian,
Poisson, Binomial, CountPair -- has no counterpart, so none of them is
refereeable and none of their statements belongs in an opportunity.
"""

UNMATCHED_FAMILIES = {
    "CategoricalEmission",
    "GaussianEmission",
    "PoissonEmission",
    "BinomialEmission",
    "CountPairEmission",
}
"""Upstream families with no `cnaster` counterpart: 377 statements, **excluded
from the denominator** by `.coveragerc-oracle`'s `exclude_also`.

The figure measures coverage against what `cnaster` can do, so capability
upstream adds that the subject has no counterpart for must not dilute it.
Excluding these took `emissions.py` from 779 statements to 402 and the surface
from 2,541 to 2,164 -- 37.78 to 38.63 per cent, for no change in what is
validated.

`CountPairEmission` is the one worth naming twice. `port` uses it -- the
fixtures draw counts through it -- but drawing is the `upstream` role and not
the `upstream_oracle` one (#69): it did not decide an expected value, so it
does not referee, so it does not enter.
"""


def _declared_modules() -> set[str]:
    """Dotted names from `.coveragerc-oracle`'s `[report] include` globs."""
    parser = configparser.ConfigParser()
    parser.read(ORACLE_CONFIG)
    declared = set()
    for line in parser["report"]["include"].splitlines():
        glob = line.strip()
        if not glob:
            continue
        tail = glob.split("snakes_and_ladders/", 1)[1].removesuffix(".py")
        declared.add("snakes_and_ladders." + tail.replace("/", "."))
    return declared


def _upstream_emissions_source() -> str:
    spec = importlib.util.find_spec("snakes_and_ladders.emissions")
    assert spec is not None
    assert spec.origin is not None
    return Path(spec.origin).read_text()


@pytest.mark.analytic
def test_every_declared_module_names_the_cnaster_code_it_referees() -> None:
    """No module joins the surface without a counterpart on the subject's side.

    The guard that makes the matching one-way. Adding a glob to
    `.coveragerc-oracle` and nothing else fails here, which is the case the
    rule exists to prevent: upstream's shape deciding what this repository
    measures.
    """
    declared = _declared_modules()
    undeclared = set(CORRESPONDENCE) - declared
    unmatched = declared - set(CORRESPONDENCE)

    assert not unmatched, f"declared with no stated cnaster counterpart: {unmatched}"
    assert not undeclared, f"matched but no longer declared: {undeclared}"


@pytest.mark.analytic
def test_every_cnaster_counterpart_still_exists() -> None:
    """A rename on the subject's side breaks the correspondence, loudly.

    `find_spec` rather than `import`: importing executes the module body and
    would lift the subject's coverage without a test having run, which is the
    same reason `test_coverage_scope.py` resolves its modules this way.
    """
    missing = {
        upstream: counterpart
        for upstream, counterparts in CORRESPONDENCE.items()
        for counterpart in counterparts
        if importlib.util.find_spec(counterpart) is None
    }

    assert not missing, f"cnaster counterparts that no longer resolve: {missing}"


@pytest.mark.cnaster
def test_cnaster_implements_exactly_two_emission_families() -> None:
    """**Derived from `cnaster`, not asserted about it (#128).**

    The one-way rule needs the subject's side measured rather than assumed, so
    this reads `hmm_nophasing` for its per-observation kernels and finds
    `_nb_logpmf_1d` and `_bb_logpmf_1d` -- negative binomial and beta-binomial,
    and nothing else.

    If `cnaster` ever grows a third, this fails and the family it corresponds
    to upstream becomes reachable for the first time. That is the only event
    that should widen the emission surface.
    """
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


@pytest.mark.analytic
def test_the_unmatched_emission_families_are_the_ones_named() -> None:
    """The emission surface's reachable part, pinned as a set rather than a hope.

    `emissions.py` is the largest module on the surface and most of it is
    unreachable: of 779 statements, 208 belong to the two matched families and
    the shared base, and 377 to families with no counterpart. So the ceiling
    on this module is **59 uncovered statements, not 401** -- 2.3 points of the
    surface rather than 15.8, which is the difference between an opportunity
    and a wish.

    Upstream adding a family turns this red; the fix is to list it, not to
    count it as reachable.
    """
    tree = ast.parse(_upstream_emissions_source())
    families = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Emission")
    }
    matched = {"NegativeBinomialEmission", "BetaBinomialEmission"}
    base = {"CountEmissionFamily", "EmissionFamily"}

    assert families - matched - base == UNMATCHED_FAMILIES, (
        f"upstream's families are {sorted(families)}"
    )


@pytest.mark.analytic
def test_every_unmatched_family_is_excluded_from_the_denominator() -> None:
    """The two lists cannot drift, which is what makes the figure stable.

    `UNMATCHED_FAMILIES` is derived from the two sources;
    `.coveragerc-oracle`'s `exclude_also` is what the report acts on. If they
    disagree, the denominator either counts capability `cnaster` cannot
    referee -- the dilution the exclusion exists to prevent -- or hides a
    family that does have a counterpart.

    With `test_the_unmatched_emission_families_are_the_ones_named`, a family
    added upstream fails here until it is either matched to `cnaster` code or
    excluded, so upstream growth cannot move this repository's number silently
    in either direction.
    """
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
