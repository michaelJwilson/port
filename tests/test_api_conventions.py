"""`CLAUDE.md`'s API Conventions, read from the source.

A convention without a guard drifts, so each one that can be read from an
annotation, a name or a field is read here. Only what `port` owns is held to
them: a name `cnaster` defines is a drop-in and keeps `cnaster`'s signature.

`KNOWN` is every current departure, each with the finding in #401 that
converges it. It only shrinks: an entry that no longer occurs fails, so the
fix and the deletion land together. A new departure fails with the term that
replaces it.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from port.extensions.vocabulary import TERMS, replaced_by

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "python" / "port"
SKIPPED = {"sandbox", "deprecated", "tests", "__pycache__"}

KNOWN: dict[str, str] = {
    # --- F1, F8, F10: the label-solver seam ---------------------------------
    "port.patch.icm.interface:IcmResult field cost": "F1",
    "port.patch.icm.interface:IcmResult field niter": "F10",
    "port.patch.icm.alpha_expansion:potts_graph_from arg beta": "F2",
    "port.patch.icm.alpha_expansion:potts_energy arg beta": "F2",
    "port.patch.icm.alpha_expansion:alpha_expansion_sweep arg beta": "F2",
    "port.patch.icm.alpha_expansion:alpha_expansion_sweep arg tol": "F8",
    "port.extensions.label_solver:expansion_then_floor arg beta": "F2",
    "port.extensions.label_solver:sal_icm_sweep arg beta": "F2",
    "port.extensions.label_solver:sal_icm_sweep arg tol": "F8",
    "port.extensions.label_solver:expansion_then_floor sibling": "F8",
    "port.extensions.label_solver:sal_icm_sweep sibling": "F8",
    # NB the seam's two floored sal solvers (#410 step 3): the caller passes
    #    `beta` and `tol` by name, so they carry the seam's words until F2/F8
    #    rename all five together.
    "port.extensions.label_solver:expansion_then_merge arg beta": "F2",
    "port.extensions.label_solver:expansion_then_merge arg tol": "F8",
    "port.extensions.label_solver:sal_icm_floor_sweep arg beta": "F2",
    "port.extensions.label_solver:sal_icm_floor_sweep arg tol": "F8",
    "port.extensions.label_solver:expansion_then_merge sibling": "F8",
    "port.extensions.label_solver:sal_icm_floor_sweep sibling": "F8",
    "port.patch.icm.alpha_expansion:alpha_expansion_sweep sibling": "F8",
    # --- F2, F3, F5: count-kernel and data names ----------------------------
    "port.patch.normal_spot:cumulative_and_mass arg alpha": "F3",
    "port.patch.normal_spot:cumulative_and_mass arg beta": "F2",
    "port.patch.normal_spot:removal_indicator arg alpha": "F3",
    "port.patch.normal_spot:removal_indicator arg beta": "F2",
    "port.extensions.integer_copy:success_probability_variance arg alpha": "F3",
    "port.extensions.integer_copy:success_probability_variance arg beta": "F2",
    "port.patch.hmm_nophasing.nb_logpmf:nb_logpmf_1d arg alpha": "F3",
    "port.extensions.copy_likelihood:Pseudobulk field alpha": "F3",
    "port.extensions.copy_likelihood:Pseudobulk field tau": "F5",
    "port.extensions.copy_likelihood:Pseudobulk field total_bb_rd": "F5",
    "port.extensions.copy_likelihood:Pseudobulk field log_lambda": "F5",
    "port.extensions.parameter_errors:shift_weights arg log_mus": "F5",
    "port.patch.hmm_nophasing.logmu_shift:shifts arg log_mus": "F5",
    "port.patch.hmm_initialize.filtering:FilterRecord field n_bins": "F5",
    "port.patch.plotting.loh_density:loh_model arg n_bins": "F5",
    "port.extensions.outputs:states arg fit": "F6",
    "port.extensions.outputs:binlevel arg fit": "F6",
    "port.extensions.outputs:segments arg fit": "F6",
    "port.patch.hmrf.core_inference:pin_neutral arg result": "F6",
    # --- F4, F9, F14: HMM initialization ------------------------------------
    "port.extensions.hmm_init_trials:Trial field score": "F4",
    "port.extensions.hmm_init_trials:Trial field result": "F6",
    "port.extensions.hmm_init_trials:Trial field seed": "F9",
    "port.patch.hmm_initialize.backends:Selection field score": "F4",
    "port.patch.hmm_initialize.backends:cnaster_gmm_backend arg seed": "F9",
    "port.patch.hmm_initialize.backends:sal_emission_backend arg seed": "F9",
    "port.patch.hmm_initialize.backends:cnaster_gmm_backend sibling": "F9",
    "port.sim.run_sim_gen:generate arg seed": "F9",
    "port.sim.manifest:simulation_manifest arg random_state": "F9",
    # --- F11: integer copies ------------------------------------------------
    "port.extensions.copy_likelihood:decode arg max_passes": "F11",
    "port.extensions.copy_likelihood:Decoded termination": "F11",
    "port.patch.icm.interface:IcmResult termination": "F10",
}
"""Departures found by #401's audit, keyed `module:name kind word`."""

SIBLINGS: dict[str, tuple[str, ...]] = {
    "port.patch.icm.interface:icm_sweep": (
        "port.patch.icm.alpha_expansion:alpha_expansion_sweep",
        "port.extensions.label_solver:sal_icm_sweep",
        "port.extensions.label_solver:expansion_then_floor",
        "port.extensions.label_solver:expansion_then_merge",
        "port.extensions.label_solver:sal_icm_floor_sweep",
    ),
    "port.patch.hmm_initialize.backends:sal_emission_backend": (
        "port.patch.hmm_initialize.backends:cnaster_gmm_backend",
    ),
}
"""Entry points one setting chooses between, each against the first: the same
arguments in the same order, and the same result."""

ITERATION_FIELDS = {"niter", "iterations", "n_iter", "passes", "cycles", "converged"}
"""A result carrying one of these reports an iterative run."""

TENSOR = re.compile(r"\b(torch|jnp|jax)\.")


@dataclass(frozen=True)
class Entry:
    """One public callable or result class `port` owns."""

    module: str
    name: str
    node: ast.FunctionDef | ast.ClassDef


def _sources(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if not SKIPPED & set(path.relative_to(root).parts):
            yield path


def _installed(package: str) -> Path:
    module = __import__(package)
    return Path(next(iter(module.__path__)))


@cache
def _cnaster_names() -> frozenset[str]:
    """Every function and class `cnaster` defines, at any depth."""
    names = set()
    for path in _sources(_installed("cnaster")):
        with warnings.catch_warnings():
            # NB `cnaster` carries an invalid escape in a plotting label.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                names.add(node.name)
    return frozenset(names)


@cache
def _defined() -> tuple[Entry, ...]:
    """Every public top-level callable and class under `python/port`."""
    entries = []
    for path in _sources(PORT):
        module = ".".join(path.relative_to(PORT.parent).with_suffix("").parts)
        for node in ast.parse(path.read_text(), str(path)).body:
            if isinstance(
                node, ast.FunctionDef | ast.ClassDef
            ) and not node.name.startswith("_"):
                entries.append(Entry(module, node.name, node))
    return tuple(entries)


def _entries() -> tuple[Entry, ...]:
    """What `port` owns: every public name bar the drop-ins."""
    return tuple(e for e in _defined() if e.name not in _cnaster_names())


def _arguments(node: ast.FunctionDef) -> list[ast.arg]:
    arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    return [a for a in arguments if a.arg not in {"self", "cls"}]


def _fields(node: ast.ClassDef) -> list[str]:
    return [
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    ]


def _words() -> Iterator[tuple[str, str]]:
    """`(key, word)` for every argument and field name `port` owns."""
    for entry in _entries():
        where = f"{entry.module}:{entry.name}"
        if isinstance(entry.node, ast.FunctionDef):
            for argument in _arguments(entry.node):
                yield f"{where} arg {argument.arg}", argument.arg
            continue
        for field in _fields(entry.node):
            yield f"{where} field {field}", field
        for method in entry.node.body:
            if isinstance(method, ast.FunctionDef) and not method.name.startswith("_"):
                for argument in _arguments(method):
                    key = f"{where}.{method.name} arg {argument.arg}"
                    yield key, argument.arg


def _departures() -> dict[str, str]:
    """Every departure the rules below can read, keyed as `KNOWN` is."""
    retired = replaced_by()
    found = {key: retired[word] for key, word in _words() if word in retired}

    for entry in _entries():
        if isinstance(entry.node, ast.ClassDef):
            fields = set(_fields(entry.node))
            if fields & ITERATION_FIELDS and "termination" not in fields:
                found[f"{entry.module}:{entry.name} termination"] = "termination"

    by_key = {f"{e.module}:{e.name}": e.node for e in _defined()}
    for reference, others in SIBLINGS.items():
        first = by_key[reference]
        assert isinstance(first, ast.FunctionDef)
        for other in others:
            node = by_key[other]
            assert isinstance(node, ast.FunctionDef)
            if _shape(node) != _shape(first):
                found[f"{other} sibling"] = reference

    return found


def _shape(node: ast.FunctionDef) -> tuple[tuple[str, ...], str | None]:
    """Arguments in order and the result annotation, what siblings share."""
    names = tuple(a.arg for a in _arguments(node))
    variadic = tuple(
        f"*{a.arg}" for a in (node.args.vararg, node.args.kwarg) if a is not None
    )
    result = ast.unparse(node.returns) if node.returns else None
    return names + variadic, result


@pytest.mark.infra
def test_each_word_names_one_term() -> None:
    """A term's name is not another's retired word, and a word retires once."""
    names = [term.name for term in TERMS]
    retired = [old for term in TERMS for old in term.replaces]

    assert len(set(names)) == len(names), "a term is defined twice"
    assert len(set(retired)) == len(retired), "a word retires into two terms"
    assert not set(names) & set(retired), "a term's name is also retired"


@pytest.mark.infra
@pytest.mark.parametrize(
    ("source", "package"), [("cnaster", "cnaster"), ("sal", "sal")]
)
def test_each_term_is_its_references_word(source: str, package: str) -> None:
    """A term taken from `cnaster` or `sal` is a word that package uses."""
    text = "\n".join(p.read_text() for p in _sources(_installed(package)))
    missing = [
        term.name
        for term in TERMS
        if term.source == source and not re.search(rf"\b{term.name}\b", text)
    ]
    assert not missing, f"{package} does not use {missing}"


@pytest.mark.infra
def test_ports_own_api_uses_the_vocabulary() -> None:
    """No new retired word, no missing `termination`, no sibling that differs."""
    found = _departures()
    new = {key: found[key] for key in found.keys() - KNOWN.keys()}

    assert not new, (
        "departures from CLAUDE.md's API Conventions, each with what replaces "
        f"it: {new}"
    )


@pytest.mark.infra
def test_known_departures_still_occur() -> None:
    """A fixed departure leaves `KNOWN` in the same change."""
    stale = sorted(KNOWN.keys() - _departures().keys())
    assert not stale, f"fixed, so delete from KNOWN: {stale}"


@pytest.mark.infra
def test_tensors_cross_only_behind_a_named_module() -> None:
    """A `torch` or `jax` annotation is in a module whose name says so."""
    offenders = []
    for entry in _entries():
        if re.search(r"torch|jax", entry.module):
            continue
        for node in ast.walk(entry.node):
            annotation = getattr(node, "annotation", None) or getattr(
                node, "returns", None
            )
            if annotation is not None and TENSOR.search(ast.unparse(annotation)):
                offenders.append(f"{entry.module}:{entry.name}")
    assert not offenders, f"tensor in a public signature: {sorted(set(offenders))}"
