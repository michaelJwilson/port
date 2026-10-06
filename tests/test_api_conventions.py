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
    "port.extensions.parameter_errors:shift_weights arg log_mus": "F5",
    "port.patch.hmm_nophasing.logmu_shift:shifts arg log_mus": "F5",
    "port.extensions.outputs:states arg fit": "F6",
    "port.extensions.outputs:binlevel arg fit": "F6",
    "port.extensions.outputs:segments arg fit": "F6",
    "port.patch.hmrf.core_inference:pin_neutral arg result": "F6",
    # --- T- #673 G6: the simulation machinery, moved from `tests/` ----------
    # NB `SKIPPED` hid these under `tests/`. `CoreInferenceTruth.seed` is a
    #    field `tests.metrics.fixture_hash` hashes by name, so renaming it
    #    moves every recorded dev fixture (`07b82e92`); the builders' `seed`
    #    and `max_iter` follow it and `cnaster`'s configuration key.
    "port.sim.truth:CoreInferenceTruth field seed": "G6",
    "port.sim.truth:core_inference_truth arg seed": "G6",
    "port.sim.he_slide:mock_he arg seed": "G6",
    "port.sim.fixtures:purify arg seed": "G6",
    "port.sim.run_config:run_cnaster_config arg max_iter": "G6",
    # NB G3: the audits and realizations, moved the same way; `max_iter` is
    #    `run_cnaster_config`'s, which `audit_truth` passes through.
    "port.qa.audit:audit_truth arg max_iter": "G3",
    "port.qa.audit:audit_errors arg seed": "G3",
    "port.sim.realizations:realize arg seed": "G3",
    "port.sim.realizations:chosen arg seed": "G3",
}
"""Departures found by #401's audit, keyed `module:name kind word`."""

SIBLINGS: dict[str, tuple[str, ...]] = {
    "port.patch.icm.interface:icm_sweep": (
        "port.patch.icm.alpha_expansion:alpha_expansion_sweep",
        "port.extensions.label_solver:sal_icm_sweep",
        "port.extensions.label_solver:expansion_then_floor",
        "port.extensions.label_solver:expansion_then_merge",
        "port.extensions.label_solver:sal_icm_floor_sweep",
        "port.extensions.label_solver:fusion_then_merge",
        "port.extensions.label_solver:sal_icm_argmax_sweep",
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
