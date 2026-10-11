"""The functions `run_cnamaste` reaches: a static call graph over the AST, resolved through the live module namespace.

`reached()` starts at `cnamaste.scripts.run_cnamaste.run_cnamaste` and follows,
in each reached function's *live* definition (the one its module's name is
bound to, so a shadowed duplicate is never entered):

- a call or a reference to a name its module binds to a cnamaste function or
  class (a reference counts: `hmmclass=hmm_nophasing` is reached);
- `module.name`, `Class.name`, `self.name` and `cls.name`, through the live
  objects;
- `x.name` on a receiver it cannot resolve, to every method or property `name`
  of a class already reached (an over-approximation, iterated to a fixpoint).

A reached class contributes its `__init__`, `__post_init__` and the dunder
methods Python calls implicitly. Nested functions belong to their parent.
Plots are cut: the stubbed names (`PLOTS`) and the plotting modules. The
graph is static: it reaches branches easy's configuration does not take.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = ("cnamaste.scripts.run_cnamaste", "run_cnamaste")
PLOTS = {"plot_clones_genomic", "plot_clones_spatial", "plot_copy_number_profile", "plot_he", "write_fig"}
"""`capture.py` stubs these: they read and draw, and nothing downstream reads what they return."""
PLOT_MODULES = {"cnamaste.plotting", "cnamaste.plot_genomic", "cnamaste.plot_copy_number_profile", "cnamaste.palette"}
IMPLICIT_CALLS: dict[str, tuple[str, str]] = {
    "logger:RuntimeFormatter.format": ("logger:warning_once", "logging formats every record with the formatter get_logger installs"),
    "logger:RuntimePhaseFilter.filter": ("logger:warning_once", "logging filters every record through the filter get_logger installs"),
    "logger:SharedStateLogger.runtime_phase": ("scripts.run_cnamaste:run_cnamaste", "`logger.runtime_phase = ...`: the class logging.setLoggerClass installs"),
    "cna_hmrf_result:CloneAssignment.num_clones": ("cna_hmrf_result:CnaHMRFResult.validate", "getattr(self.assignment, 'num_clones')"),
    "cna_hmrf_result:CloneAssignment.unique_clone_labels": ("cna_hmrf_result:CnaHMRFResult.validate", "through num_clones"),
}
"""Calls the AST cannot see (a class `logging` instantiates, `getattr` by a string): callee -> (caller, how)."""
IMPLICIT = {"__init__", "__post_init__", "__getitem__", "__setitem__", "__setattr__", "__iter__", "__str__", "__repr__", "__call__"}


@dataclass(frozen=True)
class Fn:
    module: str
    qualname: str
    line: int
    """The live definition's `def` line."""

    @property
    def key(self) -> str:
        return f"{self.module.removeprefix('cnamaste.')}:{self.qualname}"

    @property
    def where(self) -> str:
        return f"{self.module.removeprefix('cnamaste.').replace('.', '/')}.py:{self.line}"


@dataclass
class Graph:
    nodes: dict[Fn, set[Fn]] = field(default_factory=dict)
    """Each reached function -> the functions it reaches directly."""
    parents: dict[Fn, Fn | None] = field(default_factory=dict)
    """Each reached function -> a function reaching it (`None` for the root), a static call preferred to a match by name."""
    by_name: set[Fn] = field(default_factory=set)
    """Functions reached only through a method name on an unresolved receiver: the over-approximation."""
    shadowed: dict[str, list[int]] = field(default_factory=dict)
    """A reached `module:qualname` -> the lines of its dead copies (`dead_copies`)."""

    def find(self, key: str) -> Fn:
        found = [f for f in self.nodes if f.key == key]
        if not found:
            raise KeyError(key)
        return found[0]

    def keys(self) -> set[str]:
        return {f.key for f in self.nodes}


def unwrap(obj: Any) -> Any:
    """The Python function under numba's dispatcher, `functools.wraps`, `partial`, `classmethod`, `staticmethod`."""
    for _ in range(8):
        if isinstance(obj, functools.partial):
            obj = obj.func
        elif isinstance(obj, (classmethod, staticmethod)):
            obj = obj.__func__
        elif inspect.ismethod(obj):
            obj = obj.__func__
        elif hasattr(obj, "py_func"):
            obj = obj.py_func
        elif hasattr(obj, "__wrapped__"):
            obj = obj.__wrapped__
        else:
            break
    return obj


def ours(obj: Any) -> bool:
    module = getattr(obj, "__module__", None) or ""
    return module.startswith("cnamaste") and module not in PLOT_MODULES


@functools.cache
def _tree(module: str) -> ast.Module:
    return ast.parse(Path(sys.modules[module].__file__ or "").read_text())


def definitions(module: str) -> dict[str, list[ast.FunctionDef]]:
    """Every top-level function and method `def` in `module`, by qualname, in source order (duplicates kept)."""
    found: dict[str, list[ast.FunctionDef]] = {}
    for node in _tree(module).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, []).append(node)  # type: ignore[arg-type]
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                accessor = any(isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter") for d in getattr(sub, "decorator_list", []))
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not accessor:  # NB a property's setter is not a copy
                    found.setdefault(f"{node.name}.{sub.name}", []).append(sub)  # type: ignore[arg-type]
    return found


def dead_copies(module: str) -> dict[str, list[int]]:
    """qualname -> the `def` line of every copy in `module`: real definitions (the last binds; the earlier are
    shadowed) and definitions inside a module- or class-level string literal (code commented out as a string)."""
    found: dict[str, list[int]] = {q: [n.lineno for n in nodes] for q, nodes in definitions(module).items()}

    def strings(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                text = node.value.value
                for match in re.finditer(r"^[ \t]*def (\w+)\(", text, re.M):
                    found.setdefault(prefix + match.group(1), []).append(node.lineno + text[: match.start()].count("\n"))
            elif isinstance(node, ast.ClassDef) and not prefix:
                strings(node.body, node.name + ".")

    strings(_tree(module).body, "")
    return {q: sorted(lines) for q, lines in found.items()}


def _first_line(node: ast.FunctionDef) -> int:
    return min([node.lineno] + [d.lineno for d in node.decorator_list])


def live_node(fn: Any) -> tuple[Fn, ast.FunctionDef] | None:
    """`fn`'s live definition: the `def` whose code object the module's name is bound to."""
    fn = unwrap(fn)
    code = getattr(fn, "__code__", None)
    if code is None or not ours(fn):
        return None
    module, qualname = fn.__module__, fn.__qualname__
    if "<locals>" in qualname or code.co_filename != sys.modules[module].__file__:
        return None  # NB a nested function belongs to its parent; a dataclass' generated method has no def
    for node in definitions(module).get(qualname, []):
        if _first_line(node) == code.co_firstlineno or node.lineno == code.co_firstlineno:
            return Fn(module, qualname, node.lineno), node
    raise LookupError(f"{module}:{qualname}: no def at line {code.co_firstlineno}")


def _chain(node: ast.expr) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _class_of(fn: Fn) -> type | None:
    if "." not in fn.qualname:
        return None
    return getattr(sys.modules[fn.module], fn.qualname.split(".")[0], None)


def _members(cls: type, name: str) -> list[Any]:
    """`cls.name` as a function or a property's getter, through the MRO."""
    for klass in cls.__mro__:
        if name in vars(klass):
            raw = vars(klass)[name]
            return [raw.fget] if isinstance(raw, property) else [raw]
    return []


def reached() -> Graph:
    """The call graph from `run_cnamaste`, live definitions only."""
    root_module = importlib.import_module(ROOT[0])
    graph = Graph()
    classes: set[type] = set()
    dynamic: dict[str, set[Fn]] = {}
    todo: list[tuple[Any, Fn | None, bool]] = [(getattr(root_module, ROOT[1]), None, False)]
    by_name: set[Fn] = set()

    def add_class(cls: type, parent: Fn | None) -> None:
        if cls in classes or not ours(cls) or cls.__module__ in PLOT_MODULES:
            return
        classes.add(cls)
        for klass in cls.__mro__:
            if not ours(klass):
                continue
            for name in IMPLICIT & set(vars(klass)):
                todo.append((vars(klass)[name], parent, False))
        for name, callers in dynamic.items():
            for member in _members(cls, name):
                for caller in callers:
                    todo.append((member, caller, True))

    pending = True
    while todo or pending:
        if not todo:
            pending = False
            for key, (caller, _) in IMPLICIT_CALLS.items():
                module, _, qualname = key.partition(":")
                obj: Any = importlib.import_module(f"cnamaste.{module}")
                for part in qualname.split("."):
                    obj = inspect.getattr_static(obj, part) if isinstance(obj, type) else getattr(obj, part)
                todo.append((obj.fget if isinstance(obj, property) else obj, graph.find(caller), False))
            continue
        obj, parent, named = todo.pop()
        obj = unwrap(obj)
        if isinstance(obj, type):
            add_class(obj, parent)
            continue
        found = live_node(obj)
        if found is None:
            continue
        fn, node = found
        if fn.qualname in PLOTS:
            continue
        if parent is not None:
            graph.nodes.setdefault(parent, set()).add(fn)
        if fn in graph.nodes:
            if fn in by_name and not named:
                graph.parents[fn] = parent  # NB a static caller is the better witness than a match by name
                by_name.discard(fn)
            continue
        graph.nodes[fn] = set()
        graph.parents[fn] = parent
        if named:
            by_name.add(fn)
        names = vars(sys.modules[fn.module])
        owner = _class_of(fn)
        dead = [line for line in dead_copies(fn.module).get(fn.qualname, []) if line != fn.line]
        if dead:
            graph.shadowed[fn.key] = dead
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in names:
                target = names[sub.id]
                if sub.id in PLOTS:
                    continue
                if isinstance(target, type) or callable(target):
                    todo.append((target, fn, False))
            elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                chain = _chain(sub)
                if chain and chain[0] in ("self", "cls") and owner is not None and len(chain) == 2:
                    todo.extend((m, fn, False) for m in _members(owner, chain[1]))
                    continue
                if chain and chain[0] in names and len(chain) >= 2:
                    target = names[chain[0]]
                    for part in chain[1:]:
                        target = inspect.getattr_static(target, part, None) if isinstance(target, type) else getattr(target, part, None)
                        if target is None:
                            break
                    if isinstance(target, property):
                        target = target.fget
                    if target is not None and (isinstance(target, type) or callable(target) or isinstance(target, (classmethod, staticmethod))):
                        if chain[-1] not in PLOTS:
                            todo.append((target, fn, False))
                        continue
                    if inspect.ismodule(names[chain[0]]):
                        continue
                dynamic.setdefault(sub.attr, set()).add(fn)
                for cls in list(classes):
                    todo.extend((m, fn, True) for m in _members(cls, sub.attr))
    graph.by_name = by_name
    return graph


def rows(graph: Graph) -> list[tuple[str, str, str]]:
    """(function, module:line, reached from) per reached function, in module then line order."""
    out = []
    for fn in sorted(graph.nodes, key=lambda f: (f.module, f.line)):
        parent = graph.parents[fn]
        out.append((fn.key, fn.where, parent.key if parent else "(entry point)"))
    return out


if __name__ == "__main__":
    g = reached()
    for r in rows(g):
        print(" | ".join(r))
    print(len(g.nodes), "reached;", "shadowed:", g.shadowed)
