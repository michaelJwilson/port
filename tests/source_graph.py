"""`port`'s source read as data: modules, import edges, state writes, test markers.

The #517 guards (E2, E3, E5) each ask a question of the source rather than of
a run -- which module reaches which, what is written after import, which
tests count -- and answer it here, once, from the AST. Nothing is imported
but `port.pipeline`, so a guard built on this cannot fail because a module's
import has a side effect.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Iterable
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

from tests import ROOT

if TYPE_CHECKING:
    from port.pipeline import Swap

PACKAGE = ROOT / "python" / "port"
TESTS = ROOT / "tests"

COUNTING = frozenset({"end2end", "oracle"})
"""The two markers `CLAUDE.md` says count."""

MUTATORS = frozenset(
    {
        "__setitem__",
        "add",
        "append",
        "clear",
        "discard",
        "extend",
        "insert",
        "pop",
        "popitem",
        "remove",
        "setdefault",
        "update",
    }
)

__all__ = [
    "COUNTING",
    "counting_mentions",
    "edges",
    "modules",
    "reached",
    "row_modules",
    "state_writes",
    "tables",
]


@cache
def modules() -> dict[str, Path]:
    """Every module under `python/port`, by dotted name; a package by its own."""
    found = {}

    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        name = ".".join(parts).removesuffix(".__init__")
        found[name] = path

    return found


@cache
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _base(module: str, node: ast.ImportFrom) -> str:
    """The absolute module an `ImportFrom` reads from."""
    if not node.level:
        return node.module or ""

    parts = module.split(".")

    if modules()[module].name != "__init__.py":
        parts = parts[:-1]

    parts = parts[: len(parts) - (node.level - 1)]
    return ".".join([*parts, *([node.module] if node.module else [])])


@cache
def _exports(package: str) -> dict[str, str]:
    """What a package's `__init__` re-exports, by name, to the defining module."""
    out: dict[str, str] = {}
    path = modules().get(package)

    if path is None or path.name != "__init__.py":
        return out

    for node in _tree(path).body:
        if isinstance(node, ast.ImportFrom):
            base = _base(package, node)

            for alias in node.names:
                out[alias.asname or alias.name] = _target(base, alias.name)

    return out


def _target(base: str, name: str) -> str:
    """The module `from base import name` lands in, seeing through re-exports."""
    known = modules()

    if f"{base}.{name}" in known:
        return f"{base}.{name}"

    if base in known and known[base].name == "__init__.py":
        return _exports(base).get(name, base)

    return base


@cache
def edges(module: str) -> frozenset[str]:
    """Every `port` module `module` imports, at any depth of its body.

    A `from package import name` is followed through the package's
    `__init__` to the module that defines `name`, so a package that
    re-exports does not make everything it re-exports look used. A
    `"port.x:y"` string is an edge too: it is how a swap row names what it
    installs.
    """
    known = modules()
    out: set[str] = set()

    for node in ast.walk(_tree(known[module])):
        if isinstance(node, ast.ImportFrom):
            base = _base(module, node)
            out |= {_target(base, alias.name) for alias in node.names}
        elif isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names}
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("port.")
            and ":" in node.value
        ):
            base, _, attribute = node.value.partition(":")
            out.add(_target(base, attribute.split(".")[0]))

    return frozenset(name for name in out if name in known and name != module)


def reached(roots: Iterable[str]) -> frozenset[str]:
    """Everything `roots` import, transitively, not counting a package's re-exports."""
    known = modules()
    seen: set[str] = set()
    stack = list(roots)

    while stack:
        module = stack.pop()

        if module in seen:
            continue

        seen.add(module)

        if known[module].name == "__init__.py" and module not in roots:
            continue

        stack.extend(edges(module))

    return frozenset(seen)


def tables() -> dict[str, tuple[Swap, ...]]:
    """Every swap table `port.pipeline` exports."""
    import port.pipeline

    return {
        name: getattr(port.pipeline, name)
        for name in port.pipeline.__all__
        if name == "SWAPS" or name.endswith("_SWAPS")
    }


def row_modules() -> frozenset[str]:
    """The modules that define what some row installs."""
    out = set()

    for swaps in tables().values():
        for swap in swaps:
            base, _, attribute = swap.replacement.partition(":")
            __import__(base)
            target = sys.modules[base]

            for part in attribute.split("."):
                target = getattr(target, part)

            out.add(getattr(target, "__module__", base))

    return frozenset(out)


def state_writes() -> dict[str, frozenset[str]]:
    """Every module-level name `port` writes after import, with where.

    A write is a `global` rebinding, an attribute store or `setattr` on an
    imported `port`/`cnaster` module or class or on one this module
    defines, and an item store or mutating call on a module-level name.
    Keyed by the name written, as `module.name` or `module.Class.name`.
    Excludes `sandbox/`, which installs nothing.
    """
    found: dict[str, set[str]] = {}

    for module, path in modules().items():
        if ".sandbox" in module:
            continue

        tree = _tree(path)
        resolve = partial(_owner, module, _imports(module, tree), _toplevel(tree))

        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                _writes(module, function, resolve, found)

    return {name: frozenset(sites) for name, sites in found.items()}


def _imports(module: str, tree: ast.Module) -> dict[str, str]:
    out = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _base(module, node)

            for alias in node.names:
                out[alias.asname or alias.name] = f"{base}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )

    return {
        name: target
        for name, target in out.items()
        if target.startswith(("port.", "cnaster."))
    }


def _owner(
    module: str, imported: dict[str, str], toplevel: set[str], name: str
) -> str | None:
    """What `name`, read inside `module`, refers to, if it is shared state."""
    if name in imported:
        return imported[name]

    return f"{module}.{name}" if name in toplevel else None


def _toplevel(tree: ast.Module) -> set[str]:
    names = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {t.id for t in targets if isinstance(t, ast.Name)}

    return names


def _writes(
    module: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    resolve: Callable[[str], str | None],
    found: dict[str, set[str]],
) -> None:
    local = {a.arg for a in [*function.args.args, *function.args.kwonlyargs]}

    def note(name: str, node: ast.AST, how: str) -> None:
        found.setdefault(name, set()).add(
            f"{module}:{getattr(node, 'lineno', 0)} {how}"
        )

    for node in ast.walk(function):
        if isinstance(node, ast.Global):
            for name in node.names:
                note(f"{module}.{name}", node, "global")

        targets: list[ast.expr] = []

        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]

        for target in targets:
            if (
                isinstance(target, ast.Attribute | ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id not in local
                and (owner := resolve(target.value.id))
            ):
                if isinstance(target, ast.Attribute):
                    note(f"{owner}.{target.attr}", node, "store")
                else:
                    note(owner, node, "item")

        if not isinstance(node, ast.Call):
            continue

        callee = node.func

        if (
            isinstance(callee, ast.Attribute)
            and callee.attr in MUTATORS
            and isinstance(callee.value, ast.Name)
            and callee.value.id not in local
            and (owner := resolve(callee.value.id))
        ):
            note(owner, node, f".{callee.attr}")

        arguments = list(node.args)

        if isinstance(callee, ast.Name) and callee.id == "setattr":
            arguments = [ast.Name("setattr"), *arguments]

        for index, argument in enumerate(arguments[:-2]):
            if isinstance(argument, ast.Name) and argument.id == "setattr":
                owner_node, attribute = arguments[index + 1], arguments[index + 2]

                if (
                    isinstance(owner_node, ast.Name)
                    and isinstance(attribute, ast.Constant)
                    and isinstance(attribute.value, str)
                    and (owner := resolve(owner_node.id))
                ):
                    note(f"{owner}.{attribute.value}", node, "setattr")

                break


def _markers(decorators: Iterable[ast.expr]) -> set[str]:
    out = set()

    for decorator in decorators:
        for node in ast.walk(decorator):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "mark"
            ):
                out.add(node.attr)

    return out


def _names(node: ast.AST) -> set[str]:
    out = set()

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out |= set(sub.value.replace(":", " ").replace(".", " ").split())

    return out


@cache
def counting_mentions() -> dict[str, frozenset[str]]:
    """Each name, to the counting tests that mention it.

    A counting test is a test function marked `end2end` or `oracle`, by
    decorator or by the module's `pytestmark`. It mentions a name if the name
    appears in its body, in its parameters' fixtures, or in any function or
    fixture of its own module it calls, transitively -- a test that scores a
    stage through a helper counts for that stage.
    """
    out: dict[str, set[str]] = {}

    for path in sorted(TESTS.rglob("test_*.py")):
        tree = _tree(path)
        module_marks: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                module_marks |= _markers([node.value])

        helpers = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for name, function in helpers.items():
            if not name.startswith("test_"):
                continue

            if not (_markers(function.decorator_list) | module_marks) & COUNTING:
                continue

            mentioned: set[str] = set()
            stack: list[ast.AST] = [function]
            visited: set[str] = {name}

            while stack:
                body = stack.pop()
                found = _names(body)
                mentioned |= found

                if isinstance(body, ast.FunctionDef | ast.AsyncFunctionDef):
                    found |= {a.arg for a in body.args.args}

                for callee in found & helpers.keys() - visited:
                    visited.add(callee)
                    stack.append(helpers[callee])

            for word in mentioned:
                out.setdefault(word, set()).add(f"{path.name}::{name}")

    return {name: frozenset(tests) for name, tests in out.items()}
