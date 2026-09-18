"""The coverage gate measures what it claims to measure.

`pyproject.toml` names an installed directory so coverage counts every file
`cnaster` ships rather than the handful a test imports. A path carries a
Python version, so it can go stale silently and leave the gate reporting a
high fraction of a small denominator. This fails instead.
"""

import pathlib
import tomllib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.infra
def test_coverage_source_is_the_installed_cnaster() -> None:
    """The configured path is the package `import cnaster` resolves to."""
    import cnaster

    configured = [
        PROJECT_ROOT / entry
        for entry in tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())[
            "tool"
        ]["coverage"]["run"]["source"]
        if "cnaster" in entry
    ]
    assert len(configured) == 1, "expected exactly one cnaster source entry"

    # NB cnaster ships no `__init__.py`, so it is a namespace package and
    #    `__file__` is None; `__path__` is what names its directory.
    installed = pathlib.Path(next(iter(cnaster.__path__))).resolve()
    assert configured[0].resolve() == installed, (
        f"coverage measures {configured[0]}, but cnaster is installed at "
        f"{installed}; the gate is reporting a fraction of the wrong denominator"
    )


ORACLE_CONFIG = PROJECT_ROOT / ".coveragerc-oracle"
"""Where the oracle surface is declared.

Separate from `pyproject.toml`'s gate because the two measure different
things: that one asks how much of `cnaster` is validated, this one asks how
much of `snakes_and_ladders` the validation rests on.
"""


def _declared_oracle_modules() -> set[str]:
    """The surface as dotted module names, from the report's `include` globs."""
    import configparser

    parser = configparser.ConfigParser()
    parser.read(ORACLE_CONFIG)

    return {
        glob.removeprefix("*/").removesuffix(".py").replace("/", ".")
        for glob in parser["report"]["include"].split()
    }


@pytest.mark.infra
def test_every_declared_oracle_module_is_the_installed_one() -> None:
    """The surface names real modules, from the package `import` resolves to.

    The same failure the `cnaster` guard above catches, one dependency over: a
    renamed or moved upstream module would leave the oracle floor measuring a
    smaller denominator and reporting a higher fraction of it.

    Resolved with `find_spec` rather than imported. Importing all nine would
    execute their module bodies and raise the oracle figure by several points
    without a single referee having been used -- coverage by import, which is
    the theatre this floor exists to avoid.
    """
    import importlib.util

    import snakes_and_ladders

    installed = pathlib.Path(next(iter(snakes_and_ladders.__path__))).resolve()

    for name in sorted(_declared_oracle_modules()):
        spec = importlib.util.find_spec(name)
        assert spec is not None, f"{name} does not resolve"
        assert spec.origin is not None, f"{name} resolves to no file"
        assert pathlib.Path(spec.origin).resolve().is_relative_to(installed), (
            f"{name} resolves outside the installed package"
        )


@pytest.mark.infra
def test_no_test_referees_against_an_undeclared_upstream_module() -> None:
    """A test cannot use upstream as a referee without it being counted.

    This is the guard that makes the oracle floor mean something. Without it
    the number is gamed by omission: drop a module from the surface and the
    denominator shrinks, the fraction rises, and the claims rest on exactly as
    much of upstream as before.

    The rule is absolute: anything upstream a test reaches for is the
    referee. An exemption would need a reason, and none has been needed --
    every upstream module the suite imports today is in the surface.

    Scanned from the source rather than from `sys.modules`, so an import
    reached only on a branch no test takes is still counted.
    """
    import ast

    declared = _declared_oracle_modules()
    offenders: dict[str, set[str]] = {}

    for path in sorted((PROJECT_ROOT / "tests").glob("*.py")):
        tree = ast.parse(path.read_text())
        used: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "snakes_and_ladders"
            ):
                used.add(node.module or "")
            elif isinstance(node, ast.Import):
                used.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("snakes_and_ladders")
                )

        undeclared = {
            name
            for name in used
            if name != "snakes_and_ladders" and name not in declared
        }
        if undeclared:
            offenders[path.name] = undeclared

    assert not offenders, (
        f"upstream modules imported by tests but absent from "
        f"{ORACLE_CONFIG.name}: {offenders}"
    )
