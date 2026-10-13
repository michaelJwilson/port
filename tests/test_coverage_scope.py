"""The coverage gate's configured paths are the installed `cnaster` and declared oracle modules."""

import ast
import configparser
import importlib.util
import pathlib
import tomllib

import cnaster
import pytest
import sal

from tests import ROOT


@pytest.mark.infra
def test_coverage_source_is_the_installed_cnaster() -> None:
    """The configured path is the package `import cnaster` resolves to."""

    configured = [
        ROOT / entry
        for entry in tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
            "coverage"
        ]["run"]["source"]
        if "cnaster" in entry
    ]
    assert len(configured) == 1, "expected exactly one cnaster source entry"

    # NB cnaster is a namespace package: `__file__` is None.
    installed = pathlib.Path(next(iter(cnaster.__path__))).resolve()
    assert configured[0].resolve() == installed, (
        f"coverage measures {configured[0]}, but cnaster is installed at "
        f"{installed}; the gate is reporting a fraction of the wrong denominator"
    )


ORACLE_CONFIG = ROOT / ".coveragerc-oracle"
"""Where the oracle surface over `snakes_and_ladders` is declared."""


def _declared_oracle_modules() -> set[str]:
    """The surface as dotted module names, from the report's `include` globs."""

    parser = configparser.ConfigParser()
    parser.read(ORACLE_CONFIG)

    return {
        glob.removeprefix("*/").removesuffix("/*").removesuffix(".py").replace("/", ".")
        for glob in parser["report"]["include"].split()
    }


@pytest.mark.infra
def test_every_declared_oracle_module_is_the_installed_one() -> None:
    """Each declared oracle module resolves via `find_spec` in the installed package."""

    installed = pathlib.Path(next(iter(sal.__path__))).resolve()

    for name in sorted(_declared_oracle_modules()):
        spec = importlib.util.find_spec(name)
        assert spec is not None, f"{name} does not resolve"
        assert spec.origin is not None, f"{name} resolves to no file"
        assert pathlib.Path(spec.origin).resolve().is_relative_to(installed), (
            f"{name} resolves outside the installed package"
        )


@pytest.mark.infra
def test_no_test_referees_against_an_undeclared_upstream_module() -> None:
    """Every upstream module a test imports is in the declared oracle surface."""

    declared = _declared_oracle_modules()
    offenders: dict[str, set[str]] = {}

    for path in sorted((ROOT / "tests").glob("*.py")):
        tree = ast.parse(path.read_text())
        used: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "sal"
            ):
                used.add(node.module or "")
            elif isinstance(node, ast.Import):
                used.update(
                    alias.name for alias in node.names if alias.name.startswith("sal")
                )

        undeclared = {
            name
            for name in used
            if name != "sal"
            and not any(name == d or name.startswith(d + ".") for d in declared)
        }
        if undeclared:
            offenders[path.name] = undeclared

    assert not offenders, (
        f"upstream modules imported by tests but absent from "
        f"{ORACLE_CONFIG.name}: {offenders}"
    )
