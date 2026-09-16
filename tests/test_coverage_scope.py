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


@pytest.mark.analytic
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
