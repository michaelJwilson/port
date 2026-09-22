"""Three signatures `cnaster`'s `port` branch broke, and the rows that carry them.

**#259 stage 1.** Moving the pin from `finish_annotation_mushift3` to `port`
found three defects that stop the pipeline before it fits anything. Each is a
call site and a signature disagreeing, and none is a calculation `port`
replaces, so the rows say "the run happens" rather than "the number is
better".

A fourth is pinned here and is **not** one of them: `cnaster.wolff` does not
import, which #71 recorded against the previous pin. It is kept beside these
because it is the same shape of defect and because its tripwire is the
importer count, not the pin.

`bug`: each test pins a defect in the subject. They pass while the defect is
there and **fail when `cnaster` fixes it**, which is the exit condition -- a
compatibility row that outlives its reason is a row nobody will remove.

Each reads `port.patch.*.UPSTREAM`, the class captured at import time, rather
than the module attribute: the rows are installed for the whole session
(`tests/conftest.py`), so the attribute is the shim and inspecting it would
assert the patch against itself.

The emission comparison that would normally referee a `SWAPS` row is not
available here: the unpatched arm raises before producing one.
"""

from __future__ import annotations

import inspect

import pytest
from port.pipeline import COMPAT_SWAPS


def _row(module: str, name: str) -> object:
    matched = [
        swap for swap in COMPAT_SWAPS if swap.module == module and swap.name == name
    ]

    assert len(matched) == 1, f"expected one row for {module}.{name}, got {matched}"

    return matched[0]


@pytest.mark.patch
def test_the_captured_upstream_is_not_the_installed_shim() -> None:
    """What the three `bug` tests below rest on, asserted rather than assumed.

    If `install` had run before the patch modules imported, `UPSTREAM` would
    be the shim and every defect below would read as fixed. So this pins that
    the capture is `cnaster`'s own class, by the file it is defined in.
    """
    import cnaster.hmm_nophasing
    import cnaster.hmm_phased
    from port.patch.hmm_nophasing import UPSTREAM as upstream_nophasing
    from port.patch.hmm_phased import UPSTREAM as upstream_phased

    for upstream, module in (
        (upstream_nophasing, cnaster.hmm_nophasing),
        (upstream_phased, cnaster.hmm_phased),
    ):
        assert inspect.getfile(upstream) == module.__file__
        assert upstream is not getattr(module, upstream.__name__), (
            "the shim is not installed, so these tests referee nothing"
        )


# NB three tests stood here and are removed rather than repaired (#259).
#    They pinned the two signature breaks and the shift guard that
#    `cnaster@port` has now fixed: the rename is finished in both
#    directions and the guard reads the variable it indexes. Each was
#    written to fail loudly at exactly this moment, and each did. What
#    they refereed no longer exists, so keeping them would mean pinning
#    a defect that is gone.


@pytest.mark.bug
def test_the_sampling_module_does_not_import_and_nothing_notices() -> None:
    """A defect that predates the pin move, and owes no row.

    **#71 recorded it against `finish_annotation_mushift3`**, with the same
    message: `cnaster/wolff.py` imports `get_clone_label_annotation` from
    `cnaster.annotation`, which exports three names and not that one. So this
    is not something the `port` branch broke, and `pyproject.toml`'s `omit`
    list already declines to hide it behind a coverage figure.

    What is new is the second half. **Nothing in the package imports it** --
    `scripts/run_cnaster.py:62` and `hmrf.py:12` comment theirs out and the
    only call site sits inside a string literal -- so no row is owed and #71's
    *"run_cnaster branches into initialize_clones_wolff on a configuration
    key"* does not hold on this branch. That is what the importer count below
    pins, and what makes it fail the day the module becomes reachable.
    """
    with pytest.raises(ImportError, match="get_clone_label_annotation"):
        import cnaster.wolff

    import pathlib

    import cnaster

    package = pathlib.Path(cnaster.__file__).parent
    importers = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        for line in path.read_text().splitlines()
        if line.lstrip().startswith(("from cnaster.wolff", "import cnaster.wolff"))
    ]

    assert not importers, (
        f"{importers} now import it, so the ImportError is reachable and a row "
        "or an upstream fix is owed (#259)"
    )
