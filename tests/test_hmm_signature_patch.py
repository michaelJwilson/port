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


@pytest.mark.bug
def test_the_phased_override_refuses_the_keyword_its_caller_passes() -> None:
    """`hmm_phased`'s emission never took the rename `hmm_nophasing` did.

    `hmm_nophasing.py:791` passes `num_segments_clones` unconditionally, and
    `class hmm_phased(hmm_nophasing)` means `self.` reaches the override
    whenever the phased model fits. Every phased fit raises.
    """
    from port.patch.hmm_phased import UPSTREAM as upstream

    taken = inspect.signature(
        upstream.compute_emission_probability_nb_betabinom_coded
    ).parameters

    assert "clone_lengths" in taken, "upstream stopped taking the old name"
    assert "num_segments_clones" not in taken, (
        "upstream now accepts the new name; drop the hmm_phased row (#259)"
    )

    _row("cnaster.hmm_phased", "hmm_phased")


@pytest.mark.bug
def test_the_fit_refuses_two_keywords_its_caller_passes() -> None:
    """`hmm.py:122-124` calls with `propagate_errors` and the old `clone_lengths`.

    `_run_optimization_pipeline` takes neither -- `num_segments_clones` is
    the rename, and its `# **kwargs,` is commented out -- so every fit
    raises before the phased one gets the chance to.
    """
    from port.patch.hmm_nophasing import UPSTREAM as upstream

    taken = inspect.signature(upstream._run_optimization_pipeline).parameters

    assert "propagate_errors" not in taken, (
        "upstream now accepts `propagate_errors`; drop half the mixin (#259)"
    )
    assert "clone_lengths" not in taken, (
        "upstream now accepts the old name; drop the other half (#259)"
    )
    assert "num_segments_clones" in taken

    _row("cnaster.hmm_nophasing", "hmm_nophasing")


@pytest.mark.bug
def test_the_shift_is_guarded_on_the_wrong_variable() -> None:
    """The guard tests `normal_log_lambda`; the call indexes `copy_states`.

    `copy_states` defaults to `None` and the call site never passes it, and
    `compute_logmu_shifts` is `@njit`, so the first call is a compile failure
    rather than a `TypeError`. Asserted on the source because the failure is
    numba's and reproducing it costs a compile.
    """
    from port.patch.hmm_nophasing import UPSTREAM as upstream

    source = inspect.getsource(upstream.compute_emission_probability_nb_betabinom_coded)

    assert "if normal_log_lambda is not None:" in source, (
        "the guard moved; re-read it before trusting the patch (#259)"
    )
    assert "copy_states is not None" not in source, (
        "upstream now guards on the variable it indexes; drop the override (#259)"
    )
    assert "compute_logmu_shifts(log_mu, copy_states" in source


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
