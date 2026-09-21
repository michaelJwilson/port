"""Three signatures `cnaster`'s `port` branch broke, and the rows that carry them.

**#259 stage 1.** Moving the pin from `finish_annotation_mushift3` to `port`
found three defects that stop the pipeline before it fits anything. Each is a
call site and a signature disagreeing, and none is a calculation `port`
replaces, so the rows say "the run happens" rather than "the number is
better".

`bug`: each test pins a defect in the subject. They pass while the defect is
there and **fail when `cnaster` fixes it**, which is the exit condition -- a
compatibility row that outlives its reason is a row nobody will remove.

The emission comparison that would normally referee a `SWAPS` row is not
available here: the unpatched arm raises before producing one.
"""

from __future__ import annotations

import inspect

import pytest
from port.pipeline import SWAPS


def _row(module: str, name: str) -> object:
    matched = [swap for swap in SWAPS if swap.module == module and swap.name == name]

    assert len(matched) == 1, f"expected one row for {module}.{name}, got {matched}"

    return matched[0]


@pytest.mark.bug
def test_the_phased_override_refuses_the_keyword_its_caller_passes() -> None:
    """`hmm_phased`'s emission never took the rename `hmm_nophasing` did.

    `hmm_nophasing.py:791` passes `num_segments_clones` unconditionally, and
    `class hmm_phased(hmm_nophasing)` means `self.` reaches the override
    whenever the phased model fits. Every phased fit raises.
    """
    from cnaster.hmm_phased import hmm_phased as upstream

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
    from cnaster.hmm_nophasing import hmm_nophasing as upstream

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
    import cnaster.hmm_nophasing as module

    source = inspect.getsource(
        module.hmm_nophasing.compute_emission_probability_nb_betabinom_coded
    )

    assert "if normal_log_lambda is not None:" in source, (
        "the guard moved; re-read it before trusting the patch (#259)"
    )
    assert "copy_states is not None" not in source, (
        "upstream now guards on the variable it indexes; drop the override (#259)"
    )
    assert "compute_logmu_shifts(log_mu, copy_states" in source


@pytest.mark.patch
def test_the_patched_classes_are_upstreams_with_the_keywords_added() -> None:
    """Subclasses, so everything not named here is upstream's own.

    The rows replace a **class** because `port.pipeline.Swap` names a module
    attribute and these are attributes of a class. What that buys is this
    assertion: the patch cannot silently diverge in a method it does not
    mention.
    """
    from cnaster.hmm_nophasing import hmm_nophasing as upstream_nophasing
    from cnaster.hmm_phased import hmm_phased as upstream_phased
    from port.patch.hmm_nophasing import hmm_nophasing as patched_nophasing
    from port.patch.hmm_phased import hmm_phased as patched_phased

    assert issubclass(patched_nophasing, upstream_nophasing)
    assert issubclass(patched_phased, upstream_phased)

    for patched in (patched_nophasing, patched_phased):
        taken = inspect.signature(patched._run_optimization_pipeline).parameters

        assert {"clone_lengths", "propagate_errors"} <= set(taken), (
            f"{patched.__name__} does not take what its caller passes"
        )

    taken = inspect.signature(
        patched_phased.compute_emission_probability_nb_betabinom_coded
    ).parameters

    assert "num_segments_clones" in taken
