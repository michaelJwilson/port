"""`--warm-up` compiles the kernels before the clock starts (#211)."""

import sys
from typing import Any

import pytest
from port import pipeline
from port.pipeline import Warmed, _kernels, warm


@pytest.mark.smoke
def test_the_warm_up_compiles_every_kernel_it_names() -> None:
    """No misses, and each named kernel carries a compiled signature afterwards."""
    warmed = warm()

    assert not warmed.missed, warmed.report()
    assert warmed.seconds > 0.0

    for target, _ in _kernels():
        module_name, _, attribute = target.partition(":")
        kernel: Any = getattr(sys.modules[module_name], attribute)

        assert kernel.nopython_signatures, f"{target} was not compiled"


@pytest.mark.infra
def test_a_kernel_that_cannot_be_warmed_is_reported_and_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kernel whose signature moved is reported, and the run still happens."""

    def broken() -> tuple[tuple[str, Any], ...]:
        return (("port.pipeline:warm", (1, 2, 3)),)

    monkeypatch.setattr(pipeline, "_kernels", broken)

    warmed = pipeline.warm()

    assert warmed.compiled == ()
    assert len(warmed.missed) == 1
    assert "not a compiled kernel" in warmed.missed[0]


@pytest.mark.infra
def test_the_report_names_what_was_missed() -> None:
    """`--warm-up` prints this line, so it has to say both numbers."""
    report = Warmed(1.5, ("a", "b"), ("c (TypeError: no)",)).report()

    assert "2 kernels in 1.50s" in report
    assert "missed c (TypeError: no)" in report
