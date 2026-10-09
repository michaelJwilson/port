"""A compiled kernel's first call is recorded apart from its warm calls (#204)."""

import sys
import time
from typing import Any

import pytest
from port.pipeline import Spent, Swap, instrumented


@pytest.mark.smoke
def test_the_first_call_is_recorded_apart_from_the_rest() -> None:
    """One slow call then three fast: `first` is 1.0, `warm` averages 0.1333."""
    entry = Spent()

    for elapsed in (1.0, 0.1, 0.1, 0.2):
        if entry.calls == 0:
            entry.first = elapsed
        entry.calls += 1
        entry.seconds += elapsed

    assert entry.calls == 4
    assert entry.first == pytest.approx(1.0)
    assert entry.warm == pytest.approx(0.4)
    assert entry.warm_calls == 3
    assert entry.warm / entry.warm_calls == pytest.approx(0.1333, rel=1e-3)


@pytest.mark.smoke
def test_a_single_call_reports_no_warm_average() -> None:
    """A single call has a first call and no warm average."""
    entry = Spent(calls=1, seconds=2.5, first=2.5)

    assert entry.warm == pytest.approx(0.0)
    assert entry.warm_calls == 0


@pytest.mark.smoke
def test_the_timer_attributes_the_first_call_to_the_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`instrumented` charges a real function's slow first call to `first`."""

    class Stub:
        """A module-like object carrying one name to rebind."""

        def __init__(self) -> None:
            self.calls = 0

        def slow_first(self) -> int:
            self.calls += 1
            time.sleep(0.05 if self.calls == 1 else 0.001)
            return self.calls

    stub = Stub()
    module: Any = type("module", (), {"__name__": "stub_module"})()
    module.slow_first = stub.slow_first

    monkeypatch.setitem(sys.modules, "stub_module", module)

    swaps = (Swap("stub_module", "slow_first", "stub_module:slow_first", 204),)

    with instrumented(swaps) as spent:
        for _ in range(3):
            module.slow_first()

    entry = spent["slow_first"]

    assert entry.calls == 3
    assert entry.first > 0.04, "the first call was not the slow one"
    assert entry.warm < entry.first, "compilation was charged to the warm calls"
