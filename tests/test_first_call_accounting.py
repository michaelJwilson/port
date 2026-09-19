"""A compiled kernel's first call, kept apart from its cost (#204).

`port`'s field kernel compiles in 2.9 s and runs in 0.006 s. Summed into one
row that reads as a kernel five hundred times slower than it is, and the
first comparison this repository published -- 2.503 s against `cnaster`'s
0.167 s -- was **compilation against a warm cache**, because every earlier
test run had filled `cnaster`'s and none had filled `port`'s.

`CLAUDE.md`: a measurement carries the conditions that decided it, and where
a first call *is* the cost it is reported as its own number rather than
buried inside the stage that paid it.
"""

import time
from typing import Any

import pytest
from port.pipeline import Spent, Swap, instrumented


@pytest.mark.infra
def test_the_first_call_is_recorded_apart_from_the_rest() -> None:
    """One slow call and three fast ones is not four medium ones.

    The arithmetic the report rests on: `first` is the first call alone and
    `warm` is everything after it, so a stage's per-call cost is measured
    over the calls that did not pay for compilation.
    """
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


@pytest.mark.infra
def test_a_single_call_reports_no_warm_average() -> None:
    """A stage called once has a first call and nothing to average.

    Pinned because the alternative -- dividing by zero, or calling the first
    call an average -- is exactly the confusion this accounting exists to
    remove.
    """
    entry = Spent(calls=1, seconds=2.5, first=2.5)

    assert entry.warm == pytest.approx(0.0)
    assert entry.warm_calls == 0


@pytest.mark.infra
def test_the_timer_attributes_the_first_call_to_the_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`instrumented` measures a real function and splits its first call.

    Driven through a stand-in rather than a kernel: what is under test is the
    accounting, and a `numba` function would make the test a compiler
    benchmark whose first call is whatever the on-disk cache happens to hold.
    """
    import sys

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
