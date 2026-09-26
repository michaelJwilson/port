"""Every context manager that changes module state puts it back when its block raises (#408).

A flag, a rebinding or a kept list left behind by a failed block changes
every later call in the process -- in a test run, every later test. The
check is by snapshot, not by reading: every attribute of every loaded `port`
and `cnaster` module, and the contents of every module-level list, dict and
set, before the block and after it raised.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import pytest


def _snapshot() -> dict[tuple[str, str], Any]:
    state: dict[tuple[str, str], Any] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(("port", "cnaster")) or module is None:
            continue
        for attribute, value in list(vars(module).items()):
            if attribute.startswith("__"):
                continue
            contents: Any = None
            if isinstance(value, list | tuple):
                contents = [id(item) for item in value]
            elif isinstance(value, dict):
                contents = {key: id(item) for key, item in value.items()}
            elif isinstance(value, set | frozenset):
                contents = frozenset(id(item) for item in value)
            state[(name, attribute)] = (id(value), contents)
    return state


def _managers() -> list[tuple[str, Callable[[], AbstractContextManager[Any]]]]:
    from port.extensions.copy_likelihood import capture
    from port.extensions.sal import sal
    from port.patch.hmm_initialize.distinct import distinct_init
    from port.patch.hmm_nophasing import logmu_shift
    from port.patch.icm.floor import floor_merge
    from port.patch.integer_copy import by_likelihood
    from port.patch.lattice import rust_lattices
    from port.pipeline import FIGURE_SWAPS, PLOT_OFF_SWAPS, SWAPS, patched

    return [
        ("capture", capture),
        ("sal", sal),
        ("distinct_init", distinct_init),
        ("logmu_shift", logmu_shift),
        ("floor_merge", floor_merge),
        ("by_likelihood", by_likelihood),
        ("rust_lattices", rust_lattices),
        ("patched", lambda: patched(SWAPS + FIGURE_SWAPS + PLOT_OFF_SWAPS)),
    ]


@pytest.mark.infra
@pytest.mark.parametrize("index", range(8))
def test_a_raising_block_leaves_no_module_state_behind(index: int) -> None:
    name, manager = _managers()[index]
    before = _snapshot()

    def _raise_inside() -> None:
        with manager():
            msg = f"inside {name}"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="inside"):
        _raise_inside()

    after = _snapshot()
    changed = sorted(
        f"{module}.{attribute}"
        for key, value in before.items()
        if after.get(key) != value
        for module, attribute in [key]
    )
    assert not changed, f"{name} left behind: {changed[:8]}"
