"""Every context manager that changes module state restores it when its block raises
(#408).

Checked by snapshot of every loaded `port`/`cnaster` module attribute and container.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import pytest
from port.extensions.copy_likelihood import capture
from port.patch.lattice import rust_lattices
from port.pipeline import FIGURE_SWAPS, PLOT_OFF_SWAPS, SWAPS, patched, release


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
    return [
        ("capture", capture),
        ("rust_lattices", rust_lattices),
        ("patched", lambda: patched(SWAPS + FIGURE_SWAPS + PLOT_OFF_SWAPS)),
    ]


@pytest.mark.infra
@pytest.mark.parametrize("index", range(3))
def test_a_raising_block_leaves_no_module_state_behind(index: int) -> None:
    name, manager = _managers()[index]
    # NB `patched` releases run state on exit (#517); earlier tests' leftovers are
    # dropped first.
    release()
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
