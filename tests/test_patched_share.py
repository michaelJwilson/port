"""The `patched` badge's line mapping in `port.qa.benchmark` (#302).

A function row counts its whole body; a class row only overridden methods. `infra`.
"""

from __future__ import annotations

import inspect
import os

import pytest


@pytest.mark.infra
def test_a_function_row_counts_its_whole_body() -> None:
    """`pipeline_clone_assignment`, replaced whole, counts every line it has."""
    import cnaster.hmrf
    from port.qa.benchmark import patched_lines

    spans = patched_lines()
    function = cnaster.hmrf.pipeline_clone_assignment
    source, start = inspect.getsourcelines(function)
    path = os.path.realpath(inspect.getsourcefile(function) or "")

    assert set(range(start, start + len(source))) <= spans[path]


@pytest.mark.infra
def test_a_class_row_counts_only_what_it_overrides() -> None:
    """`hmm_nophasing`: only the overridden `optimize` counts, not inherited
    `get_state_posteriors`.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.qa.benchmark import patched_lines

    spans = patched_lines()
    path = os.path.realpath(inspect.getsourcefile(hmm_nophasing) or "")

    def lines(name: str) -> set[int]:
        source, start = inspect.getsourcelines(getattr(hmm_nophasing, name))

        return set(range(start, start + len(source)))

    assert lines("optimize") <= spans[path]
    assert not lines("get_state_posteriors") & spans[path]


@pytest.mark.infra
def test_the_share_is_executed_lines_inside_patched_spans() -> None:
    """Three executed lines in one file, two of them patched: 2 of 3."""
    from port.qa.benchmark import share

    assert share({"a.py": {1, 2, 3}}, {"a.py": {2, 3, 9}}) == (2, 3)
    assert share({"a.py": {1}}, {}) == (0, 1)
