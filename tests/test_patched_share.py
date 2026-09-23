"""The `patched` badge's line mapping (#302).

`tests.patched_share` counts an executed `cnaster` line as patched when it
sits inside a function a default row replaces. Two rows are the shapes that
decide it: a function, whose whole body counts, and a class, of which only
the methods the replacement overrides count. `infra`: this checks the
repository's own accounting, not the subject.
"""

from __future__ import annotations

import inspect
import os

import pytest


@pytest.mark.infra
def test_a_function_row_counts_its_whole_body() -> None:
    """`pipeline_clone_assignment`, replaced whole, counts every line it has."""
    import cnaster.hmrf

    from tests.patched_share import patched_lines

    spans = patched_lines()
    function = cnaster.hmrf.pipeline_clone_assignment
    source, start = inspect.getsourcelines(function)
    path = os.path.realpath(inspect.getsourcefile(function) or "")

    assert set(range(start, start + len(source))) <= spans[path]


@pytest.mark.infra
def test_a_class_row_counts_only_what_it_overrides() -> None:
    """`hmm_nophasing`: `optimize` is overridden and counts; the rest does not.

    `get_state_posteriors` is inherited unchanged, so a run still executes
    `cnaster`'s own lines there, and counting them would overstate the share.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    from tests.patched_share import patched_lines

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
    from tests.patched_share import share

    assert share({"a.py": {1, 2, 3}}, {"a.py": {2, 3, 9}}) == (2, 3)
    assert share({"a.py": {1}}, {}) == (0, 1)
