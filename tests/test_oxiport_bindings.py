"""Integration test for the compiled Rust extension.

Requires the package to be built and installed (`uv sync`), unlike a pure
Python test: it imports `port.oxiport`, which only exists once maturin has
compiled the crate.
"""

import pytest
from port import double
from port.oxiport import double as double_from_extension


@pytest.mark.oracle
@pytest.mark.critical
def test_double() -> None:
    """The extension doubles, against the arithmetic it claims to do."""
    assert double_from_extension(21) == 42
    assert double_from_extension(0) == 0
    assert double_from_extension(-3) == -6


@pytest.mark.analytic
def test_top_level_reexport_is_the_extension_function() -> None:
    """`port.__init__` re-exports the binding rather than shadowing it."""
    assert double is double_from_extension
