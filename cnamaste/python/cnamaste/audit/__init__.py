"""cnamaste's audit library: the staged-run codec and reader, scoring, digests, segments, criteria, call graph and
fixtures, which the tests, port's capture script and `cnamaste.ledger` share (Ticket#836).

`PROJECT` is the cnamaste checkout the package runs from (an editable install, `cnamaste/python/`); `TESTS` and
`DATA` are its tests and their staged files. Installed anywhere else, importing it raises.
"""

from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
TESTS = PROJECT / "tests"
DATA = TESTS / "data"

if not ((PROJECT / "pyproject.toml").is_file() and DATA.is_dir()):
    raise ImportError(f"cnamaste.audit runs from its checkout, not {PROJECT}: install cnamaste editable")
