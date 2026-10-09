"""No tracked or addable file exceeds `pyproject.toml`'s `[tool.port] max_file_bytes`."""

from __future__ import annotations

import subprocess
import tomllib

import pytest

from tests import ROOT


@pytest.mark.infra
def test_no_file_exceeds_the_repository_limit() -> None:
    """Tracked files, and untracked ones `.gitignore` does not exclude."""
    limit = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["port"][
        "max_file_bytes"
    ]
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    sizes = {
        name: (ROOT / name).stat().st_size
        for name in listed.split("\0")
        if name and (ROOT / name).is_file()
    }
    over = {name: size for name, size in sizes.items() if size > limit}

    assert not over, f"over {limit:_} bytes: {over}"
    assert sizes, "git listed no files"
