"""`cnamaste/`: `cnaster` at the lock's pin, renamed (#392).

`VENDOR.toml` states the copy: the source commit, the rename, the modules
left out, and the modules a stage has since rewritten ("folded"). This reads
it, maps each copied file back to the pin, and -- run as a module -- writes
the stage-0 copy from the installed `cnaster`, which `uv` installs at exactly
the lock's commit.

    python -m tests.vendor_cnamaste
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "cnamaste"
PACKAGE = VENDOR / "python" / "cnamaste"


@dataclass(frozen=True)
class Manifest:
    """`VENDOR.toml`, read."""

    commit: str
    renames: tuple[tuple[str, str], ...]
    paths: dict[str, str]
    excluded: tuple[str, ...]
    dropped: tuple[str, ...]
    folded: dict[str, str]
    added: dict[str, str]

    def rename(self, text: str) -> str:
        """The pin's text as the copy carries it."""
        for old, new in self.renames:
            text = text.replace(old, new)
        return text

    def restore(self, text: str) -> str:
        """The copy's text as the pin carries it."""
        for old, new in reversed(self.renames):
            text = text.replace(new, old)
        return text

    def copied(self, pinned: str) -> str:
        """A pin path, relative to the package, as the copy names it."""
        return self.paths.get(pinned, pinned)


def manifest() -> Manifest:
    """`cnamaste/VENDOR.toml`."""
    data = tomllib.loads((VENDOR / "VENDOR.toml").read_text())
    return Manifest(
        commit=data["source"]["commit"],
        renames=tuple((r["old"], r["new"]) for r in data["rename"]),
        paths=dict(data["paths"]),
        excluded=tuple(data["excluded"]["trees"]),
        dropped=tuple(d["module"] for d in data["dropped"] if d["module"]),
        folded=dict(data.get("folded", {})),
        added=dict(data.get("added", {})),
    )


def installed() -> Path:
    """The installed `cnaster` package directory (a namespace package)."""
    spec = importlib.util.find_spec("cnaster.io")
    assert spec is not None
    assert spec.origin is not None
    return Path(spec.origin).parent


def installed_commit() -> str:
    """The commit `uv` installed `cnaster` from, as its `direct_url.json` says."""
    site = installed().parent
    (record,) = site.glob("cnaster-*.dist-info/direct_url.json")
    return str(json.loads(record.read_text())["vcs_info"]["commit_id"])


def pinned_files(vendor: Manifest) -> list[str]:
    """The pin's in-scope `.py` files, relative to the package, less the dropped."""
    source = installed()
    files = []
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        if relative.split("/")[0] in vendor.excluded or "__pycache__" in relative:
            continue
        if relative not in vendor.dropped:
            files.append(relative)
    return files


def write() -> None:
    """Write the stage-0 copy: every pinned file, renamed."""
    vendor = manifest()
    assert installed_commit() == vendor.commit, "installed cnaster is not the pin"
    source = installed()
    for relative in pinned_files(vendor):
        text = (source / relative).read_bytes().decode()
        for _, new in vendor.renames:
            assert new not in text, f"{new!r} already at the pin in {relative}"
        target = PACKAGE / vendor.copied(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(vendor.rename(text).encode())


if __name__ == "__main__":
    write()
