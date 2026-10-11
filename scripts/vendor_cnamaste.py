"""`cnamaste`: the pinned `cnaster`'s run path, copied and renamed (T- #836 PR1).

    python -m scripts.vendor_cnamaste [--check]

Copies every module `cnaster.scripts.run_cnaster` reaches through its live
imports, from the installed `cnaster` at the commit `uv.lock` pins, into
`cnamaste/python/cnamaste/`, and renames `cnaster` to `cnamaste` throughout: file
names, imports, identifiers, strings and comments, each in the case the source
spells it (`CNAster` is `CNAmaste`). Nothing else changes. `__init__.py` is
written empty at the package root and in `scripts/`, which `cnaster` ships
without, so the copy installs as a regular package. The project around it,
`cnamaste/pyproject.toml`, `uv.lock` and `LICENSE`, is cnamaste's own.

`DEPARTED` declares each file a later T- #836 PR edits, with that PR. `--check`
writes nothing and exits 1 where a file differs from the copy without a
declaration, where a declared file no longer differs, or where a file is
missing or extra: the committed tree is this script's output plus the declared
edits. `ADDED` declares each file cnamaste owns beyond the copy, under
`cnamaste/tests/`, and `--check` refuses one there it does not declare. The
capture that writes its staged-run file is port's, `scripts/capture_cnamaste.py`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cnamaste" / "python" / "cnamaste"
ENTRY = "cnaster.scripts.run_cnaster"

DEPARTED: dict[str, str] = {
    "scripts/run_cnamaste.py": "PR1, #105: a removed bin's genes leave the gene-level output",
    "spatial.py": "PR1, #692: rectangular clones that admit no assignment are banded",
    "utils.py": "PR1, T- #692: write_fig releases each Text's cached renderer",
}
"""Copied file -> the T- #836 PR that edits it. Only grows."""

PROJECT = ROOT / "cnamaste"
OWNED = ("tests",)
"""cnamaste's own directory beside the copy: every file in it is declared in `ADDED`."""

ADDED: dict[str, str] = {
    "tests/conftest.py": "PR2: the session fixtures over a staged run",
    "tests/audit/capture.py": "PR2: the staged-run file's codec and reader",
    "tests/audit/segments.py": "PR2: port's extensions/segments.py, trimmed",
    "tests/audit/criteria.py": "PR2: each level's units and counts, recomputed from the staged inputs",
    "tests/audit/scoring.py": "PR2: port's qa/scoring.py and the truth reader",
    "tests/data/sim_2d4ce9a9.hdf5": "PR2: CalicoST easy, staged",
    "tests/test_stages.py": "PR2: replay, bookkeeping and science per stage",
    "tests/test_lineage.py": "PR2: the segment and clone hierarchies",
    "tests/test_lookups.py": "PR2: index lookups through merges, reindexing and filtering",
    "tests/test_roundtrip.py": "PR2: the run's inputs against its outputs",
    "tests/test_defects.py": "PR2: one row per port issue, a strict xfail where cnamaste has the defect",
    "tests/test_runtime.py": "PR2: the runtime goals port measured, as placeholders",
    "tests/data/rectangular_hang.npz": "PR2: port's tests/data copy, #304/#692's coordinates",
    "tests/test_config.py": "PR3: the shipped configurations, key by key",
    "zenodo_sim_config.yaml": "PR3: cnaster 4adad4d's zenodo_sim_config.yaml, unmodified",
    "config.yaml": "PR3: cnaster 4adad4d's shipped config.yaml, unmodified",
    "tests/audit/callgraph.py": "PR4: the functions run_cnamaste reaches, static graph over the live namespace",
    "tests/audit/fn.py": "PR4: the per-function rows, their context and shared inputs",
    "tests/test_fn_inventory.py": "PR4: every reached function has rows or a stated reason",
    "tests/test_fn_io.py": "PR4: per-function rows, loading, configuration, logging",
    "tests/test_fn_omics.py": "PR4: per-function rows, the genome segmentation and its counts",
    "tests/test_fn_phasing.py": "PR4: per-function rows, the phase kernel and the phased HMM",
    "tests/test_fn_spatial.py": "PR4: per-function rows, partitions and the spot graph",
    "tests/test_fn_normal.py": "PR4: per-function rows, normal spots and the beta-binomial fit",
    "tests/test_fn_hmm.py": "PR4: per-function rows, emissions, lattices, M step, initialization",
    "tests/test_properties.py": "PR5: model properties: normalisation, limits, invariance, monotonicity, phase flip, invalid values",
    "tests/test_fn_hmrf.py": "PR4: per-function rows, the clone field, ICM, merges, re-indexing",
    "tests/test_fn_integer.py": "PR4: per-function rows, the integer copy-number decoder",
    "tests/test_fn_outputs.py": "PR4: per-function rows, the outputs and run_cnamaste's glue",
}
"""File cnamaste owns beyond the copy and its project files, relative to `cnamaste/` -> the T- #836 PR
that adds it. Only grows. Every file under `OWNED` must be here."""


def pinned() -> tuple[str, Path]:
    """The `cnaster` commit `uv.lock` pins and the installed package, refused unless they agree."""
    lock = (ROOT / "uv.lock").read_text()
    commit = re.search(
        r'name = "cnaster"\nversion = "[^"]+"\nsource = \{ git = "[^"]+#([0-9a-f]{40})"',
        lock,
    )
    spec = find_spec("cnaster")
    if commit is None or spec is None or not spec.submodule_search_locations:
        msg = "uv.lock pins no cnaster commit, or cnaster is not installed"
        raise SystemExit(msg)
    package = Path(next(iter(spec.submodule_search_locations)))
    installed = json.loads(
        next(package.parent.glob("cnaster-*.dist-info/direct_url.json")).read_text()
    )
    if installed["vcs_info"]["commit_id"] != commit[1]:
        msg = f"cnaster {installed['vcs_info']['commit_id'][:7]} is installed, uv.lock pins {commit[1][:7]}: uv sync"
        raise SystemExit(msg)
    return commit[1], package


def reached(package: Path) -> list[Path]:
    """`ENTRY` and every `cnaster` module it imports, transitively, relative to `package`."""
    seen: set[Path] = set()
    todo = [ENTRY]
    while todo:
        relative = Path(*todo.pop().split(".")[1:])
        path = relative.with_suffix(".py")
        if path in seen or not (package / path).exists():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse((package / path).read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "cnaster"
            ):
                todo += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                todo += [
                    a.name for a in node.names if a.name.split(".")[0] == "cnaster"
                ]
    return sorted(seen)


def renamed(text: str) -> str:
    """`cnaster` -> `cnamaste` in every spelling, its case kept: `CNA` as written, then `maste` or `MASTE`."""

    def one(match: re.Match[str]) -> str:
        word, tail = match[0], match[0][3:]
        if not (tail.islower() or tail.isupper()):
            msg = f"no rule renames {word!r}"
            raise ValueError(msg)
        return word[:3] + ("MASTE" if tail.isupper() else "maste")

    return re.sub(r"(?i)cnaster", one, text)


def copy() -> dict[str, str]:
    """The copied tree, relative path -> text."""
    _, package = pinned()
    tree = {
        renamed(str(p)): renamed((package / p).read_text()) for p in reached(package)
    }
    return tree | {"__init__.py": "", "scripts/__init__.py": ""}


def check(tree: dict[str, str]) -> list[str]:
    """What differs between `tree` and `OUT` beyond `DEPARTED`."""
    found = {
        str(p.relative_to(OUT))
        for p in OUT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    problems = [f"extra: {p}" for p in sorted(found - set(tree))]
    problems += [f"missing: {p}" for p in sorted(set(tree) - found)]
    for path in sorted(set(tree) & found):
        differs = (OUT / path).read_text() != tree[path]
        if differs and path not in DEPARTED:
            problems.append(f"undeclared edit: {path}")
        if not differs and path in DEPARTED:
            problems.append(f"declared by {DEPARTED[path]} but unedited: {path}")
    owned = {
        str(p.relative_to(PROJECT))
        for d in OWNED
        for p in (PROJECT / d).rglob("*")
        if p.is_file() and not {"__pycache__", ".pytest_cache"} & set(p.parts)
    }
    problems += [
        f"undeclared addition: cnamaste/{p}" for p in sorted(owned - set(ADDED))
    ]
    problems += [
        f"added but missing: cnamaste/{p}"
        for p in sorted(ADDED)
        if not (PROJECT / p).is_file()
    ]
    return problems + [
        f"declared but not copied: {p}" for p in sorted(set(DEPARTED) - set(tree))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.vendor_cnamaste",
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 where the tree differs",
    )
    arguments = parser.parse_args(argv)

    tree = copy()
    if arguments.check:
        problems = check(tree)
        print(
            "\n".join(problems)
            or f"python/cnamaste is cnaster {pinned()[0][:7]}, renamed"
        )
        return 1 if problems else 0
    for path, text in tree.items():
        if path in DEPARTED:
            continue
        (OUT / path).parent.mkdir(parents=True, exist_ok=True)
        (OUT / path).write_text(text)
    print(
        f"{len(tree)} files from cnaster {pinned()[0][:7]} into {OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
