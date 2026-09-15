# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Authority

[`snakes_and_ladders`](https://github.com/michaelJwilson/snakes_and_ladders)
is this repository's upstream: `port` depends on it, and its `CLAUDE.md` is
authoritative here. Read it first and follow it in full. Its **Writing
Style** rules govern every document, docstring, comment, commit message, PR
and posted plan in this repository, as they do there.

Adopted without restatement, and not repeated below:

| Section | Applies here |
| --- | --- |
| Writing Style | In full. Paramount, as upstream states |
| Environment & Tooling | In full; the toolchain is the same, `uv` and a pinned `rust-toolchain.toml` |
| High Performance frameworks | In full, reading `oxiport` for `oxi_snakes_and_ladders` |
| High Performance coding | In full |
| Testing & Quality Assurance | In full |
| Definition of Done | In full |
| Dependencies & External Tools | In full. Permission first, OSI licence, flag under 1,000 stars |
| Conventions | In full, less what the Repository Map below narrows |

Where this file and upstream disagree, this file wins, and the disagreement
is the reason to write it down. It is otherwise short by intent: a rule
worth stating for both repositories belongs upstream, not copied here, where
the copy would drift.

## Project

`port` is a scientific repository built on `snakes_and_ladders`. It holds
the same separation of concerns — infrastructure distinct from application —
and the same standard: correctness and reproducibility of numerical results
are required, despite inconvenience.

The scope is not yet fixed. Until it is, a change that would set it belongs
in a ticket first.

## Repository Map

This file is authoritative. The repository is young, and the upstream map
names documents that do not exist here yet:

| Document | Job |
| --- | --- |
| `README.md` | Project overview and installation |
| `CLAUDE.md` | This file |

Add a document from the upstream map when the repository has the content
for it, not ahead of it — `ROADMAP.md` and `TICKETS.md` when work is
planned past the current change, `DEV.md` and `INSTALL.md` when `README.md`
can no longer carry both, `CHANGELOG.md` when `towncrier` has a release to
build. `pyproject.toml` already configures `towncrier`, `sphinx`, `mypy
--strict` and the `ruff` rule set, so the tooling precedes the documents
rather than waiting on them.

## Conventions

*   **Layout:** Python under `python/port/`, the Rust crate under `src/`,
    tests under `tests/`. `python-source` and `module-name` in
    `pyproject.toml` bind the first two; changing either without the other
    breaks the import.
*   **Versioning:** `Cargo.toml` carries the version. `pyproject.toml`
    declares it dynamic and maturin reads it across, so the two cannot
    drift.
*   **The upstream dependency:** `snakes_and_ladders` is pinned to its
    `main` branch by `[tool.uv.sources]`. It is a git source, so a lockfile
    is what makes a build reproducible; regenerate and commit it with any
    dependency change.
*   **Dependency repositories are read only:** `snakes_and_ladders`,
    `cnaster` and anything else `[tool.uv.sources]` names are cloned to read
    and to pin, never to write. Do not branch, push or open a pull request
    against them, and do not ask for the access to. Where one of them needs
    a change, report it here with the evidence and leave landing it to that
    repository. A pin this repository controls is the lever; their history
    is not.
