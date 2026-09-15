# port

A scientific repository built on
[`snakes_and_ladders`](https://github.com/michaelJwilson/snakes_and_ladders),
holding the same separation of infrastructure from application and the same
standard: correctness and reproducibility of numerical results are required.

Python lives under `python/port/`; the CPU-bound work belongs in the Rust
crate under `src/`, exposed to Python as `port.oxiport`. `CLAUDE.md` records
the rules the repository is developed under, and delegates the shared ones
upstream.

The scope is not yet fixed — the package currently exposes one placeholder
binding, `double`, which establishes the Rust-to-Python path and nothing
else.

## Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | >= 3.12.2 | `requires-python` in `pyproject.toml` |
| Rust | 1.94.1 | pinned by `rust-toolchain.toml`; `rustup` installs it automatically |
| [uv](https://docs.astral.sh/uv/) | >= 0.8.17 | |

`maturin` is the PEP 517 build backend, so a Rust toolchain is required even
for a Python-only workflow: any install compiles the crate.

## Install

```
uv sync --locked --all-extras
source .venv/bin/activate
```

That builds the Rust extension, installs `port` in editable form, and
resolves `snakes_and_ladders` from git against `uv.lock`. The first sync
compiles two Rust crates and downloads PyTorch, so allow several minutes;
later syncs are cached.

To install a single extra rather than all four (`dev`, `test`, `docs`,
`notebooks`):

```
uv sync --locked --extra test
```

Without `uv`, any PEP 517 front end works, but the git dependency is then
unpinned and the extension is rebuilt rather than reused:

```
pip install -e '.[dev,test]'
```

After changing a dependency, run `uv lock` and commit the updated `uv.lock`
in the same change.

## Checks

```
uv run pytest                                          # tests
uv run ruff check . && uv run ruff format --check .     # lint, format
uv run mypy                                            # types, --strict
cargo clippy --all-targets -- -D warnings               # Rust lint
cargo fmt --check                                       # Rust format
```

`mypy` reads its paths from `pyproject.toml` (`python/`, `tests/`). The
compiled extension is typed by the hand-written stub
`python/port/oxiport.pyi`, which must be kept in step with the
`#[pyfunction]` definitions in `src/lib.rs`.

## Layout

| Path | Contents |
| --- | --- |
| `python/port/` | The Python package; `python-source` in `pyproject.toml` |
| `src/` | The Rust crate `oxiport`, bound as `port.oxiport` |
| `tests/` | The suite; `testpaths` in `pyproject.toml` |
| `Cargo.toml` | The single source of the version, which maturin reads across |
