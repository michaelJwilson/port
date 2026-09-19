# port

A scientific repository built on
[`snakes_and_ladders`](https://github.com/michaelJwilson/snakes_and_ladders),
holding the same separation of infrastructure from application and the same
standard: correctness and reproducibility of numerical results are required.

`port` adds a job upstream does not have: it **validates and optimizes a
dependency it does not own**. `cnaster` is the subject, the
[`cna-maste-paper`](https://github.com/michaelJwilson/cna-maste-paper) states
the method, and `snakes_and_ladders` implements the parts of that method which
are not application specific. Where all three describe one quantity, all three
compute it, and the agreement is reported as a number rather than as the word
"matches".

Neither dependency can host this comparison; each would have to depend on the
other. Both are **read only** here: `port` pins them, measures them and reports
on them, and lands nothing in either.

Python lives under `python/port/`; the CPU-bound work belongs in the Rust crate
under `src/`, exposed to Python as `port.oxiport`.

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

To install a single extra rather than all five (`dev`, `test`, `docs`,
`notebooks`, `calicost`):

```
uv sync --locked --extra test
```

`calicost` is the odd one. It pins
[CalicoST](https://github.com/raphael-group/CalicoST), the program `cnaster`
was rewritten from, at a commit rather than a branch, because it is a
reference this repository **reads and does not run** -- see
`docs/audit-logmu-shift-calicost.md`,
`docs/audit-integer-copy-calicost.md` and
`docs/audit-cnaster-calicost-divergence.md`. It is an extra rather than a
dependency because nothing on the default path imports it.

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

## Running the pipeline patched

```
port-run-cnaster config.yaml                 # cnaster's pipeline, port's replacements
port-run-cnaster --no-patch config.yaml      # the same run, nothing rebound
port-run-cnaster --time-stages config.yaml   # what the replacements cost in the run
port-run-cnaster --list                      # what would be rebound, and why
```

`port.pipeline.SWAPS` is the table -- one row per `cnaster` name `port`
replaces, each naming the ticket that measured it -- and `patched()` is the
context manager that installs and restores it. The rebinding follows a name
wherever it has been imported, because `run_cnaster` holds its own
`from cnaster.omics import ...`.

A patched run is held to reproducing an unpatched one artifact by artifact
(`tests/test_patched_entry_point.py`), which is the claim that makes the
speed claims worth reading.

## Layout

| Path | Contents |
| --- | --- |
| `python/port/` | The Python package; `python-source` in `pyproject.toml` |
| `src/` | The Rust crate `oxiport`, bound as `port.oxiport` |
| `tests/` | The suite; `testpaths` in `pyproject.toml` |
| `Cargo.toml` | The single source of the version, which maturin reads across |

# Infrastructure

## The contract an agent reads first

| Contract | Governs |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | Every rule the repository is developed under. It mirrors upstream's section for section, so the two diff against each other; where they disagree, this one wins and the disagreement is the reason it is written down. |

One package and one crate, so one file is the whole of the guidance. A
submodule `CLAUDE.md` is added when a directory has details this file should
not carry, not before.

## What enforces the claims

| Mechanism | What it refuses |
| --- | --- |
| Two CI jobs | A stale `uv.lock`, a lint or format failure, an untyped definition, a failing test, a `clippy` warning |
| Registered markers | A test not checked against exactly one of `end2end`, `oracle`, `analytic`, `patch`, `backend`, `bug`, `warning`, `snapshot`, `smoke`, `infra` (#157), plus the tiers `release`, `preprocessing`, `benchmark` and `critical`. Only `end2end` and `oracle` count toward coverage |
| `--cov-fail-under` over the whole of `cnaster` | A figure that rises for importing less. The denominator is the dependency, so the number says how much of the subject is validated |
| [`tests/test_coverage_scope.py`](tests/test_coverage_scope.py) | A gate silently measuring a fraction of the subject after a Python version bump |
| [`tests/test_planning_documents_agree.py`](tests/test_planning_documents_agree.py) | The three planning documents naming different work |
| [`.github/pull_request_template.md`](.github/pull_request_template.md) | A ratio with no pinned output, a patch with no ratio, an unstated difference between the references |

## The documents

| Document | Contents |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | The stages, and the loop a change passes through |
| [TICKETS.md](TICKETS.md) | What is filed and not done, grouped by the milestone it serves |
| [STATUS.md](STATUS.md) | What has landed, with the measurement that established it |
| [CLAUDE.md](CLAUDE.md) | The rules |

`DEV.md`, `INSTALL.md` and `CHANGELOG.md` are added when the content for them
exists, not ahead of it: `README.md` still carries installation and
development, and `towncrier` has no release to build.

# Application

## What exists, measured

Coverage over the whole of `cnaster` is **13.82 per cent** — 766 of 5,544
statements — on 224 tests. Low by construction, and it rises only by validating
more of the subject.

| Claim | Realized |
| --- | --- |
| The emission, scored at fixed parameters against upstream | max abs diff `1.5e-13`; total log-likelihood `8.0e-13`; the phased total exact at every parameter tried |
| The combined transition against `cnaster`'s own builder | `2e-17` |
| Two M steps sharing no solver, parameterization or start | agree to `5e-5` relative on `(alpha, beta)` at their maxima |
| `cnaster`'s shipped EM criterion | stops `7.48` nats short at 19 per cent error in `alpha`, reporting `converged: True` |
| `cnaster` maximises what upstream minimises | cost plus energy is `0` to `7.1e-15` across every labelling tried |

`STATUS.md` carries the rest, and says which rows are unmeasured.
