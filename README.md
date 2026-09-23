# port

[![e2e](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-judged.json)](#what-the-badges-mean)
[![oracle](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-oracle.json)](#what-the-badges-mean)
[![drop-in](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-dropin.json)](#what-the-badges-mean)
[![all](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/coverage-reach.json)](#what-the-badges-mean)
[![speed](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/run-speed.json)](#what-the-badges-mean)
[![mem](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/run-mem.json)](#what-the-badges-mean)
[![instance](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/instance.json)](#what-the-badges-mean)
[![patched](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/michaelJwilson/port/main/.badges/patched.json)](#what-the-badges-mean)

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

## What the badges mean

Eight numbers, and each is a claim rather than a decoration.
`.badges/measurements.json` holds every value with the selection, denominator
and commit that produced it, `python -m tests.badges` derives the badges from
it, and `tests/test_badges_agree.py` fails when the two disagree -- the same
guard `tests/test_planning_documents_agree.py` puts on the planning
documents.

**Four coverage guards, because one figure would answer four questions
badly** (#159, #281). Three measure a dependency this repository does not
own and are low by construction; the fourth measures `port`'s own
replacements and is high for the same reason. One of the four is off: see
the table.

| badge | selection | denominator | what it says |
| --- | --- | --- | --- |
| **e2e** | `end2end` | `cnaster` | how much of the subject is **validated end to end**, against the truth that generated the data. `oracle` is excluded because the badge beside it claims that word |
| **oracle** | the referee's own reach | `snakes_and_ladders` | **disabled** (#282), so it renders `/` rather than a figure nothing measures. It said how much of upstream is used as a referee, separately so it could not rise by importing more of upstream |
| **all** | the other eight markers | `cnaster` | how much is merely **run**, rather than judged against anything outside `cnaster` |
| **drop-in** | `patch or cnaster` | `python/port/patch` | how much of what `port` wrote to replace something is reached by the test comparing it with the something. The one guard whose denominator is ours, so the one with a high floor |

**`speed` and `mem`** are patched `run_cnaster` against `--no-patch`: wall
time and peak resident memory, each arm in its own process, in ratio units.

**`instance`** is what makes those two readable, and `CLAUDE.md` is explicit
that a ratio read at a gate size decides nothing -- so the three are a set.
It carries the size as `obs x spots x states`, which is what a tier name
cannot: two instances both called stress can differ by more than the patch
being measured does. It asserts nothing and is blue for that reason.

**`patched`** is how much of what a run executes `port` has replaced: of
the `cnaster` lines an unpatched `run_cnaster` executes on the dev instance
(`numba` disabled, so a kernel's body counts), the share inside a function a
default row of `run_cnaster_port` replaces -- for a class, its overridden
methods (#302). Measured by `python -m tests.patched_share`, not per pull
request, since it is a whole run; blue, because it asserts nothing.

`tests/test_badges_agree.py` is what keeps them together. It refuses a
recorded ratio that does not name its instance, carry exactly two arms, and
show both arms exiting 0 -- a ratio from an arm that did not complete is not
a ratio -- and it refuses a ratio rendered while `instance` still reads `/`.

**The badges are pinned to `main`, so a pull request does not show its own
figures** -- the eight URLs above all read `/main/.badges/`, and a README
cannot render a branch-relative badge without making `main`'s README wrong.
CI closes that with a report instead (#271): `tests/badge_report.py` renders
this branch's guards against its base, delta first, into the job summary and
into one pull request comment rewritten in place on each push. It reports and
never gates -- `tests/check_badges.py` is what fails the job on a figure that
moved and was never written down, and it runs first so the report cannot
stand in for it. A denominator that moved is called out beside the delta,
because two percentages over different denominators are not comparable and a
bare `-3.43` invites exactly that mistake.

Neither ratio is re-measured by CI, and that is deliberate rather than
pending. A whole `run_cnaster` on a shared two-core runner is moved more by
the runner than by the patch, so a figure from there would be a number
`CLAUDE.md` would not let this repository report. They are measured by hand
on a quiet host, and the recorded commit is what says which tree they
describe.

**`e2e` and `all` are not subtractable** (#223). They report against
different denominators -- 7,030 and 7,329 statements on the same tree with
the same config -- because a `cnaster` subdirectory module enters the figure
only when some test imports it. `cnaster` carries no `__init__.py`, so
coverage's directory scan never reaches `scripts/`, but the tracer measures
whatever runs and the source prefix then admits it. The 299-statement
difference is `scripts/run_cnaster.py`, which guard 3's selection imports
and guard 1's does not.

The direction is what makes it worth fixing rather than noting: bringing that
entry point under an `end2end` test would add its statements to `e2e`'s
denominator, and unless the test covered more than 45.69 per cent of them the
guard would **fall** for validating the most live code the subject has.

**A badge reading `/` has no measurement yet**, and that is the point: not a
zero, which is a claim, and not a last-known figure from a commit nobody can
name. `all` is unwired (#159's "Done when" asks for it and is unmet) and the
ratio pair is #91.

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

To install a single extra rather than all six (`dev`, `test`, `docs`,
`notebooks`, `calicost`, `track`):

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

`track` is the other odd one, and it is the one to read before running
`--all-extras`. It pins [Aim](https://github.com/aimhubio/aim), the run
store behind `snakes_and_ladders.track`, which #251 records
`run_cnaster`'s optimizations through. `track.Run` is a Protocol written
with `aim.Run`'s own signatures, so the recording path imports, types and
tests with `aim` absent; only a reader who wants the UI installs it.

Two things come with it. The resolution goes **194 packages to 219** --
`aim` carries the web server `aim up` runs, which the recording path never
touches. And that server holds two advisories with no fixed version,
**PYSEC-2026-1087** (XSS in the report endpoint) and **PYSEC-2026-1088** (a
sandbox escape in the query handler). CI syncs `--extra dev --extra test`
and so never installs it; `uv sync --locked --all-extras` does, and a
`pip-audit` after that command reports both. Sync without this extra before
auditing.

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
run_cnaster_port config.yaml                 # cnaster's pipeline, port's replacements
run_cnaster_port --no-patch config.yaml      # the same run, nothing rebound
run_cnaster_port --no-figures config.yaml    # the replacements that reproduce bitwise
run_cnaster_port --time-stages config.yaml   # what the replacements cost in the run
run_cnaster_port --no-outputs config.yaml    # skip the fitted/decoded tables below
run_cnaster_port --list                      # what would be rebound, and why
run_cnaster_port --audit-config config.yaml # what the config states that cnaster does not use (#324)
```

**A patched run also writes the seam between the fit and the integers**
(#331): beside `cnaster`'s files, and without touching them,
`port.extensions.outputs` writes `cnv_states.tsv` (each fitted state, the
`(A, B)` each clone decodes it to, and its share of the clone's bins),
`cnv_segments.tsv` (runs of equal `(A, B)`), `cnv_binlevel.tsv` (the
posterior-mean `mu` and `p` per bin), `clone_labels_integer.tsv` (each
spot's clone named by its integer copy profile, so clones that decode alike
at every bin are one clone, #344) and `manifest.json` (states, clones,
likelihoods, the configuration's caps and the flags). Off with `--no-patch`,
so the baseline arm writes what `cnaster` writes.

`port.pipeline.SWAPS` is the table -- one row per `cnaster` name `port`
replaces, each naming the ticket that measured it -- and `patched()` is the
context manager that installs and restores it. The rebinding follows a name
wherever it has been imported, because `run_cnaster` holds its own
`from cnaster.omics import ...`.

**Every row of `SWAPS` reproduces `cnaster` artifact by artifact**
(`tests/test_patched_entry_point.py`), which is the claim that makes the
speed claims worth reading. `FIGURE_SWAPS` is a second table that does not:
lowering the dpi and merging the rasterizing groups writes a different file
by design (#195). It is **in the default** because it is the largest win
here, and `--no-figures` is the arm that reproduces bitwise.

Measured at 4,000 x 1,980 x 5, against `--no-patch`:

| installed | wall | peak RSS |
| --- | ---: | ---: |
| nothing | 192.06 s | 11.35 GB |
| `SWAPS` | 156.63 s | 11.33 GB |
| `SWAPS` + `FIGURE_SWAPS` | 120.48 s | 3.69 GB |

So the figure swaps are most of the runtime win and all of the memory one.
At this instance the emission array is about 0.3 GB against an 11.35 GB
peak, which says plotting caps this run rather than the emission array --
a different regime from #90's declared scale, not a contradiction of it.

`cnaster` appends a fit record to `cnaster.perf` in the repository root on
every run. It is **not tracked** (#222): nothing reads it, no test
references it, and its rows carry no commit or instance, so it is a log
rather than a measurement record. A tracked file that changes on every run
trains a reader to ignore `git status`.

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
| Registered markers | A test not checked against exactly one of `end2end`, `oracle`, `analytic`, `patch`, `backend`, `bug`, `warning`, `snapshot`, `smoke`, `infra` (#157), plus the second axes `critical` and `cnaster`, and the tiers `release`, `preprocessing` and `benchmark`. Only `end2end` and `oracle` count toward coverage, and CI runs neither the `oracle` tests nor their guard while #282 holds |
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
