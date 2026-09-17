# CLAUDE.md

Guidance for Claude Code when working in this repository.

This file mirrors [`snakes_and_ladders/CLAUDE.md`](https://github.com/michaelJwilson/snakes_and_ladders)
section for section, in its order and under its headings, so the two diff
against each other. Upstream remains authoritative: where a heading below
carries less than upstream's does, upstream's governs the remainder; where
the two disagree, this file wins, and the disagreement is the reason it is
written down.

Everything specific to `port` is gathered under **The application** at the
end. That is where a rule with no home under an upstream heading goes, and a
rule that fits neither is a rule to delete rather than to place.

## Writing Style
1.  **(Reviewer) Time is money and context windows are short:** Be concise
    and direct, use active voice, and limit to the most important facts, in
    priority. Limit tickets and normal PRs to 40 lines, limit PRs with
    extended discussion to 60 lines, e.g. on release.
2.  **Be precise:** Use exact facts and numbers ("40% faster") instead of
    vague intensifiers ("much faster"), except where this lacks meaning,
    e.g. byte reproduction of figures.
3.  **Stay neutral and objective:** Avoid hype, subjective opinions, weak
    qualifiers, and delivering points in both positive and negative. Use
    nouns and verbs; avoid adjectives and adverbs.
4.  **Provide evidence:** Back every claim in PRs and commits with benchmark
    numbers, test validated outputs, or reproductions.
5.  **Maintain formatting:** Apply naming, terminology, notation and syntax
    consistently.
6.  **CLAUDE.md edits are rare:** Do not add technical details to
    `CLAUDE.md` files, but principles. These edits are rare, as (lack of)
    principles become apparent.
7.  **Maintain tone** throughout the repository and associated work.
8.  **Open with a TL;DR:** every ticket, pull request, plan and review opens
    with the result -- the number, the decision, or what broke -- in O(1)
    lines before any context, so the opening does not grow with the body.

These rules are paramount, as upstream states: every document, `CLAUDE.md`,
docstring, comment, commit message, PR, and plan or comment posted to a
thread. They are restated rather than referenced because a reader of `port`
alone must not have to follow a link to find the rules governing every line
they write.

## Project

`port` is a scientific repository built on `snakes_and_ladders`. Correctness
and reproducibility of numerical and scientific results are required,
despite inconvenience.

It holds upstream's separation of concerns -- infrastructure distinct from
application -- and adds a job upstream does not have: `port` exists to
**validate and optimize a dependency it does not own**. Three references
define that job, and **The application** below states how they relate.

The scope is not yet fixed. Until it is, a change that would set it belongs
in a ticket first.

## Repository Map

This file is authoritative. The repository is young, and upstream's map
names documents that do not exist here yet:

| Document | Job |
| --- | --- |
| `README.md` | Project overview, installation, and the index to the rest |
| `CLAUDE.md` | This file |
| `ROADMAP.md` | The stages, and the loop a change passes through |
| `TICKETS.md` | What is filed and not done, grouped by milestone |
| `STATUS.md` | What has landed, with the measurement that established it |

The last three carry one list of milestones between them, and
`tests/test_planning_documents_agree.py` refuses a change that edits it in
one file alone. A planning document that can drift silently is one that
will.

Add a document from the upstream map when the repository has the content
for it, not ahead of it -- `DEV.md` and `INSTALL.md` when `README.md`
can no longer carry both, `CHANGELOG.md` when `towncrier` has a release to
build, `REFERENCES.md` when the citations outgrow **The application**.
`pyproject.toml` already configures `towncrier`, `sphinx`, `mypy --strict`
and the `ruff` rule set, so the tooling precedes the documents rather than
waiting on them.

Upstream carries submodules, each with its own `CLAUDE.md`. `port` has one
package and one crate, so this file is the whole of the guidance. A
submodule `CLAUDE.md` is added when a directory has details this file should
not carry, not before.

## Environment & Tooling
*   **Python (3.12):** Manage via `uv`. Run `uv sync --locked --all-extras`.
    Regenerate locks with `uv lock` and commit `uv.lock` in the same PR.
*   **Rust:** Compiler pinned via `rust-toolchain.toml`. Lockfile is
    `Cargo.lock`. Update with `cargo update` and commit.
*   **Lint/Format (Python):** `ruff check .` and `ruff format --check .`
*   **Type Check (Python):** `mypy --strict`, over the paths in
    `pyproject.toml`'s `files` (`python/`, `tests/`).
*   **Lint/Format (Rust):** `cargo clippy --all-targets -- -D warnings` and
    `cargo fmt --check`.
*   **Audit:** `pip-audit` (Python) and `cargo audit` (Rust).
*   **Docs:** Build with `sphinx-build -W`.
*   **The dependencies are git sources.** `snakes_and_ladders` and `cnaster`
    are pinned by `[tool.uv.sources]` to a branch, so the lockfile is what
    makes a build reproducible. Regenerate and commit it with any dependency
    change, and prefer `uv lock --upgrade-package <name>`: a bare
    `uv lock --upgrade` moves every package and buries the change under the
    diff.

## High Performance frameworks
*   **Measurement.** Upstream's rule, and the one cited most often here.
    Validity is established at every tier -- at gate sizes so a merge has
    something to gate on, and at stress sizes because a claim that holds
    only where the problem is small is not the claim being made. A
    **speedup** is established at stress sizes alone. A ratio read at a gate
    size decides nothing in either direction, and an optimization whose only
    evidence is a gate-sized benchmark has not been measured.
*   **The Oracle.** Every accelerated kernel keeps its pure Python/NumPy
    implementation as an oracle. Regression tests pin the accelerated output
    against it, and recover known values on sims.
*   **Rust backend (`oxiport`).** Upstream's `oxi_snakes_and_ladders` rule,
    read for `oxiport`: a CPU-bound hot path earns the port at $\ge 2\times$
    over the vectorized NumPy reference at realistic sizes, and below that
    the simpler code wins and the port is reverted rather than kept.
*   **GPU.** Upstream's $\ge 10\times$ bar applies unchanged. `port` has no
    GPU path today; the rule is here so that proposing one is a measurement
    rather than a preference.

## High Performance coding

Upstream's section governs in full. Its principles are restated here; its
evidence is not, because a number measured on upstream's fixtures is
upstream's result, and quoting it here would be a claim this repository has
not made.

*   **Profile first.** `cProfile` decides what is worth testing and whether
    alternatives are superior.
*   **Do less work before doing the same work faster.** An algorithmic cut
    outranks a mechanical one, and the profile says which is available.
*   **Warm starts before cold ones.** A search recomputing per step what it
    could update incrementally pays the full cost per step, and asking for
    that comes before reaching for a faster language.
*   **Recompute or store is a decision, and unmade it defaults to
    recompute.** A derived quantity rebuilt inside a loop is a store nobody
    has chosen yet.
*   **Cost depends on the data, not only on its size.** A route chosen on
    problem size alone is chosen on the wrong variable, and a default taken
    from one fixture is a default taken from one dataset.
*   **Caches, memory layout, vectorization, branch misprediction, inlining,
    allocation, the FFI boundary, parallelism and the GIL.** Upstream states
    each, and the statement is not improved by restating it. Its exception
    is noted here because it is the one most easily lost: the
    contiguous-layout rule stops at the Python boundary and inverts for a
    pure-Python inner loop.
*   **Compiled backends.** `njit` for ease, Rust carries the load.

## Testing & Quality Assurance

The rules this repository exists to **apply to a dependency**. Restated in
full for that reason.

*   **Simulate component-wise.** Build fixtures across a set of sizes, from
    a known generative model with a seeded generator. Test components
    individually and in combination.
*   **Fixtures carry their own truth.** Oracles, and recovery of known
    parameters and configurations, are what validate; they are required, not
    optional.
*   **Pin to independent sources.** Validate expected values against
    analytic properties, brute-force computations, or a second
    implementation, with stated tolerances.
*   **Check known mathematical properties:** limits, invariants,
    conservation.
*   **Cross-precision agreement is a tolerance.** Given `float32` and
    `float64`, the tolerance is the higher precision's.
*   **No coverage theatre.** A test asserting only output shapes or the
    absence of an exception is forbidden. Document the gaps it leaves and
    track them with a ticket.
*   **Every test says what it is checked against.** A marker names the
    referee, and `pyproject.toml` registers the names.
*   **Time is money.** Test and build frameworks are justified against a
    time and computational budget.
*   **The per-PR tier is the fast gate; the release gate runs everything.**
    A test over the per-PR duration cap, or whose claim is not needed to
    gate a merge, carries the `release` marker. The CI job must deselect it,
    or the marker is documentation.

A test that cannot say what would have to be wrong for it to fail is not yet
a test.

## Definition of Done
1.  **Regression Test:** Asserts scientific validity against known
    simulations, simpler or alternative algorithms, and pins expected
    output.
2.  **Benchmark:** New or changed hot functions include a `pytest-benchmark`
    (Python) or `criterion` bench (Rust). Baselines reported in the PR, at
    the sizes the Measurement rule requires.
3.  **Coverage:** the `--cov-fail-under` gate is maintained or raised. Never
    lower it to pass a PR. **The application** states what it is measured
    against, which is not what upstream measures.
4.  **Docs & Tooling:** CI covers the new code. `ruff`, `mypy` and `cargo`
    checks pass locally. Documentation Sync is satisfied.
5.  **Dependency Hygiene:** Follows the OSI-licence and external-tools
    rules.

## Dependencies & External Tools
*   Ask for explicit permission before adding new tools or dependencies.
*   Must be open source (OSI-approved licence).
*   Flag any proposed dependency with $<1,000$ GitHub stars (or equivalent
    ecosystem metric).

## Conventions
*   **Documentation Sync:** Any change affecting behaviour, CI, dev setup or
    math models must update, in the same PR, whichever documents it makes
    inaccurate: `README.md`, `CLAUDE.md`, and whichever of the upstream map
    exist by then. If the change is user-visible, add a fragment under
    `changelog.d/`, whose `README.md` states what counts as user-visible here
    -- a narrow set, because most of what this repository produces is not
    visible to an importer of `port`.
*   **Code Standards:** Use type hints where possible. Do not introduce
    silent behaviour changes, e.g. default parameters. Keep dependencies
    minimal and justify additions.
*   **Define abstractions and APIs where they align and simplify multiple
    use cases.**
*   **Dev Standards:** One PR per ticket, to minimize review and tests.
*   **A ticket reports as it goes:** open a branch and draft a PR
    immediately when starting work.
*   **Throughput:** one agent works at a time, in its own worktree, and it
    has the whole host.
*   **Model Routing:** Claude Fable carries the judgement: it plans tickets,
    writes plans and PRs, reviews work by lesser models, and handles review
    tickets. Delegate to Opus otherwise.
*   **Layout:** Python under `python/port/`, the Rust crate under `src/`,
    tests under `tests/`. `python-source` and `module-name` in
    `pyproject.toml` bind the first two; changing either without the other
    breaks the import.
*   **Versioning:** `Cargo.toml` carries the version. `pyproject.toml`
    declares it dynamic and maturin reads it across, so the two cannot
    drift.

## The application

Everything above is upstream's, restated or adopted. Everything below is
`port`'s, and exists because this repository validates a dependency it does
not own.

### The three references

The paper states the method, `cnaster` implements it, and
`snakes_and_ladders` implements the parts of it that are not application
specific.

*   [`cna-maste-paper`](https://github.com/michaelJwilson/cna-maste-paper) is
    the statement of methods and results: the likelihood and its Potts
    spatial prior, the hidden Markov formulation, phasing and genome
    segmentation, the efficient emission evaluation, the label solvers,
    initialization, model selection, integer copy numbers and the expected
    runtime. It is the only reference that says what the code is *supposed*
    to do rather than what some implementation does. It is not a dependency:
    nothing imports it, and the pin that governs it is a citation rather
    than a lockfile.
*   Where `cnaster` and the paper disagree, neither is automatically right.
    The paper may describe an intent the code has not reached, or the code
    may have learned something the paper has not recorded. Say which, with
    the evidence.

**They agree, to a stated tolerance.** Where all three describe one
quantity, all three compute it, and the agreement is reported as a number
rather than as the word "matches". What a tolerance is, and what counts as
having stated one, is settled by **Testing & Quality Assurance** and
**High Performance frameworks** above.

The exception is a difference in the problem itself, and it counts once it
is **stated where the work is** -- a plan, a pull request, a commit message,
a docstring, a ticket -- naming which reference differs, in what regime, and
why it is a choice rather than a defect. It does not need a ticket of its
own; it needs to be findable by whoever reads the comparison next. An
unstated difference is a defect by default, whichever side it favours.
Silence is what the rule exists to prevent, because three references that
were never compared look exactly like three that agree.

### Working against a repository you do not own

*   **Dependency repositories are read only.** `snakes_and_ladders`,
    `cnaster` and anything else `[tool.uv.sources]` names are cloned to read
    and to pin, never to write. Do not branch, push or open a pull request
    against them, and do not ask for the access to. Where one of them needs
    a change, report it here with the evidence and leave landing it to that
    repository. A pin this repository controls is the lever; their history
    is not.
*   **A dependency's `sandbox/` and `deprecated/` are out of scope.** They
    hold work that repository has set aside. `port` validates what a user of
    the dependency gets, so neither is tested, measured nor counted by
    default. Being in the wheel is not what puts code in scope; being
    reachable from an installed entry point is.
    Out of scope is a default, not a prohibition: either is fair game when
    the work requires it. What the rule forbids is reaching in silently. A
    measurement that touches one says which tree, and why the question could
    not be answered from the installed path, so a reader can tell a
    deliberate excursion from a wrong turn.
    A name appearing in one of these trees does not settle where it lives: a
    live copy may be reached from a console script through an alias under
    another name. Scope is decided by tracing the live import to the
    definition, not by matching the identifier.

### Auditing, and proposing a change to someone else's code

*   **An audit names its upstream correspondence.** A ticket auditing a
    dependency states what `snakes_and_ladders` would have to change for the
    problem to be expressible there, or records that it would not and why.
    Measuring one implementation against itself is a profile; the referee is
    what makes it an audit, and the cost of the upstream change is what
    decides whether the comparison is worth having. Where the correspondence
    holds only in part, say which measurements were taken in the regime
    where it holds.
*   **An optimization arrives with its patch, its validation and its
    numbers.** None of the three substitutes for another: a ratio with no
    pinned output has not been shown to compute the same thing, and a patch
    with no ratio has not been shown to be worth its diff.
*   **Reach for what `snakes_and_ladders` already carries.** An optimization
    is considered first as functionality that exists upstream, then as one
    that could, and only then as one written here. Which of the three it is
    belongs in the pull request, because it decides who maintains the
    result.
*   **Simplification needs no speedup; a speedup claim needs 2x.** A patch
    that makes the existing code plainer is worth landing on its evidence of
    equivalence alone. A patch offered as faster is held to the bar
    **High Performance frameworks** sets, measured at a stress size.

### Measuring the subject

*   **Coverage is measured against the whole of `cnaster`, not what is
    imported.** This repository exists to validate a dependency, so the
    denominator is the dependency, not the modules a test happens to load. A
    gate over what is imported reads as progress for importing less, and as
    a regression for bringing a new module under test. The figure is low and
    is meant to be: it says how much of the subject is validated, and it
    rises only by validating more of it. `pyproject.toml` records what the
    denominator currently *is*, measured rather than assumed, and what it
    excludes.
*   **A compiled kernel is tested even where coverage cannot see it.**
    `numba` reports nothing, so `@njit` functions read as uncovered however
    hard they are exercised. They carry tests regardless, against an
    independent reference, and the coverage figure is never the reason a
    kernel goes untested. Where a test exists only to reach a compiled
    kernel, say so in its docstring, because the report will not.
