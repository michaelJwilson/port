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
*   **An audit names its upstream correspondence:** a ticket auditing a
    dependency states what `snakes_and_ladders` would have to change for the
    problem to be expressible there, or records that it would not and why.
    Measuring one implementation against itself is a profile; the referee is
    what makes it an audit, and the cost of the upstream change is what
    decides whether the comparison is worth having. Where the correspondence
    holds only in part, say which measurements were taken in the regime
    where it holds.
*   **An optimization arrives with its patch, its validation and its
    numbers:** a pull request proposing one carries the change itself, the
    tests pinning it against the behaviour it replaces, and the benchmark it
    is claimed on, at the sizes upstream's Measurement rule requires. None of
    the three substitutes for another: a ratio with no pinned output has not
    been shown to compute the same thing, and a patch with no ratio has not
    been shown to be worth its diff.
*   **Reach for what `snakes_and_ladders` already carries:** an optimization
    is considered first as functionality that exists upstream, then as one
    that could, and only then as one written here. Which of the three it is
    belongs in the pull request, because it decides who maintains the result.
*   **Simplification needs no speedup; a speedup claim needs 2x:** a patch
    that makes the existing code plainer is worth landing on its evidence of
    equivalence alone. A patch offered as faster is held to the bar upstream
    sets, measured at a stress size, and below it the simpler code wins and
    the change is reverted rather than kept.
*   **Coverage is measured against the whole of `cnaster`, not what is
    imported:** this repository exists to validate a dependency, so the
    denominator is every file that dependency ships. A gate over the modules
    a test happens to load reads as progress for importing less, and reads
    as a regression for bringing a new module under test. The figure is low
    and is meant to be: it states how much of the subject is validated, and
    it rises only by validating more of it.
*   **A dependency's `sandbox/` is out of scope:** it holds work that
    repository has set aside -- unpackaged, absent from the wheel, and
    unreachable from an install. `port` validates what a user of the
    dependency gets, so sandbox code is not tested, not measured and not
    counted, unless it is asked for by name. Where a ticket needs something
    that lives there, the dependency is on that code moving into the
    installed tree, and the ticket says so rather than reaching in.
*   **A compiled kernel is tested even where coverage cannot see it:**
    `numba` reports nothing, so `@njit` functions read as uncovered however
    hard they are exercised. They carry tests regardless, against an
    independent reference, and the coverage figure is never the reason a
    kernel goes untested. Where a test exists only to reach a compiled
    kernel, say so in its docstring, because the report will not.

## The paper

[`cna-maste-paper`](https://github.com/michaelJwilson/cna-maste-paper) is the
statement of methods and results that `cnaster` implements: the likelihood
and its Potts spatial prior, the hidden Markov formulation, phasing and
genome segmentation, the efficient emission evaluation, the label solvers,
initialization, model selection, integer copy numbers and the expected
runtime.

It is the third reference this repository works against, and the only one
that says what the code is *supposed* to do rather than what some
implementation does. Where `cnaster` and the paper disagree, neither is
automatically right: the paper may describe an intent the code has not
reached, or the code may have learned something the paper has not recorded.
Say which, with the evidence, rather than assuming the text is the
specification or that the implementation is the truth.

It is read only on the same terms as the dependencies, and it is not a
dependency: nothing here imports it, and the pin that governs it is a
citation rather than a lockfile.

## Scientific validation

Restated from upstream rather than adopted by reference. These are the rules
this repository exists to apply to a dependency, and the one place a reader
of `port` alone must not have to follow a link to find them.

*   **Simulate component-wise.** Build fixtures across a set of sizes, from a
    known generative model with a seeded generator. Test components
    individually and in combination.
*   **Fixtures carry their own truth.** Oracles, and recovery of known
    parameters and configurations, are what validate; they are required, not
    optional.
*   **Pin to independent sources.** Validate expected values against analytic
    properties, brute-force computations, or a second implementation, with
    stated tolerances.
*   **Check known mathematical properties:** limits, invariants, conservation.
*   **Cross-precision agreement is a tolerance.** Given `float32` and
    `float64`, the tolerance is the higher precision's.
*   **No coverage theatre.** A test asserting only output shapes or the
    absence of an exception is forbidden. Document the gaps it leaves and
    track them with a ticket.
*   **Every test says what it is checked against.** A marker names the
    referee, and `pyproject.toml` registers the names.

Correctness and reproducibility of numerical results are required, despite
inconvenience. A test that cannot say what would have to be wrong for it to
fail is not yet a test.
