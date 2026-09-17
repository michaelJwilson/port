# STATUS

**Nine pull requests have merged and ten are open. Coverage over the whole of
`cnaster` is 13.82 per cent — 766 of 5,544 statements — on 224 tests.** Every
row below carries the measurement that established it, or says the claim is
unmeasured. A row reading "landed" with no number is the failure this document
exists to prevent.

Measured on `main` at `9cd81f0`, `pytest -m "not release"`, one thread.

## Summary

| Roadmap item | Status | Evidence | Key PRs |
| --- | --- | --- | --- |
| §0 Development loop | Landed. Two CI jobs, a pull-request template, five registered markers, a coverage floor over the dependency | `--locked` sync fails a stale lockfile before a merge; `release` deselected in CI after it was found deselecting nothing | [#1](https://github.com/michaelJwilson/port/pull/1), [#21](https://github.com/michaelJwilson/port/pull/21), [#42](https://github.com/michaelJwilson/port/pull/42) |
| 1.1 Fixtures and planted truth | Started. Single-chain and phased fixtures landed; the spatial fixtures are written and did not reach `main`; the two modelling decisions are made | The overdispersion latent decided by measurement: the aggregate recovers a planted 0.1667 as 0.1663 and 0.1672 at clones of 8 and 64 spots under a clone-shared latent, and 0.0255 and 0.0031 under a per-spot one | [#22](https://github.com/michaelJwilson/port/pull/22), [#44](https://github.com/michaelJwilson/port/pull/44) |
| 1.2 The emission and the phased HMM | Landed at fixed parameters. Three implementations not yet pinned against each other | Emission max abs diff 1.5e-13; total log-likelihood 8.0e-13; the phased total exact at every parameter tried; the combined transition 2e-17 against `cnaster`'s own builder | [#22](https://github.com/michaelJwilson/port/pull/22) |
| 1.3 The M step and the EM loop | The M step refereed; monotonicity and the pipeline not started | Two optimizers sharing no solver, parameterization or start agree to 5e-5 relative on `(alpha, beta)` at their maxima. `cnaster`'s shipped criterion stops 7.48 nats short at 19 per cent error in `alpha` and reports `converged: True` | [#31](https://github.com/michaelJwilson/port/pull/31) |
| 1.4 The spatial graph and the label solve | Fixtures written, stranded off `main`; no solver comparison yet | `cnaster` maximises what upstream minimises exactly: cost plus energy is 0 to 7.1e-15 across every labelling tried, nothing tuned | [#44](https://github.com/michaelJwilson/port/pull/44) |
| 1.5 Scale, uncertainty and integer copy | Not started. The decoder is in flight | — | [#56](https://github.com/michaelJwilson/port/pull/56) |
| 1.6 End to end | Not started | — | — |
| 2.1 Coverage over the whole of cnaster | Landed as a gate. The figure is low by construction | The denominator is 5,544 statements, `cnaster`'s 37 top-level modules. By package name the same suite read 32.71 per cent of 743; measured against the dependency it was 4.38 per cent, and is 13.82 per cent now | [#23](https://github.com/michaelJwilson/port/pull/23), [#3](https://github.com/michaelJwilson/port/pull/3), [#31](https://github.com/michaelJwilson/port/pull/31) |
| 2.2 The input path and the entry points | Not started, and invisible to the gate | Dropping `deprecated/` (19 modules) from the wheel moved the denominator by **one** statement, which proves the scan never reached it. `scripts/` is outside for the same reason and is worth +919 statements | — |
| 3.1 The three references | Three audits written, all in flight | — | [#51](https://github.com/michaelJwilson/port/pull/51), [#52](https://github.com/michaelJwilson/port/pull/52), [#53](https://github.com/michaelJwilson/port/pull/53) |
| 3.2 Runtime and memory audits | One landed as a profile; the input path not started | See 4.1 | — |
| 4.1 The HMM/spatial boundary | Profiled, and five patches in flight | The emission producer is 93.0 per cent of the boundary; the field 0.3 per cent, the solve 0.2, the adjacency 0.1 | [#60](https://github.com/michaelJwilson/port/pull/60)–[#64](https://github.com/michaelJwilson/port/pull/64) |
| 4.2 The emission and the M step | Not started | The profile puts the time here, and nothing has been measured against the alternative | — |
| 4.3 Defects found in the subject | Four found, none landed upstream | #30 measured; #45, #46 and the `wolff` import failure reproduced | [#31](https://github.com/michaelJwilson/port/pull/31) |
| 5.1 Upstream reports | Four reports, each with a measurement | Upstream's `external_field` is bitwise unchanged by a covariate, max abs diff 0.000e+00; `baum_welch_family` raises on a two-channel family; `__init__.py` is worth +919 statements to the gate | — |
| 5.2 The documents and the gate | Started. `CLAUDE.md` mirrors upstream; `changelog.d` and the docs builds do not exist | `CLAUDE.md` 213 to 335 lines, ten upstream headings in upstream's order | [#42](https://github.com/michaelJwilson/port/pull/42), [#55](https://github.com/michaelJwilson/port/pull/55) |

## Two facts about the record itself

**#44's work never reached `main`.** It merged into `claude/charming-hypatia-5z7qp3`
forty seconds after that branch merged to `main`, so the spatial labelling
fixtures and the 15.35 per cent floor they carried are in the history and not in
the tree. `main` is at 13.82 per cent and has no `potts_labels`.

**#17, #18 and #19 are open and their pull requests merged.** The closing
keywords did not fire because the pull requests targeted stack branches rather
than the default branch. They are done; the tracker does not know it.

## Milestone 1.1 — Fixtures and planted truth

`tests/fixtures.py` declares truth with upstream simulators and a seed;
`tests/adapters.py` converts an instance to `cnaster`'s arguments. Both are the
reusable artefact and every later rung extends them.

Two decisions are made and recorded rather than left to whatever the simulator
emits. The overdispersion latent is shared across a clone's spots, forced by the
dispersion being one parameter shared across states and clones: at clone sizes 8
and 64 the aggregate's dispersion is invariant under the clone-shared latent and
falls as `1/f` under the per-spot one, 8.2x apart in the negative binomial and
8.5x in the beta-binomial. The segmentation is #67's and is not yet decided.

## Milestone 1.2 — The emission and the phased HMM

Scored at **fixed** parameters, which removes the optimizer, the convergence
criterion and the local optima, so a disagreement is a defect in one
implementation and can be nothing else.

## Milestone 1.3 — The M step and the EM loop

The shortfall in #30 is not a tolerance to widen. `get_em_solver_params` hands
`L-BFGS-B` an `ftol` relative to the objective's magnitude, and the objective
sums over rows, so the criterion loosens as the data grows: at 2,400
observations `|f|` is 7,734 and `1e-6` is an absolute criterion of `8e-3`.

## Milestone 1.4 — The spatial graph and the label solve

The sign and weight correspondence is exact, which is what makes a solver
comparison possible. The comparison itself is #40 and has not been made.

## Milestone 1.5 — Scale, uncertainty and integer copy

Nothing landed. #56 decodes integer copy number into the paper's credible set
and is in flight.

## Milestone 1.6 — End to end

Nothing landed. #4 holds the fixture claim, #69 separates it from the oracle
claim, and #66 and #67 are its two open decisions.

## Milestone 2.1 — Coverage over the whole of cnaster

171 of the 500 missed statements in the loaded modules sit inside `@njit`
functions, which report no line information. No test can move that number, and
the kernels carry tests regardless.

## Milestone 2.2 — The input path and the entry points

The most in-scope code `cnaster` has — five installed console entry points —
and the only in-scope code the gate cannot report on.

## Milestone 3.1 — The three references

Written and unmerged. The audits establish that the paper's emission mean is
state-independent as written, that `cnaster` implements the integer-copy method
the paper contrasts itself with, and that upstream carries the spatial layer in
full.

## Milestone 3.2 — Runtime and memory audits

Only the boundary has been profiled. `load_input_data` and its consumer are #7
and #11 and are unmeasured.

## Milestone 4.1 — The HMM/spatial boundary

Warm timings, second pass, at `K = 7`, `G = 3,000`, `N = 2,500`:

| stage | time | share |
| --- | ---: | ---: |
| pool | 847 ms | 6.5 % |
| emission | 12,180 ms | 93.0 % |
| field | 45.0 ms | 0.3 % |
| adjacency | 7.1 ms | 0.1 % |
| solve | 21.5 ms | 0.2 % |

Peak resident memory 7.52 GB against an emission array of 1.68 GB. The field's
first call is 1,451 ms of `numba` compilation against 20.5 ms of work, which is
why the timings are second-pass.

This reorders the audit's own ranking. Item 2 — scoring only the decoded states
— is the only patch that touches the 93 per cent; the other four are
simplifications and each says so rather than claiming a speedup.

## Milestone 4.2 — The emission and the M step

Where the profile puts the time, and where nothing has been measured.

## Milestone 4.3 — Defects found in the subject

Four, none of them `port`'s to land: a criterion that reports convergence it did
not reach, a solver that is not reproducible and mutates its caller's array, a
fit that writes to the working directory, and a shipped module that raises on
import.

## Milestone 5.1 — Upstream reports

Each carries a number rather than a request. The sharpest is a null: no upstream
change produces the fixture 1.1 needs, because every upstream draw is per vertex
and the decided latent is not.

## Milestone 5.2 — The documents and the gate

`changelog.d/` does not exist, so Documentation Sync has been satisfied by
default on every pull request so far. `CLAUDE.md` names `sphinx-build -W` and
there is no `conf.py` on any branch. Both are the same failure: a declared
requirement with nothing behind it reads as satisfied.
