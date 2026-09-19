# Audit: 108 open tickets, what can close, and what should be one ticket

**7 verified and closed. Of the 18 filed for a read, 5 have been read and 3
do not close — the original claim of 25 was too high, and the corrections are
below. Thirty-one more are one finding each against a read-only dependency
and should be eight registers rather than thirty-one tickets. Nine carry a
number that has since moved.**

Measured on `main` at `fa022ea`, against the 108 open at that commit.

The tracker has a known failure mode and `STATUS.md` already records it: a
pull request that targets a stack branch rather than the default branch never
fires its closing keyword. Most of what follows is that failure repeated, not
work nobody did.

---

# Part A — what can close

Three tiers, and the tier is the evidence rather than the confidence.

## A1. Verified — the "Done when" was read, clause by clause

| # | Title | Why it is done |
| --- | --- | --- |
| 59 | Audit the HMM/spatial boundary | All three clauses met. The profile is `STATUS.md` §4.1 (emission 12,180 ms, field 45.0 ms, adjacency 7.1 ms, solve 21.5 ms, peak 7.52 GB); all five candidates measured with their own bench file; every reimplementation pinned bitwise. #217 then installed all five and showed a whole patched run reproduces artifact by artifact. **No pull request ever named it**, which is the only reason it is open |
| 167 | `load_input_data`: where its time and memory go | All three clauses met: `tests/test_load_input_data_patch.py` (bitwise) and its bench at both tiers, the range filter's 39x, and the two items the patch cannot reach stated with numbers |
| 17 | A coverage floor over `cnaster` | `STATUS.md`: "open and their pull requests merged… They are done; the tracker does not know it" |
| 18 | Tier one: raise coverage without importing anything new | As above |
| 19 | Tier two: bring further `cnaster` modules under test | As above |
| 39 | Pin that `CountEncoder`'s compression does not move the score | `test_deduplicated_emission_matches_dense` asserts it **bitwise**, plus a round-trip identity and a duplication check |
| 66 | Decided: the fixture draws through upstream's families | The decision is recorded where it is used, `tests/fixtures.py:655`, and `STATUS.md` §1.1 carries the measurement that made it (0.1663 and 0.1672 against 0.0255 and 0.0031) |

## A2. Strong — a dedicated test module and a merged pull request

Each of these has the artefact its ticket asked for. I did not re-read every
clause, so each wants a one-minute read before closing.

| # | Title | Evidence |
| --- | --- | --- |
| 140 | How upstream would referee `icm`, `hmrf`, `hmm`… | `test_hmm_oracle.py`, `test_icm_oracle.py` and both benches; #141 and #144 merged |
| 97 | The per-block oracle rung | `test_oracle_rung.py`, `test_hmm_oracle_bench.py` |
| 128 | The oracle surface is 37.70 per cent | `test_oracle_correspondence.py`; #130 merged |
| 106 | The fixture plants no balanced clone | `test_normal_state.py`; #118 merged |
| 89 | Validate `determine_normal_baseline` | #162, "Judge the normal baseline against the planted exposure" |
| 116 | `cnaster` should emit a simulation manifest | `python/port/patch/simulation_manifest.py` and its test; the ticket body is the upstream report its "Done when" asks for |
| 25 | Referee the integer copy fit against exhaustive enumeration | `test_integer_copy.py`, eight tests including the lattice, the planted copies and the credible set's coverage rate |
| 12 | Audit the spatial graph and smoothing | `test_pseudobulk_and_adjacency.py`, `test_preprocessing_spatial.py` |
| 96 | `initial_phase_given_partition` is the one prep stage stepped over | `test_phasing_bench.py`; #129 merged |
| 16 | The phased transfer matrix: hoisting and factorization | The title **is** the outcome — both measured below the 2x bar, which `CLAUDE.md` calls a dismissal carrying its number |
| 103 | A method for taking plotting out of the coverage denominator | The method is in `pyproject.toml` and four test modules cite it as the maintainer's stated exception |
| 5 | Per-clone normalization of `log_mu` | `test_logmu_shifts.py`: a vectorized reference plus three invariants |

## A3. Superseded — **three of these six were wrong**

Re-read clause by clause at `bc2744e`, which A1 got and this tier did not.
Three do not close, and the pattern is the same each time: a ticket was
matched on its *title* against work that resembled it, rather than on its
"Done when".

| # | Title | Verdict on a read |
| --- | --- | --- |
| 110 | Dead code in the denominator | **Closes, and as done rather than superseded.** `pyproject.toml` omits `sim.py` with its reason and leaves `wolff.py` counted with a comment naming #71 — which is its "Done when", clause for clause |
| 94 | Where the remaining 3,672 uncovered statements are | Not re-read. The figure has moved again (#223) |
| 35 | The coverage gate cannot see the console entry points | Not re-read. #223 measured the 899 statements it is about |
| 132 | Collecting one module costs a later module ten statements | **Does not close.** Its "Done when" is that the mechanism is *named* — a module-level cache, a global, a once-emitting logger, or coverage itself. Nothing named it, and `test_phasing_bench.py` still carries the `release` quarantine |
| 50 | Tickets and PRs exceed Writing Style's line limits | **Does not close.** The PR template adopted option 2, but #50 asks for it in `CLAUDE.md` as a stated difference, and for a pass over the open tickets. `CLAUDE.md`'s Writing Style 1 still reads unamended |
| 72 | `CLAUDE.md` names `sphinx-build -W` and there is nothing to build | **Does not close, and is not a duplicate.** #71 builds `cnaster`'s docs; #72 builds `port`'s, and says so — "beside #71's build". They share no `conf.py` by design |

**#103 leaves A2 for the same reason.** Its "Done when" is that one of three
options is chosen *and the reason is in `pyproject.toml`*. Measured: there is
no plotting exclusion there, and all five modules are still in the
denominator at 0.00 per cent. The method was cited, never adopted. #226 now
carries the decision.

### What this says about the audit

A1 was read clause by clause and held. A2 and A3 were matched on resemblance,
and at least four of the eighteen are wrong — which is the error rate #221
was filed to catch and the reason it says "read before closing" rather than
"close". The lesson is not that the tiers were too generous; it is that
**a title is not a "Done when"**, and only one of these tiers was checked
against one.

---

# Part B — consolidate: 31 tickets, 8 registers

**Thirty-one open tickets are each one finding against a dependency this
repository cannot change.** `CLAUDE.md` is explicit that such a finding "is
reported there with the evidence and leaves landing it to that repository", so
none of them has a fix that could close it here. As thirty-one tickets they
are unrankable and unreportable; as eight registers grouped by where the
report goes, each is one upstream conversation.

| Register | Absorbs | n |
| --- | --- | ---: |
| `cnaster` defects: the input path | 84, 88, 176, 177, 178, 179, 182, 189 | 8 |
| `cnaster` defects: the HMM and the M step | 30, 135, 136, 143, 146 | 5 |
| `cnaster` defects: labelling and the spatial graph | 45, 58, 81, 180 | 4 |
| `cnaster` defects: normal and annotation | 165, 166, 183 | 3 |
| `cnaster` defects: integer copy | 134, 181 | 2 |
| `cnaster` defects: plotting and H&E | 113, 115 | 2 |
| `cnaster` defects: `run_sim_analysis` | 117, 119 | 2 |
| `cnaster` defects: the rest | 20, 46, 78, 82, 102, 105 | 6 |

Each register keeps every finding's evidence as a table row — the symptom, the
test that pins it, and what it would take upstream — so nothing is lost. The
pins stay where they are; they are written to fail when `cnaster` fixes the
defect, and that is what actually tracks these.

**Two smaller consolidations:**

- **Upstream requirements: #32, #99, #109 and #57** all ask "what does
  `snakes_and_ladders` need before it can referee more of `cnaster`". One
  standing ticket, with #65 and #70 as its open items.
- **The coverage guards: #139, #153, #159 and #163** are four framings of one
  question. #159 is the live one; the others are its history.

---

# Part C — rewrite: nine tickets whose premise has moved

| # | What is stale | What it should say |
| --- | --- | --- |
| 160 | Asks for all 21 stages; five landed (#164, #169, #170, #171, #173) | Narrow to the stages still unjudged, and recut the figure |
| 111 | Lists 11 cold modules; `test_cold_inference.py` and `test_cold_plots.py` reached several | Narrow to what is still cold |
| 90 | "the patches that would remove it" — #206 **installed** the fused field | It is now one experiment: run the key instance at `S = 5,000` under `run_cnaster_port`. That is the whole remaining ticket |
| 174 | Claims 14.3x; #203 measures the same patch **9.7x slower** at the dev instance | Restate the ratio with the regime each number was taken in. The two are not in conflict — they are different sizes — and the ticket should say so rather than leave a reader to find #203 |
| 94 | "3,672 uncovered statements" | The number moved; close (A3) or recut it |
| 128 | "37.70 per cent" | Stale figure in the title |
| 139 | "7.92 per cent" | Stale figure in the title |
| 70 | "what #660 has already landed" | A bare cross-repo pull request number; qualify it as `snakes_and_ladders#660` |
| 195 | **Closed in error.** `Closes #195` fired from #215's body despite the qualifier | Its "rest of the outside" item — profiling `phasing.py`, `integer_copy.py` and `hmrf.py`'s wrappers at a stress size — needs a ticket of its own |

---

# What this would leave

| | now | after |
| --- | ---: | ---: |
| open tickets | 108 | **60** |
| one-finding defect tickets | 31 | 8 |
| tickets carrying a stale number | 9 | 0 |

The 25 closable are 23 per cent of the tracker, and all but six are work that
was done and never recorded. That is the number worth acting on: the register
is not behind, it is mis-filed.

---

# Acted on

**A1 is closed.** #17, #18, #19, #39, #59, #66 and #167, each with a comment
naming what satisfied it -- a close with no evidence is the drift this audit
exists to undo.

The rest is filed rather than done, one ticket per section:

| ticket | what it carries |
| --- | --- |
| #218 | #195's undone half: profile `phasing`, `integer_copy` and `hmrf`'s wrappers at a stress size |
| #219 | Part B, the 31 into 8 registers -- blocked on one decision, eight tickets or one document |
| #220 | Part C, the nine stale premises, including the #174/#203 contradiction and #90's single experiment |
| #221 | A2 and A3, the eighteen that want a read before closing |

Open at `fa022ea`: 108. Open after A1 and these four: **105**.
