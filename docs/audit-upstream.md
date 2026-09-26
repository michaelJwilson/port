# Audit: the paper and `cnaster` against `snakes_and_ladders`

Issues #29 and #32, together. They are two directions of one correspondence:
**#29** asks which of the paper's methods already have an upstream backend,
**#32** asks what upstream would need before it can referee the rest.

Paper at `0478c74`; upstream at the `main` pin; `cnaster` at `4adad4d`.

**Everything upstream is missing is one capability: it cannot take
per-observation exogenous data.** All four confirmed gaps are covariates --
exposure on the count mean, trials on the allele channel, and the switch
probability on the transition. Nothing else in #32's list survived: one item
was wrong, two were answered, one is cheaper as a test here, and one is a
policy decision rather than code.

**Upstream carries more of the paper than the paper's own implementation
does.** Every solver `wolff.tex` argues about is implemented there, behind
one enum, with a Rust backend -- and `search/ground_state.py` is a
budget-charged harness that already answers the question the paper asserts.

One finding is against `port` rather than either dependency: PR #44
reimplemented an enumeration limit upstream already ships, at the same value.

---

# Part A -- the paper's methods, classified (#29)

`CLAUDE.md`: an optimization is considered first as functionality that exists
upstream, then as one that could, and only then as one written here. The same
question, asked of a method.

| paper | section | upstream | class |
| --- | --- | --- | --- |
| Potts spatial prior | `likelihood.tex` | `sim/graph.PottsGraph`, `likelihood/potts.py` | **exists** |
| MRF ground state | `likelihood.tex` | `search/alpha_expansion.energy` | **exists** |
| ICM | `wolff.tex` | `iterated_conditional_modes`, `run_icm` | **exists** |
| Wolff + annealing | `wolff.tex` | `PottsMove.WOLFF`, `anneal_potts`, `run_wolff` | **exists** |
| Swendsen-Wang (named, dismissed) | `wolff.tex` | `run_swendsen_wang` | **exists** |
| alpha-expansion, min-cut | `wolff.tex` | `alpha_expansion` + Rust `max_flow`, `ising_ground_state` | **exists** |
| belief propagation, decimation | `wolff.tex` | `likelihood/belief_propagation.py`, `message_passing.max_product` | **exists** |
| HMM: `T_kl`, `E_k(b)`, Viterbi, forward/backward | `hidden_markov.tex` | `likelihood/forward_backward.py`, `opt/hmm.py` | **exists** |
| circulant genomic prior | `likelihood.tex` | the transition #15 pinned | **exists** |
| NB and BB emissions | `emission.tex` | `emissions.py`, `reestimate` | **exists** |
| Laplace errors, inverse Hessian | `integer_copy_numbers.tex` | `opt/fit.observed_information`, `constrained_standard_errors` | **exists** |
| exhaustive oracle | -- | `enumeration.refuse_oversized` | **exists** |
| sufficient statistics `B^k_i` | `efficient_likelihood.tex` | -- | **could** |
| variational inference | `vi.tex`, `minka.tex` | -- | **could** |
| Pareto-front model selection | `prior_model_selection.tex` | -- | **could** |
| integer copy number | `integer_copy_numbers.tex` | -- | **`cnaster`-only** |
| phasing, genetic distance, segmentation | `phasing.tex` | -- | **`cnaster`-only** |
| H&E segmentation | `he_clone_initialization.tex` | -- | **`cnaster`-only** |
| normal-spot identification | -- | -- | **`cnaster`-only** |

## A1. The solver argument can be tested today, and upstream already built the harness

`wolff.tex` makes three falsifiable claims: ICM freezes fluctuations into
spurious disjoint clusters; Wolff with annealing escapes them; alpha-expansion
complements both. #28 established that `cnaster` ships Wolff unwired and has
no alpha-expansion at all, so the paper's own implementation cannot test its
own argument.

Upstream can. `search/ground_state.py` runs **nine methods** -- `run_greedy`,
`run_icm`, `run_gibbs_zero`, `run_anneal`, `run_swendsen_wang`, `run_wolff`,
`run_tempering`, `run_alpha_expansion`, and `max_product` -- on three rungs,
and three of its design decisions matter here.

**It charges a budget in site visits, not sweeps.** Its docstring:

> A Wolff step flips one cluster while a heat-bath sweep touches every site,
> so equal sweeps hand the cluster moves a free lattice per move. One visit
> is one read or write of a site's label by a move.

A Wolff-versus-ICM comparison at equal sweeps is not a comparison, and #8 and
#40 would have had to discover that.

**It referees twice, and the second referee is the one this paper needs.**
`Structure` scores a labelling against the *generating parameters* -- the size
tilt, the null class' occupancy, whether the mean field rises monotonically
across size quartiles. Upstream's reason:

> a low energy that fails the structure is a low-energy state of a
> *different* model.

That is the shape of `wolff.tex`'s claim. "Spurious disjoint clusters that
local updates fail to resolve" is a statement about the *structure* of the
labelling, not its energy, and an energy-only comparison cannot detect it.
`structure()` is the closest thing to a ready-made test of the paper's
metastasis argument that exists anywhere.

**It names a stress instance**: open triangular 71x71, `q = 10`, `J = 0.7`,
as `spatio_only/release`. #32 item 9 asked for exactly this and can close:
reuse it rather than inventing a size.

**Action:** #8, #29 and #40 should run against `ground_state.py` rather than
build a harness. The measurement `wolff.tex` asserts is one call away.

## A2. What upstream does not have, and should not

**Integer copy number.** No lattice of allele copies, no ploidy, nothing
discretising a continuous fit onto a bounded integer grid. Confirmed by
search. Not a gap: #25 shows enumeration is an *exact* oracle over sixteen
ACN states, which is better than a second implementation -- adding the concept
upstream to referee it would be building a second implementation in order to
check the first.

**Phasing, genetic distance, genome segmentation.** No contig, no switch
error, no variable-length segment. Application specific.

**H&E, normal-spot identification, the assay's preprocessing.** Likewise.

**Variational inference.** No ELBO, no variational posterior anywhere --
searched. `vi.tex` and `minka.tex` have no upstream counterpart. Both are
unreachable from the compiled paper (#27 §5), so this blocks nothing today.

**Pareto-front model selection.** No `pareto`, no curvature search. Neither
reference implements `prior_model_selection.tex` (#28 §13), so a `port` test
of it has nothing on either side to compare.

## A3. Neither implementation has the paper's efficient evaluation

#28 §7 found `cnaster` uses `CountEncoder` deduplication rather than the
paper's cumulative sufficient statistics. Upstream does not have them either.

`opt/potts.py` does carry a "sufficient statistics" enumeration, and it is a
**different object**: configurations entering a partition function through
agreeing edges and sites per state, so `log Z` becomes a `logsumexp` over
precomputed rows. Exact enumeration for a Potts partition function, not the
per-observation cumulative counts `B^k_i = sum 1(b_gn > i)` that
`efficient_likelihood.tex` derives.

**So the paper's central efficiency derivation is implemented nowhere.** It is
the one method in the paper with no implementation on either side, which makes
it the cleanest candidate for **could exist upstream**: the derivation is
application-neutral -- a property of the beta-binomial and the Gamma
recursion, not of copy number -- and upstream already owns both families and
their M steps.

---

# Part B -- what upstream needs before it can referee (#32)

Restating #32's list against what was measured here. **Item 6 was wrong and
is corrected. Items 8 and 9 are answered and can close.**

| # | gap | status |
| ---: | --- | --- |
| 1 | exposure-aware count family | **confirmed** -- blocks a referee |
| 2 | beta-binomial trials taken exogenously | **confirmed** -- same change |
| 3 | transition varying along the chain | **confirmed** |
| 4 | structured transfer matrix | **confirmed**, expressiveness only |
| 5 | joint M step over a packed vector | **not needed** -- #37's separability is cheaper |
| 6 | a spatial layer | **wrong -- upstream carries it in full** |
| 7 | cross-framework ratios | **confirmed** -- a policy decision, not code |
| 8 | upstream's dtype policy | **answered** |
| 9 | a stress size upstream runs | **answered** |

## B0. All four gaps are one missing capability

Stated here because the list above reads as four unrelated features and is
not. Upstream's model is **position-independent by construction**: an
emission family carries one mean per state, and a recursion carries one
`(m, m)` for the whole chain. Every confirmed gap is the same consequence.

| item | the covariate | where it enters |
| ---: | --- | --- |
| 1 | exposure | the negative binomial's mean, `lam = exposure[i] * mu` |
| 2 | trials | the beta-binomial's `n_gn` |
| 3 | switch probability | the transition, per position |
| 4 | -- | the structured case of 3 |

Items 3 and 4 look like a different kind of thing and are not.
`hmm_phased`'s kernel varies along the chain **because** `recomb.py` computes
it from per-position genetic distance through Haldane's mapping function. It
is a covariate on the transition exactly as exposure is a covariate on the
mean, and upstream takes a single transition for the same reason
`NegativeBinomialEmission` takes a single mean -- both assume the parameters
do not vary with position.

**One concept, two code changes.** Items 1 and 2 are emission-side: an offset
threaded through `log_density`, `sample` and `reestimate`. Items 3 and 4 are
recursion-side: `forward_log_likelihood` and its backward and EM counterparts
taking a sequence of transitions rather than one. Different modules,
different tests. They should land as two tickets even though they answer one
question, and the emission pair should land first -- it unblocks #4's second
tier, #9's runtime half and #24's varying-trials regime, where the recursion
pair unblocks only the phased rung.

**Nothing else upstream is required.** A3's sufficient statistics are worth
having and block nothing; item 7 is a decision; items 5, 6, 8 and 9 resolve
to no upstream change at all. So the whole of what `port` needs from
upstream, to referee the whole of `cnaster`, is covariates.

## B1. Item 6 was wrong

#32 said: "Whether upstream's `opt/` carries the Potts problem is
unestablished. Measure before assuming." Measured: it carries it in full --
`PottsGraph` with per-edge couplings, an `energy` both sides can be scored
under, three solvers plus six more, a Rust `max_flow` and
`ising_ground_state`, an annealing schedule, and the `ground_state.py`
harness of A1.

PR #44 has already used the first two: `cnaster`'s `calc_assignment_cost` and
upstream's `energy` agree to `7.1e-15`, sign-flipped, with nothing tuned.

**The largest single correction in this audit.** #32's wording implied the
spatial rung might need upstream work; it needs none, and the rung is open
now.

## B2. Items 1-4 are confirmed, and 1 and 2 are one change

`NegativeBinomialEmission(dispersion, mean)` carries one mean per state and
no offset; `cnaster` forms `lam = exposure[i] * mu` per observation. Only a
*constant* exposure is the same problem, absorbed as `log_mu - log(c)`, and
every adapter in `tests/adapters.py` says so.

`BetaBinomialEmission(trials, alpha, beta)` carries trials per state where
`cnaster` carries one per observation -- the same shape one level over. #24
held trials constant for exactly this reason. **They are one change twice and
should land together.**

Items 3 and 4 are also one change: a transition that varies by position, with
the phased `2K x 2K` matrix as its structured case. #16 measured the
performance argument at 1.43-1.82x, below the 2x bar, so this is justified on
**expressiveness alone** -- upstream's recursion takes one `(m, m)`, so
`hmm_phased`'s per-position kernel has no upstream form and #9's item (3) can
be profiled but not refereed.

## B3. Item 5 is not needed

`_run_optimization_pipeline` optimizes everything at once; upstream
re-estimates per family. #37 takes the cheaper route: the `mode="em"`
objective separates additively -- NB parameters enter only the count channel,
BB only the allele channel -- so the joint argmax must equal upstream's two
independent `reestimate` calls. **A test in `port`, not a feature upstream.**

## B4. Item 8 answered -- upstream is `float64` by default

`opt/hmm.py`'s objectives take a `dtype`, defaulting to `float64`, with the
reason stated: "a finite-difference derivative check is meaningless in
`float32`". `float32` appears in 24 places and is opt-in throughout.

So `CLAUDE.md`'s cross-precision rule is usable: comparisons run in `float64`
on both sides, and the higher-precision tolerance is the only one in play
unless a test opts into `float32` deliberately. **Item 8 closes as a question
answered, with no change required.**

## B5. Item 9 answered -- reuse `spatio_only/release`

`ground_state.py` names open triangular 71x71, `q = 10`, `J = 0.7`, with a
recorded thermal tilt of 0.4935 over eight chains of 60 sweeps. #40 should
measure there rather than invent a size, so the two repositories' numbers are
comparable.

Upstream also supplies the arithmetic trap to avoid, which `port` would
otherwise have walked into:

> the tilt of a *ground* state is **below** the thermal number, not above it
> ... The thermal constant therefore referees samplers, and the exact ground
> state referees minimizers; carrying one across to the other is the
> arithmetic trap this paragraph exists to name.

## B6. Item 7 stands -- and it is a decision, not code

Upstream is `torch`, `cnaster` is `numpy`/`numba`. #24 measured 1.25x at gate
and 12.8x at stress and could report it honestly only by stating what the
ratio mixes: cost-per-iteration with iterations-taken, since neither side runs
a fixed number of steps.

Upstream's `Budget` and the site-visit unit in `ground_state.py` are the
answer for the **solver** comparisons -- a charged budget makes methods
comparable without a fixed-work entry point. For the **estimator**
comparisons (#24, #47) there is no equivalent, and the choice is between
adding a fixed-work entry point upstream and `port` no longer quoting
whole-solve ratios. Decide once and write it down, rather than in every
benchmark docstring.

---

# Part C -- a finding against `port`

`CLAUDE.md`: **Reach for what `snakes_and_ladders` already carries.**

PR #44 added to `tests/fixtures.py`:

```python
MAX_ENUMERABLE_LABELLINGS = 200_000
```

with a refusal in `enumerate_minimum_energy`. Upstream already ships
`snakes_and_ladders.enumeration.refuse_oversized` with
`MAX_ENUMERABLE_CONFIGURATIONS = 200_000` -- the same value, chosen there to
end exactly this proliferation:

> It varied four ways (issue #230): `200_000` configurations in
> `likelihood.potts`, `200_000` paths in `likelihood.hmm_paths`, `20` nodes
> in `search.max_cut`, and a docstring-only `n <= 6` ... four thresholds in
> three units.

`port` made it five. The module is explicitly importable by anything: "The
module names no model, so anything may import it."

**Action:** replace the constant and the refusal with a call to
`refuse_oversized(count, what="labellings")`. One line, and it is the rule
working as intended -- the duplication was found by writing the audit the
rule asks for.

**Resolved on #355**, one step further: `enumerate_minimum_energy` is now
`enumeration.enumerated_optimum` over the negated energy, with the
configurations read reversed so the enumeration order and the tie rule are
the ones the loop had. The constant is gone.

---

# Summary of actions

**Close now:** #32 items 8 and 9, answered. #32 item 6, wrong -- upstream
carries the spatial layer in full.

**Open upstream, two tickets, and they are one concept:** covariates. First
the emission pair -- exposure on the count mean, trials on the allele channel
(#32 items 1, 2) -- which unblocks the most. Then the recursion pair -- a
transition that varies by position, with the phased matrix as its structured
case (items 3, 4) -- argued on expressiveness, never on speed (#16).

**That is the complete list of upstream work.** Everything else resolves to
no change: see B0.

**Do not open upstream:** integer copy number (#25's enumeration is better),
phasing and segmentation, H&E, the joint M step (#37 is cheaper).

**Consider upstream:** the paper's sufficient statistics (A3) -- implemented
on neither side, application-neutral, and upstream already owns both emission
families.

**Redirect in `port`:** #8, #29 and #40 run against `search/ground_state.py`
rather than building a harness. #40 measures at `spatio_only/release`.

**Fix in `port`:** the duplicated enumeration limit (Part C).

**Still unowned:** `prior_model_selection.tex`'s Pareto front is implemented
on neither side and has no ticket (#28 §13).
