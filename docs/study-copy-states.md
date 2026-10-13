# Study: copy-state starts at known clones (#540)

**TL;DR:** on `dev_tree_1s_hard` (r0 `d2938975`) at the planted clones (10 realizations × 10 seeds), seven starts end
within 1.5% of rows missed after `--sal` Baum-Welch: `lattice`, `lattice` + EM and `hmc` (1.1%), `prior`,
`anneal` and parallel tempering (1.2%), and `gaussian-em` (1.4%, on the 20 of 100 runs it does not refuse).
`cnaster`'s own `gmm_init` ends at 44.3%, CalicoST's at 31.3%, and every `emission++` variant at
18.1–32.2%. A start's own miss rate does not predict the fit's: 5 × `emission++` starts at 1.4% and ends
at 19.9%.

**The paper's `solver_combined.png` draws a second generation (T- #662):** `sim/manifests/dev_tree_1s_hard.toml`,
lognormal lengths, r0 `9ec90dc2`, not the exponential `baseline/` manifest (r0 `d2938975`) below. There the three
HMM samplers end at 1.7% missed and `lattice` at 11.9%; see **Results: lognormal `dev_tree_1s_hard`**.

**Rerun at 67d8874 (sal b61dfba, #633):** these are the rerun's numbers. It reproduces the original to 0.1
point on 17 of 21 starts, and `gaussian-em` refuses 80 of 100 runs in both. The medoid of 30 `emission++`
seeds ends at 1.1% missed ([0.9, 4.2] over r9–r12); the run of highest likelihood among them ends at 41.9%.

## Method

`run_study --copy-state-stream sim/sandbox/manifests/baseline/dev_tree_1s_hard.toml OUT --problems N --seeds 10 --held-out 3 --settings configs/copy_sampler_settings.json` (the default; `run_calibrate --copy` writes it).

1. **Problem.** Since #730, the run's own: each realization of `dev_tree_1s_hard` is drawn to disk as the
   run reads a sample, and `run_cnaster_port --sal` runs on it at its planted clones up to the RDR + BAF
   stage's Baum-Welch (`port.qa.stage`). The run's segments, phasing, pseudobulk and exposure, clones
   stacked along the genome; about 7,300 rows per realization. Before #730 the study rebuilt the problem
   (`port.sandbox.known_copy`, deleted): 1 Mb bins under #551's floor and truth-phased allele reads.
2. **Starts.** Every family in `port.sandbox.extensions.copy_starts`, each as its own algorithm's output, with no
   `sal` mixture polish. A stochastic start runs 10 seeds.
   Since #716 `STARTS` holds the 7 the paper figure draws (`calicost-gmm`, `lattice`, `prior`, `kmeans++`,
   `emission++`, `tempering-hmm`, `hmc-hmm`; `gaussian-em` and `anneal-hmm` left with #716); `--all` runs the
   rest. The table below predates it.
3. **Polish.** The run's own `pipeline_baum_welch` call at that stage, every argument as the run built it,
   with the start as `init_log_mu` and `init_p_binom`. A start is scored by the same call at `max_iter = 0`.
4. **Scored.** Gap: nats below the best log-likelihood any run reached on that realization. Missed: rows
   whose state is not the planted one under the best 1-1 matching of states. Both are reported at the
   start and after Baum-Welch. Truth is the planted states, each class's pooled depth ratio and B share, polished by the same call; the run's own initializer is recorded beside it.
5. **Tuning.** `anneal-hmm`, `tempering-hmm`, `hmc-hmm` and the `emission++` variants are tuned on 3
   held-out realizations that are never evaluated (`copy_sampler_settings.json`).

## Results: `dev_tree_1s_hard`, 10 realizations × 10 seeds

`dev_tree_1s_hard` r3–r12 of `[sample] seed = 0`, r0 hashing to `d2938975`
(`port.sim.fixtures.realization_hash`); r0–r2 tuned the samplers. The draws
are the gamma sampler's: this branch's `port.sim.draw` has no `counts_sampler`.

Median gap below the best fit reached on each realization [nats], at the start and after Baum-Welch; median
rows missed; median seconds for start and Baum-Welch together. The planted states, polished, sit 120 nats
below the best (median; 126 in the original). Rerun at 67d8874 on sal b61dfba: r3–r4 ran from scratch and
equal the dbfff01 runs in 348 of 348 (missed, and polished log-likelihood to 1e-9), so r5–r6 are the
dbfff01 runs and r3–r4, r7–r12 are new. #15 and #16 were dropped after r10 by decision, to save about 19%
of the time (11.9% + 7.5%), and run on 8. The `anchor` and `knn` variants run on all 10 here; the original
ran them on r4–r12, where the rerun's kNN reads 25.1%, as the original. Seconds mix two hosts and are not
compared.

| # | start | start gap | after BW | Missed start / after [%] | s |
| --- | --- | --- | --- | --- | --- |
| 4 | `lattice` | 4,354 | 112 | 1.1 / 1.1 | 11 |
| 21 | `hmc` on the HMM, warmed, T = 10 | 4,395 | 120 | 1.1 / 1.1 | 12 |
| 5 | `lattice` + EM | 4,381 | 105 | 1.1 / 1.1 | 19 |
| 7 | `prior` | 13,733 | 130 | 1.4 / 1.2 | 8 |
| 20 | parallel tempering on the HMM | 4,372 | 123 | 1.1 / 1.2 | 17 |
| 19 | `anneal` on the HMM | 4,384 | 122 | 1.2 / 1.2 | 18 |
| 11 | `gaussian-em` | 5,516 | 397 | 1.5 / 1.4 | 8 |
| 15 | 20 × trimmed `emission++` | 4,652 | 199 | 1.4 / 18.1 | 14 |
| 10 | `emission++` | 4,927 | 213 | 1.9 / 18.2 | 8 |
| 6 | `rdr-quantiles` | 5,148 | 1,225 | 7.4 / 18.5 | 10 |
| 14 | 5 × `emission++` by HMM likelihood | 4,639 | 136 | 1.4 / 19.9 | 10 |
| 13 | `emission++`, trimmed | 4,961 | 201 | 3.0 / 21.7 | 9 |
| 18 | `emission++`, kNN seeds | 5,160 | 221 | 5.6 / 26.5 | 9 |
| 9 | `k-means++` | 5,081 | 253 | 4.6 / 26.6 | 8 |
| 17 | `emission++`, neutral anchor | 4,886 | 227 | 3.1 / 29.9 | 11 |
| 2 | CalicoST `initialization_by_gmm` | 4,921 | 197 | 2.6 / 31.3 | 7 |
| 16 | 5 × Lloyd-refined `emission++` | 4,672 | 233 | 2.1 / 32.2 | 22 |
| 12 | `quantile` | 5,132 | 279 | 2.0 / 41.6 | 10 |
| 8 | `data` | 5,101 | 286 | 17.3 / 43.2 | 8 |
| 1 | `cnaster` `gmm_init` | 5,114 | 743 | 16.0 / 44.3 | 8 |
| 3 | `distinct` (#348) | 5,013 | 267 | 9.3 / 45.3 | 9 |

Rows 19-21 are port's own samplers, deleted in #634 for `sal.sample.hmc`'s on the same objective
(`port.sandbox.extensions.hmm_objective`, formerly `known_copy.hmm_objective`), tuned on r0-r2 and run on r3-r12 × 10 seeds (PR- #642,
n = 100 each, start and Baum-Welch seconds under the host lock with only the six sampler starts):

| start | start gap | after BW | Missed after [%], median (95% bootstrap) | runs > 5% missed | s |
| --- | --- | --- | --- | --- | --- |
| `anneal`, `sal` / port | 4,372 / 4,384 | 126 / 122 | 1.14 (1.07-1.19) / 1.19 (1.12-1.68) | 24% / 37% | 20.8 / 19.6 |
| tempering, `sal` / port | 4,425 / 4,372 | 127 / 123 | 1.12 (1.09-1.18) / 1.17 (1.11-1.60) | 15% / 29% | 19.5 / 17.5 |
| `hmc`, `sal` / port | 4,404 / 4,395 | 122 / 120 | 1.14 (1.09-1.19) / 1.10 (1.08-1.20) | 16% / 25% | 13.8 / 13.6 |

`gaussian-em` refuses on 80 of 100 runs, as in the original: a component's variance collapses on the normal
clone's point mass, and `sal` refuses rather than floor it (`ValueError` at the variance floor, every error
of the rerun). The figure is `port.studies.copy_state_plot` over the merged stream,
stamped with its data hash and code commit.

## Results: lognormal `dev_tree_1s_hard`, 10 realizations × 10 seeds (T- #662)

`sim/manifests/dev_tree_1s_hard.toml` r3–r12 (r0 `9ec90dc2`, lognormal event lengths, #619): the same name as
the table above, a different dataset. Record data `34aa0151`, drawn as panel (b) of
`docs/plots/paper/solver_combined.png`. `STARTS` only, `--seeds 10 --held-out 3 --settings
tests/studies/copy_sampler_settings.json`, 3 workers under the host lock, code `9efa28d`, sal `253c84f`.

Medians over realizations × seeds (`lattice` is deterministic: 10 runs). Gap columns as above; the planted
states, polished, sit 115 nats below the best and miss 2.9% of rows (median).

| start | runs | start gap | after BW | Missed start / after [%] | runs > 5% missed | s |
| --- | --- | --- | --- | --- | --- | --- |
| `anneal-hmm` | 100 | 4,285 | 142 | 1.6 / 1.7 | 22% | 59 |
| `tempering-hmm` | 100 | 4,384 | 141 | 1.7 / 1.7 | 17% | 63 |
| `hmc-hmm` | 98 | 4,345 | 149 | 1.7 / 1.7 | 13% | 56 |
| `gaussian-em` | 100 | 5,777 | 632 | 3.0 / 2.4 | 35% | 52 |
| `prior` | 100 | 16,192 | 584 | 2.9 / 2.6 | 24% | 53 |
| `lattice` | 10 | 4,243 | 139 | 1.5 / 11.9 | 70% | 62 |
| `emission++` | 100 | 4,927 | 300 | 3.3 / 24.5 | 67% | 62 |
| `kmeans++` | 100 | 5,159 | 442 | 6.5 / 29.8 | 83% | 54 |
| `calicost-gmm` | 100 | 4,829 | 437 | 4.4 / 38.6 | 94% | 59 |

- **`lattice`'s neutral-state split (#564).** Missed after Baum-Welch per realization, r3–r12: 22.4, 1.4, 12.1,
  1.1, 31.3, 11.6, 8.6, 2.1, 28.1, 31.9%. It starts within 1.5% (median) on every realization; Baum-Welch then
  splits the neutral state, as on r3 and r5 (T- #662). The planted states, polished, miss 27.4–44.5% on r3, r7,
  r11 and r12, so the split is a property of the likelihood on this generation, not of the start. On the
  `baseline/` generation `lattice` ends at 1.1%.
- **Two refusals.** `hmc-hmm` seed 4 on r11 and r12: `sal`'s warm-up refuses a chain that never moved. sal #1207
  (`006e49d`, T- #707) reports such a coordinate on `Adapted.flat` instead; these runs were not repeated.
  `gaussian-em` refused 50 of 50 runs on r3–r7 in PR- #661's record (data `8ab44e62`), on the venv's earlier sal
  (not recorded); on sal `253c84f` it refuses none, at #661's code `901ea94` and at `9efa28d` alike (r3, seed 0,
  same rows missed). Its numbers depend on the sal commit.
- **Assembly.** r3–r7 of `calicost-gmm`, `lattice`, `prior`, `kmeans++`, `emission++` are #661's runs: at
  `9efa28d` r3 seed 0 reproduces them bit for bit (missed and log-likelihood). The HMM samplers became `sal`'s
  after #661 (#634) and `gaussian-em` changed with sal, so those four reran on r3–r7; r8–r12 ran in full.
- **Seconds.** Jobs timed 03:23–04:14Z on 2026-10-06 shared the host with unlocked test gates; their r3–r5 jobs
  reran under the lock, results identical (120 of 120), r3's median job 84 s → 59 s. #661's reused rows
  (2026-10-05) sit within the interquartile range of the same starts' r8–r12 jobs.

The key figure (`data 7cee0a0a · code 67d8874`) is retired from `docs/plots/paper/` (#745), superseded by
`solver_combined.png` (b); `run_study --copy-state-plot OUT/<stem>.record` redraws it beside the record.

## Defects found, and what was done about them

- **`cnaster`'s NB kernel scores probability 1 when `p` rounds to 1** (#560). Baum-Welch drove a state to
  `log mu = -43` and reported -23,359 nats against the planted -76,306. Patched in log space
  (`port.patch.hmm_nophasing.nb_logpmf`), pinned against scipy to 1e-9.
- **Beta-binomial precision at large tau** (#561) and **negative `emission++` divergences** (#562): the
  latter clamped at 0 for the study.
- The numerics now come from PR- #594 (`port.pipeline.LOG_SPACE_SWAPS`; the divergence floor is sal #1136's since T- #632).
  The numbers above were measured under this branch's own patch, which swapped the negative binomial
  alone; `port.sandbox.known_copy` now also runs PR- #594's beta-binomial (#561). Rerun at 67d8874: the
table above, which matches the original on 17 of 21 starts.
- **`sal`'s surrogate samplers sample a Gaussian mixture on raw counts and snap to observed rows** (#563).
  Replaced in the study by samplers on the HMM's own NLL: port's own at first, `sal.sample.hmc`'s through
  `port.sandbox.known_copy.hmm_objective` since #634.
- **The per-clone shift lets Baum-Welch split the neutral state** (#564): 42% missed from the truth start on
  one realization, at a 505-nat higher likelihood than the correct fit.

## Choosing among seeds (rerun)

Missed after Baum-Welch [%] of the run chosen among seeds 0 to k-1, median [min, max] over realizations: (a)
highest start log-likelihood, (b) highest polished log-likelihood, (c) the lower median of the polished
log-likelihoods, (d) the medoid, the run whose decoded path disagrees least with the other k-1 under the
best 1-1 state matching.

| `emission++` | k | (a) | (b) | (c) | (d) |
| --- | --- | --- | --- | --- | --- |
| r3–r12 | 1 | 1.3 [0.8, 48.7] | 1.3 [0.8, 48.7] | = | = |
| r3–r12 | 10 | 36.8 [1.8, 56.4] | 40.9 [2.9, 60.3] | 22.9 [1.5, 56.8] | 1.3 [1.0, 43.1] |
| r9–r12 | 30 | 25.9 [2.5, 40.1] | 41.9 [27.2, 53.0] | 39.6 [14.0, 49.5] | 1.1 [0.9, 4.2] |

- `emission++` is bimodal per run: 32% of 80 runs on r3–r10 end at or below 2% missed, 46% at or above 20%.
- The polished likelihood does not rank the correct fit first: 26 of those 80 runs end above the planted
  states' polished log-likelihood, and within a realization Spearman(log-likelihood, missed) has median
  +0.09. Choosing by likelihood picks the wrong optimum more often as k grows.
- The medoid needs no truth, and at k = 30 costs about 30 × 8 s against `lattice`'s 11 s for 1.1%. It is not
  proposed as a default. Its paths are re-decoded at the fixed `ALPHA`, `TAU`, since Baum-Welch's fitted
  dispersions are not stored. Those paths differ from Baum-Welch's by up to 26 points of missed (Spearman 0.89,
  r3–r4). The missed reported is the chosen run's own.

## Why good starts end badly

The HMM likelihood has higher-scoring wrong optima: an extra deep-loss state and a split neutral state
(#471, #564). A start's miss rate therefore does not predict the fit's: `lattice` starts at 1.0% missed
and ends at 59.3% on one screened realization.

## Upstream correspondence

`snakes_and_ladders` has no per-clone normalization and no copy-state HMM start. Expressing the problem
there would need a clone-indexed rate offset in its NB emission and a registry of HMM starts. The samplers
on the HMM's NLL would need `sal` to sample the HMM objective rather than its Gaussian surrogate (#563). sal b61dfba's
`hmc`, `anneal` and `tempering` seedings now sample the mixture's own likelihood (sal #1136), not the HMM's.

## Tuned at the run's Baum-Welch (#723)

**TL;DR:** `tempering-hmm` and `hmc-hmm` were retuned at the run's Baum-Welch at the planted clones (#730), on `dev_tree_1s_hard` r0–r2 held out. Their median gap at the start's states falls from the default settings' 722.7 to 100.6 nats (tempering) and 170.3 to 131.6 (HMC). After Baum-Welch the two move in opposite directions on `solver_combined`'s 5 realizations × 5 seeds:

| start | rows missed after BW, untuned → tuned | median gap after BW (nats), untuned → tuned |
| --- | --- | --- |
| `hmc-hmm` | 2.6 → 2.4% | 127 → 175 |
| `tempering-hmm` | 2.6 → 2.9% | 136 → 120 |

- **Settings:** `hmc-hmm` temperature 1 → 100; `tempering-hmm` step 0.001 → 0.003, top temperature 10,000 → 100.
- **Why they diverge:** the tune ranks settings by the gap at the start's states, before Baum-Welch. A better start does not predict a better fit (this doc's *Why good starts end badly*).
