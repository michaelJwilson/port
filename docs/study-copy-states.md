# Study: copy-state starts at known clones (#540)

**TL;DR:** on `dev_tree_1s_hard` (r0 `d2938975`) at the planted clones (10 realizations × 10 seeds), seven starts end
within 1.5% of rows missed after `--sal` Baum-Welch: `lattice`, `lattice` + EM and `hmc` (1.1%), `prior`,
`anneal` and parallel tempering (1.2%), and `gaussian-em` (1.4%, on the 20 of 100 runs it does not refuse).
`cnaster`'s own `gmm_init` ends at 44.3%, CalicoST's at 31.3%, and every `emission++` variant at
18.1–32.2%. A start's own miss rate does not predict the fit's: 5 × `emission++` starts at 1.4% and ends
at 19.9%.

**Rerun at 67d8874 (sal b61dfba, #633):** these are the rerun's numbers. It reproduces the original to 0.1
point on 17 of 21 starts, and `gaussian-em` refuses 80 of 100 runs in both. The medoid of 30 `emission++`
seeds ends at 1.1% missed ([0.9, 4.2] over r9–r12); the run of highest likelihood among them ends at 41.9%.

## Method

`python -m tests.studies.copy_state_stream sim/manifests/baseline/dev_tree_1s_hard.toml OUT --problems N --seeds 10 --held-out 3 --settings tests/studies/copy_sampler_settings.json`.

1. **Problem.** Each realization of `dev_tree_1s_hard` is drawn and pseudobulked at its planted clones
   (`port.sandbox.known_copy.problems`): 1 Mb bins under #551's 300 normal-UMI floor, phased allele
   reads, clones stacked along the genome, 7 states. About 7,700 rows per realization.
2. **Starts.** Every family in `port.sandbox.extensions.copy_starts`, each as its own algorithm's output, with no
   `sal` mixture polish. A stochastic start runs 10 seeds.
   Since T- #660 `STARTS` holds the 9 the paper figure draws (`calicost-gmm`, `lattice`, `prior`, `kmeans++`,
   `emission++`, `gaussian-em` and the three HMM samplers); `--all` runs the rest. The table below predates it.
3. **Polish.** `--sal` Baum-Welch: `port.patch.hmm_nophasing` with the per-clone shift (#276, #293), `sal`
   emission kernels, analytic gradients and the Rust lattice. Every score, a start's included, is on this
   objective: a start is decoded, shifted per clone, and rescored.
4. **Scored.** Gap: nats below the best log-likelihood any run reached on that realization. Missed: rows
   whose state is not the planted one under the best 1-1 matching of states. Both are reported at the
   start and after Baum-Welch. Truth is the planted states, polished by the same Baum-Welch.
5. **Tuning.** `anneal-hmm`, `tempering-hmm`, `hmc-hmm` and the `emission++` variants are tuned on 3
   held-out realizations that are never evaluated (`copy_sampler_settings.json`).

## Results: `dev_tree_1s_hard`, 10 realizations × 10 seeds

`dev_tree_1s_hard` r3–r12 of `[sample] seed = 0`, r0 hashing to `d2938975`
(`tests.sim_stages.realization_hash`); r0–r2 tuned the samplers. The draws
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

`gaussian-em` refuses on 80 of 100 runs, as in the original: a component's variance collapses on the normal
clone's point mass, and `sal` refuses rather than floor it (`ValueError` at the variance floor, every error
of the rerun). The figure is `tests.studies.copy_state_plot` over the merged stream,
stamped with its data hash and code commit.

The key figure is committed as `docs/plots/paper/key_studies/557_copy-states.png` (`data 7cee0a0a ·
code 67d8874`); `python -m tests.studies.copy_state_plot OUT/<stem>.pkl` redraws it beside the pickle.

## Defects found, and what was done about them

- **`cnaster`'s NB kernel scores probability 1 when `p` rounds to 1** (#560). Baum-Welch drove a state to
  `log mu = -43` and reported -23,359 nats against the planted -76,306. Patched in log space
  (`port.patch.hmm_nophasing.nb_logpmf`), pinned against scipy to 1e-9.
- **Beta-binomial precision at large tau** (#561) and **negative `emission++` divergences** (#562): the
  latter clamped at 0 for the study.
- The numerics now come from PR- #594 (`port.pipeline.LOG_SPACE_SWAPS`, `sal_mixture.clamped_divergence`).
  The numbers above were measured under this branch's own patch, which swapped the negative binomial
  alone; `port.sandbox.known_copy` now also runs PR- #594's beta-binomial (#561). Rerun at 67d8874: the
table above, which matches the original on 17 of 21 starts.
- **`sal`'s surrogate samplers sample a Gaussian mixture on raw counts and snap to observed rows** (#563).
  Replaced in the study by samplers on the HMM's own NLL (`port.sandbox.known_copy.hmm_samplers`).
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
