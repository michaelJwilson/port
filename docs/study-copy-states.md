# Study: copy-state starts at known clones (#540)

**TL;DR:** on `dev_tree_1s_hard` (r0 `d2938975`) at the planted clones (10 realizations × 10 seeds), seven starts end
within 1.5% of rows missed after `--sal` Baum-Welch: `lattice` (1.0%), `lattice` + EM and `hmc` (1.1%),
`prior`, `anneal` and parallel tempering (1.2%), and `gaussian-em` (1.4%, where it does not refuse).
`cnaster`'s own `gmm_init` ends at 44.3%, CalicoST's at 31.3%, and every `emission++` variant at
18.2–38.3%. A start's own miss rate does not predict the fit's: 5 × `emission++` starts at 1.4% and ends
at 19.9%.

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
rows missed; median seconds for start and Baum-Welch together. The planted states, polished, sit 126 nats
below the best (median). The `anchor` and `knn` variants joined after the first realization and run on 9.

| # | start | start gap | after BW | Missed start / after [%] | s |
| --- | --- | --- | --- | --- | --- |
| 4 | `lattice` | 4,354 | 112 | 1.1 / 1.0 | 35 |
| 5 | `lattice` + EM | 4,381 | 105 | 1.1 / 1.1 | 43 |
| 21 | `hmc` on the HMM, warmed, T = 10 | 4,395 | 120 | 1.1 / 1.1 | 19 |
| 7 | `prior` | 13,733 | 130 | 1.4 / 1.2 | 13 |
| 19 | `anneal` on the HMM | 4,384 | 122 | 1.2 / 1.2 | 31 |
| 20 | parallel tempering on the HMM | 4,372 | 123 | 1.1 / 1.2 | 29 |
| 11 | `gaussian-em` | 5,516 | 397 | 1.5 / 1.4 | 13 |
| 6 | `rdr-quantiles` | 5,148 | 1,225 | 7.4 / 18.5 | 16 |
| 10 | `emission++` | 4,927 | 213 | 1.9 / 18.2 | 11 |
| 14 | 5 × `emission++` by HMM likelihood | 4,639 | 136 | 1.4 / 19.9 | 16 |
| 15 | 20 × trimmed `emission++` | 4,624 | 199 | 1.4 / 21.2 | 22 |
| 13 | `emission++`, trimmed | 4,961 | 202 | 3.0 / 22.4 | 13 |
| 18 | `emission++`, kNN seeds | 5,160 | 231 | 6.0 / 25.1 | 14 |
| 9 | `k-means++` | 5,081 | 253 | 4.6 / 26.6 | 12 |
| 17 | `emission++`, neutral anchor | 4,833 | 229 | 3.0 / 29.9 | 16 |
| 2 | CalicoST `initialization_by_gmm` | 4,921 | 197 | 2.6 / 31.3 | 13 |
| 16 | 5 × Lloyd-refined `emission++` | 4,650 | 219 | 1.9 / 38.3 | 30 |
| 12 | `quantile` | 5,132 | 279 | 2.0 / 41.6 | 12 |
| 8 | `data` | 5,101 | 286 | 17.3 / 43.2 | 11 |
| 1 | `cnaster` `gmm_init` | 5,114 | 743 | 16.0 / 44.3 | 13 |
| 3 | `distinct` (#348) | 5,013 | 267 | 9.3 / 45.3 | 15 |

`gaussian-em` refuses on some seeds: a component's variance collapses on the normal clone's point mass, and
`sal` refuses rather than floor it. The figure is `tests.studies.copy_state_plot` over the merged stream,
stamped with its data hash and code commit.

The figure is not committed: `python -m tests.studies.copy_state_plot OUT/<stem>.pkl`
redraws it beside the pickle (`git show ba34716:docs/plots/studies/copy_states_dev_tree_1s_hard.png`).

## Defects found, and what was done about them

- **`cnaster`'s NB kernel scores probability 1 when `p` rounds to 1** (#560). Baum-Welch drove a state to
  `log mu = -43` and reported -23,359 nats against the planted -76,306. Patched in log space
  (`port.patch.hmm_nophasing.nb_logpmf`), pinned against scipy to 1e-9.
- **Beta-binomial precision at large tau** (#561) and **negative `emission++` divergences** (#562): the
  latter clamped at 0 for the study.
- The numerics now come from PR- #594 (`port.pipeline.LOG_SPACE_SWAPS`, `sal_mixture.clamped_divergence`).
  The numbers above were measured under this branch's own patch, which swapped the negative binomial
  alone; `port.sandbox.known_copy` now also runs PR- #594's beta-binomial (#561). Not rerun.
- **`sal`'s surrogate samplers sample a Gaussian mixture on raw counts and snap to observed rows** (#563).
  Replaced in the study by samplers on the HMM's own NLL (`port.sandbox.known_copy.hmm_samplers`).
- **The per-clone shift lets Baum-Welch split the neutral state** (#564): 42% missed from the truth start on
  one realization, at a 505-nat higher likelihood than the correct fit.

## Why good starts end badly

The HMM likelihood has higher-scoring wrong optima: an extra deep-loss state and a split neutral state
(#471, #564). A start's miss rate therefore does not predict the fit's: `lattice` starts at 1.0% missed
and ends at 59.3% on one screened realization.

## Upstream correspondence

`snakes_and_ladders` has no per-clone normalization and no copy-state HMM start. Expressing the problem
there would need a clone-indexed rate offset in its NB emission and a registry of HMM starts. The samplers
on the HMM's NLL would need `sal` to sample the HMM objective rather than its Gaussian surrogate (#563).
