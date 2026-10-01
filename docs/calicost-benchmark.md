# CalicoST benchmark (#494, #532)

**TL;DR:** CalicoST on its shipped configuration finishes none of CalicoST
easy, hard and `dev_tree` 60 × 50 r0 in 30 minutes. The last clone assignment
it wrote, its BAF stage, scores 0.6686 (3 clones), 0.4675 (2) and 0.8601 (3).
Uncapped, with `n_clones 5`, it takes 20,243 s on `dev_tree` r0 and scores
clone ARI 0.8538 (6 clones), copy ARI 0.9075, and decodes losses as copy-neutral
LOH and balanced gains as neutral, as port does. port's numbers on the same
fixtures are in `docs/baseline-release.md`.

## Conditions

- Host: 4 cores, one job at a time.
- CalicoST: c1abcae, `python -m tests.final_benchmark SAMPLE calicost`:
  `run_calicost --shipped configuration_cna --no-align --no-figures`, once,
  under `timeout 1800`. Every value but the paths is CalicoST's shipped file,
  among them `n_clones 3`, `n_clones_rdr 2`, `npart_phasing 3`,
  `max_iter_outer 20`, `np_threshold 1.0`. The samples plant 4 clones.
  `dev_tree` has two slices and runs on CalicoST's joint file,
  `configuration_cna_multi`, with no alignment files. Its drawn slices name
  spots `<barcode>_<sample_id>`, which CalicoST's joint loader suffixes again,
  so the benchmark stages a copy with the suffix removed
  (`final_benchmark.joint_inputs`); the counts are unchanged.
- Peak is the child's maximum RSS.
- A run stopped by the cap is scored on the last clone assignment it wrote
  (`final_benchmark.baf_stage`): the BAF stage after CalicoST's
  Neyman–Pearson merge (`mergedallspots_nstates7_sp.npz`), or before it
  (`allspots_nstates7_sp.npz`, which CalicoST rewrites after every HMRF
  round, scored at its last completed round) where the merge had not run. It
  is not comparable with a final ARI: the read-depth stage had not finished.

## Results, capped at 1,800 s

| sample | clone ARI | wall | peak |
| --- | --- | --- | --- |
| CalicoST easy (`2d4ce9a9`) | not finished; BAF stage 0.6686 (3) | > 1800 s | 2.80 GB |
| CalicoST hard (`8797710b`) | not finished; BAF stage 0.4675 (2) | > 1800 s | 2.80 GB |
| `dev_tree` 60 × 50 r0 (`3381575a`) | not finished; BAF stage before the merge 0.8601 (3) | > 1800 s | 5.41 GB |

- **It stops in the read-depth stage on easy and hard**, having written
  `clone0_nstates7_smp.npz`, the first of its clones' refinements. On
  60 × 50 it stops inside the BAF stage's HMRF, after 2 of at most 20 rounds
  (easy had run 8 and hard 11 before their merges).
- **Its BAF stage asks for 3 clones**, as shipped, against 4 planted, so
  0.6686 and 0.8601 measure that setting as much as the fit. On hard the
  merge joined two of the three (0.4685 before it, 0.4675 after).

## Uncapped on `dev_tree` r0 (#532)

CalicoST ran to completion on its shipped `configuration_cna_multi` with
`n_clones 5` and no cap, on 3 cores.

| clone ARI | integer clone ARI | copy state ARI | copy ARI | exact altered | bins | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8538 (6) | 0.8538 (6) | 0.0889 | 0.9075 | 0.7095 (0.7095) | 2505 | 20,243 s | 3.43 GB |

- CalicoST was resumed twice from its npz checkpoints after the process was
  killed. Its wall is summed to the last checkpoint of each segment
  (5894 + 4697 + 9652 s).
- Copy state ARI compares the HMM state index with the planted (A, B), so it
  is low by construction.

### Every metric, from the committed outputs

`tests/data/benchmarks/dev_tree_r0/calicost.tar.xz`, scored by
`tests.sim_audit.score` against `dev_tree` r0 (`3381575a`) with no rerun.
`_pf` is phase-free; `—` is a class planted as one pair or not planted, where
ARI is undefined (`tests/sim_audit.py`).

| metric | all | LOH | balanced gain | unbalanced gain | neutral |
| --- | --- | --- | --- | --- | --- |
| copy ARI | 0.9075 | 0.0018 | — | — | — |
| copy ARI, `_pf` | 0.9075 | −0.0012 | — | — | — |
| exact | 0.9792 | 0.717 | 0.000 | — | 0.9929 |
| exact, `_pf` | | 0.717 | 0.000 | — | |

Exact altered is 0.7095 both phased and phase-free. Clone matching is planted
0 → 0, 1 → 1, 2 → 5, 3 → 4, with 6 fitted clones.

Planted (A, B) against decoded, as a fraction of each planted pair's
clone-bins (`--confusion-sampled`):

| planted \ decoded | 0,1 | 0,2 | 1,1 | 1,2 | 3,3 |
| --- | --- | --- | --- | --- | --- |
| 0,1 | 0.0175 | 0.9649 |  | 0.0175 |  |
| 1,0 |  | 0.8750 | 0.1250 |  |  |
| 0,2 | 0.0028 | 0.9577 | 0.0197 | 0.0197 |  |
| 1,1 | 0.0059 | 0.0005 | 0.9929 | 0.0005 | 0.0002 |
| 2,2 |  |  | 1.0000 |  |  |

- **Losses decode as copy-neutral LOH:** 96% of planted (0, 1) and 88% of
  (1, 0) bins come back (0, 2). Total copy 1 is all but never decoded.
- **Balanced gain decodes as neutral:** every planted (2, 2) bin comes back
  (1, 1).
- These are the two failures port's lattice decode shows on CalicoST easy
  (T- #471, T- #573), so they are not port's alone.

### Time per stage (`run.json`)

| stage | s |
| --- | --- |
| parse | 1,017 |
| BAF HMRF, 7 rounds | 3,934 |
| merge | 27 |
| read depth, clone 0 (8 rounds, 2,273 spots) | 1,337 |
| read depth, clone 1 (10 rounds, 1,910 spots) | 5,723 |
| read depth, clone 2 (7 rounds, 264 spots) | 1,891 |
| read depth, clone 3 (5 rounds, 1,553 spots) | 4,318 |
| combine, refit, reassign | 1,955 |
| integer copies and outputs | 41 |

Read-depth fitting is 13,269 s, 66% of the wall.

## Reproduce

```
python -m tests.final_benchmark numcnas1.2_cnasize5e7_ploidy2_random0 calicost
```

and the same for `numcnas6.3_cnasize1e7_ploidy2_random0` and `dev_tree` r0
at 60 × 50, drawn from `claude/467-dev-tree-size`'s manifest (a22d5ab) by
`python -m port.sim.draw sim/manifests/dev_tree.toml --into DIR`; the sample
argument is then `DIR/dev_tree/r0`.
