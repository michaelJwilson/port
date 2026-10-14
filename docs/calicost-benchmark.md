# CalicoST benchmark (#494, #532)

**TL;DR:** given 6 hours instead of 30 minutes, CalicoST finishes easy in
6,497 s and hard in 4,474 s, and **both score worse than the BAF stage the
30-minute cap had stopped them at**: clone ARI 0.5303 (4 clones) against
0.6686, and 0.0690 (2 clones) against 0.4675. On hard it fits 2 clones for
the 4 planted and decodes no altered bin in phase. Uncapped on `dev_tree`
60 × 50 r0 it takes 20,243 s and scores clone ARI 0.8538 (6 clones), copy ARI
0.9075. It decodes losses as copy-neutral LOH and balanced gains as neutral
on every fixture, as port does. port's numbers on the same fixtures are in
`docs/baseline-release.md`.

## Conditions

- Host: 4 cores, one job at a time.
- CalicoST: c1abcae, `run_benchmark --final SAMPLE calicost`:
  `run_calicost --shipped configuration_cna --no-align --no-figures`, once,
  under `timeout 1800`. Every value but the paths is CalicoST's shipped file,
  among them `n_clones 3`, `n_clones_rdr 2`, `npart_phasing 3`,
  `max_iter_outer 20`, `np_threshold 1.0`. The samples plant 4 clones.
  `dev_tree` has two slices and runs on CalicoST's joint file,
  `configuration_cna_multi`, with no alignment files. Its drawn slices name
  spots `<barcode>_<sample_id>`, which CalicoST's joint loader suffixes again,
  so the benchmark stages a copy with the suffix removed
  (`port.qa.benchmark.joint_inputs`); the counts are unchanged.
- Peak is the child's maximum RSS.
- A run stopped by the cap is scored on the last clone assignment it wrote
  (`port.qa.benchmark.baf_stage`): the BAF stage after CalicoST's
  Neyman–Pearson merge (`mergedallspots_nstates7_sp.npz`), or before it
  (`allspots_nstates7_sp.npz`, which CalicoST rewrites after every HMRF
  round, scored at its last completed round) where the merge had not run. It
  is not comparable with a final ARI: the read-depth stage had not finished.

## Reproduce on `dev_tree` r0

`tests/calicost_benchmark.sh [ROOT]` reruns the uncapped comparison below on the fixture it was measured on:
- **Fixture:** it draws `sim/manifests/baseline/dev_tree.toml` r0 and refuses it unless it hashes to `3381575a`.
- **CalicoST:** shipped `configuration_cna_multi` with `n_clones 5`, no cap. A killed attempt is rerun on the same kept root and resumes from CalicoST's checkpoints (`port.qa.benchmark.staged`, `--root`); `TRIES` defaults to 3, and the wall is the attempts' sum.
- **port:** `run_cnaster_port --sal`, `REPEATS` times (default 3), median wall.
- **Output:** every `BENCH` line goes to `ROOT/bench.jsonl`, and the script prints the table.

Run it alone on the host; `LOCK=<file>` holds a flock.

`run_study --benchmark-table [--repeats 3]` writes the paper's table,
`docs/benchmark.tex`, on the same fixture: CalicoST from its committed outputs
(`tests/data/benchmarks/dev_tree_r0/calicost.tar.xz`, not rerun) and port
`--sal` run at the current commit, which the file names with the sal pin.

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

## Results, capped at 21,600 s (#494)

Six hours apiece, on the host and conditions above; `--timeout 21600`. Both
finish well inside the cap, so these are completed runs rather than a cap's
reading of one.

| sample | clone ARI | copy ARI (`_pf`) | exact altered (`_pf`) | bins | wall | peak |
| --- | --- | --- | --- | --- | --- | --- |
| CalicoST easy (`2d4ce9a9`) | 0.5303 (4) | 0.6098 (0.6094) | 0.2326 (0.4651) | 1476 | 6,497 s | 2.79 GB |
| CalicoST hard (`8797710b`) | 0.0690 (2) | 0.6285 (0.6285) | 0.0000 (0.5500) | 1467 | 4,474 s | 2.78 GB |

- **Finishing costs clone ARI rather than earning it.** The 30-minute cap
  scored the BAF stage at 0.6686 on easy and 0.4675 on hard; the finished
  runs score 0.5303 and 0.0690. The read-depth stage, which the cap never
  reached on either, is what the difference is made in. The two numbers are
  not otherwise comparable -- a BAF-stage reading is not a final ARI -- but
  they are the same quantity on the same fit, and it falls.
- **Hard fits 2 clones for 4 planted,** so two planted clones have no fitted
  counterpart and its copy metrics are over the two that do. Easy fits 4.
- **Neither decodes an aberration well.** `exact_neutral` is 0.9924 on easy
  and 0.9973 on hard, so the neutral genome is right and the aberrations are
  what is missed; `exact_altered` is 0.2326 and 0.0000.
- Read depth is 87.5 of easy's 108.3 minutes (81%) and 56 of hard's 74.6
  (75%), against 66% on `dev_tree` r0.

### Time per stage

| stage | easy, min | hard, min |
| --- | --- | --- |
| parse | 4.0 | 6 |
| BAF HMRF and merge | 16.4 | 18.1 |
| read depth, clone 0 | 55.2 | 33.3 |
| read depth, clone 1 | 68.5 | 69.9 |
| read depth, clone 2 | 104.2 | — |
| combine, refit, decode, outputs | 108.1 | 74.4 |

Cumulative minutes from the start, as each stage's file was last written.
Hard refines two clones where easy refines three, because its BAF stage left
fewer.

### Planted against decoded

Each planted `(A, B)` against the decoded, as a fraction of that pair's
matched clone-bins (`--confusion-sampled`); a row sums to 1.

**easy:**

| planted \ decoded | 0,2 | 1,1 | 1,2 | 1,3 | 3,3 |
| --- | --- | --- | --- | --- | --- |
| 1,0 | 0.9474 | 0.0526 | | | |
| 1,1 | | 0.9924 | | 0.0055 | 0.0021 |
| 1,2 | | | 1.0000 | | |
| 2,1 | | 1.0000 | | | |
| 2,2 | | 1.0000 | | | |
| 3,1 | | 0.0323 | | 0.9677 | |

**hard:**

| planted \ decoded | 1,1 | 1,2 |
| --- | --- | --- |
| 0,1 | 0.5000 | 0.5000 |
| 1,1 | 0.9973 | 0.0027 |
| 2,1 | 0.3125 | 0.6875 |

- **Losses decode as copy-neutral LOH, or as neutral.** On easy 94.74% of
  planted (1, 0) comes back (0, 2); on hard half of planted (0, 1) comes back
  (1, 1) and half (1, 2). Total copy 1 is never decoded on either.
- **Balanced gain decodes as neutral:** every planted (2, 2) bin on easy comes
  back (1, 1). Hard plants none among its matched clones.
- These are #532's two failures on `dev_tree` r0, now on finished runs of the
  samples CalicoST's own paper ships, and they are the failures port's lattice
  decode shows on easy (T- #471, T- #573). They are not port's alone.

### Outputs

`tests/data/benchmarks/easy/` and `tests/data/benchmarks/hard/` hold each
run's `calicost.tar.xz` and a README with its row, so a metric can be taken
from them without a rerun.

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
run_benchmark --final numcnas1.2_cnasize5e7_ploidy2_random0 calicost --timeout 21600
```

and the same for `numcnas6.3_cnasize1e7_ploidy2_random0` and `dev_tree` r0
at 60 × 50, drawn from `claude/467-dev-tree-size`'s manifest (a22d5ab) by
`python -m port.sim.draw sim/manifests/dev_tree.toml --into DIR`; the sample
argument is then `DIR/dev_tree/r0`.
