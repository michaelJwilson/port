# CalicoST benchmark (#494, #532)

**TL;DR:** CalicoST on its shipped configuration finishes none of CalicoST
easy, hard and `dev_tree` 60 × 50 r0 in 30 minutes. The last clone assignment
it wrote, its BAF stage, scores 0.6686 (3 clones), 0.4675 (2) and 0.8601 (3).
Uncapped, with `n_clones 5`, it takes 20,243 s on `dev_tree` r0 and scores
clone ARI 0.8538 (6 clones), copy ARI 0.9075. port's numbers on the same
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
  (5894 + 4697 + 9652 s). Read-depth fitting is 13,269 s of it (66%); #532
  has the timing for each stage.
- Copy state ARI compares the HMM state index with the planted (A, B), so it
  is low by construction.

## Reproduce

```
python -m tests.final_benchmark numcnas1.2_cnasize5e7_ploidy2_random0 calicost
```

and the same for `numcnas6.3_cnasize1e7_ploidy2_random0` and `dev_tree` r0
at 60 × 50, drawn from `claude/467-dev-tree-size`'s manifest (a22d5ab) by
`python -m port.sim.draw sim/manifests/dev_tree.toml --into DIR`; the sample
argument is then `DIR/dev_tree/r0`.
