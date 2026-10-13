# CalicoST and port `--sal` on `dev_tree` 60 × 50 r0 (#532)

**TL;DR:** the outputs both tools wrote on one realization, and its truth, so
any CNA-recovery metric can be computed and compared without a rerun.
Scoring these archives with `tests.sim_audit.score` reproduces #532's table:

| tool | clone ARI | integer clone ARI | state ARI | copy ARI | exact | exact altered (phase-free) | bins | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CalicoST | 0.8538 (6) | 0.8538 (6) | 0.0889 | 0.9075 | 0.9792 | 0.7095 (0.7095) | 2505 | 20,243 s, 3 cores | 3.43 GB |
| port `--sal` | 0.8612 (5) | 1.0 (4) | 0.0682 | 0.9828 | 0.9954 | 0.9197 (0.9348) | 2895 | 153.6 s, 1 core | 6.11 GB |

## Archives

- `calicost.tar.xz` (1.3 MB). CalicoST c1abcae ran its shipped
  `configuration_cna_multi` with `n_clones 5`. The archive holds:
  - `clone_labels.tsv`, `cnv_seglevel.tsv` and `cnv_perstate.tsv`;
  - `rdrbaf_final_nstates7_smp.npz` (`pred_cnv`, state parameters,
    `log_gamma`, `total_llf`);
  - `mergedallspots_nstates7_sp.npz`, the BAF-stage clones;
  - `normal_candidate_barcodes.txt` and `posterior_clone_probability.npy`;
  - `calicost_config.txt` and `input_filelist.tsv`, with the paths of the
    host that ran them;
  - `run.json`: cores, wall per resumed segment, peak, and the time per stage.
- `port.tar.xz` (0.6 MB). port 95940e5 ran `--sal` with
  `merge_agreement 0.99`. The archive holds:
  - `clone_labels.tsv`, `clone_labels_integer.tsv` and `baf_clone_labels.tsv`;
  - `cnv_seglevel.tsv`, `cnv_perstate.tsv`, `cnv_states.tsv` and
    `copy_decode.tsv`;
  - `rdrbaf_final_nstates7_smp.npz`, `manifest.json` and `run.json`.
- `truth.tar.xz` (35 KB): the realization's `truth_*` files and
  `manifest.json`. `digest.json` names its `realization_hash`, 3381575a.
  A redraw of `sim/manifests/baseline/dev_tree.toml` r0 that hashes otherwise is not
  this realization.

Left out, because they are recomputable or superseded:
- CalicoST's parsed inputs (510 MB);
- its per-round checkpoints;
- its gene-level tables and the fixed-ploidy tables;
- plots.

## Use

```
mkdir out && for t in calicost port; do tar -xJf $t.tar.xz -C out; done
python -m port.sim.draw sim/manifests/baseline/dev_tree.toml --into DIR   # the counts, if a metric needs them
```

`tests.sim_audit.score(load_simulated(DIR/dev_tree/r0), out/<tool>, ...)`
gives the row above. Scoring needs the drawn sample, because it reads the
barcodes and planted copies from it. A metric that reads only the truth
files can instead use `truth.tar.xz`.
