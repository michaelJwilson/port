# CalicoST on CalicoST easy (`2d4ce9a9`), uncapped in practice (#494, #532)

**TL;DR:** the outputs CalicoST wrote on the sample its own paper ships, run
to completion in 6,497 s under a 21,600 s cap, so any CNA-recovery metric can
be computed without a rerun. Scoring `calicost.tar.xz` with
`port.qa.audit.score_sample` reproduces `docs/calicost-benchmark.md`'s row:

| clone ARI | integer clone ARI | state ARI | copy ARI | copy ARI `_pf` | exact | exact altered (`_pf`) | bins | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5303 (4) | 0.5303 (4) | 0.0224 | 0.6098 | 0.6094 | 0.9758 | 0.2326 (0.4651) | 1476 | 6,497 s, 4 cores | 2.79 GB |

## Archive

`calicost.tar.xz` (1.9 MB). CalicoST c1abcae ran its shipped
`configuration_cna`, every value its own, among them `n_clones 3` against the
4 clones the sample plants. The archive holds:

- `clone_labels.tsv`, `cnv_seglevel.tsv` and `cnv_perstate.tsv`;
- `rdrbaf_final_nstates7_smp.npz` (`pred_cnv`, state parameters, `log_gamma`,
  `total_llf`);
- `mergedallspots_nstates7_sp.npz`, the BAF-stage clones after the
  Neyman-Pearson merge, and `allspots_nstates7_sp.npz`, before it;
- `normal_candidate_barcodes.txt` and `posterior_clone_probability.npy`;
- `calicost_config.txt`, with the paths of the host that ran it;
- `run.json`: the tool string, cap, wall, peak, cores and the scored row.

Left out, because they are recomputable or superseded: the parsed inputs,
`binned_data.npz` (142 MB), the per-clone read-depth checkpoints, the
gene-level and fixed-ploidy tables, and plots.

## Use

```
mkdir out && tar -xJf calicost.tar.xz -C out
```

`port.qa.audit.score_sample(load_simulated('numcnas1.2_cnasize5e7_ploidy2_random0'), out, 'calicost', 6497.0)`
gives the row above. Scoring needs the sample, because it reads the barcodes
and planted copies from it; `sim/numcnas1.2_cnasize5e7_ploidy2_random0` is
committed, so no redraw is needed.
