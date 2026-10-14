# CalicoST on CalicoST hard (`8797710b`), uncapped in practice (#494, #532)

**TL;DR:** the outputs CalicoST wrote on the harder of the two samples its own
paper ships, run to completion in 4,474 s under a 21,600 s cap, so any
CNA-recovery metric can be computed without a rerun. Scoring
`calicost.tar.xz` with `port.qa.audit.score_sample` reproduces
`docs/calicost-benchmark.md`'s row:

| clone ARI | integer clone ARI | state ARI | copy ARI | copy ARI `_pf` | exact | exact altered (`_pf`) | bins | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0690 (2) | 0.0690 (2) | 0.0132 | 0.6285 | 0.6285 | 0.9905 | 0.0000 (0.5500) | 1467 | 4,474 s, 4 cores | 2.78 GB |

**It fits 2 clones against the 4 the sample plants**, so two planted clones
have no fitted counterpart and the copy metrics are taken over the two that
do. `exact_altered` is 0.0: no altered bin is decoded as its planted pair in
phase. `exact_neutral` is 0.9973, so the neutral genome is fine and the
aberrations are what it misses.

## Archive

`calicost.tar.xz` (2.1 MB). CalicoST c1abcae ran its shipped
`configuration_cna`, every value its own, among them `n_clones 3`. The
archive holds:

- `clone_labels.tsv`, `cnv_seglevel.tsv` and `cnv_perstate.tsv`;
- `rdrbaf_final_nstates7_smp.npz` (`pred_cnv`, state parameters, `log_gamma`,
  `total_llf`);
- `mergedallspots_nstates7_sp.npz`, the BAF-stage clones after the
  Neyman-Pearson merge, and `allspots_nstates7_sp.npz`, before it;
- `normal_candidate_barcodes.txt` and `posterior_clone_probability.npy`;
- `calicost_config.txt`, with the paths of the host that ran it;
- `run.json`: the tool string, cap, wall, peak, cores and the scored row.

Left out, because they are recomputable or superseded: the parsed inputs,
`binned_data.npz`, the per-clone read-depth checkpoints, the gene-level and
fixed-ploidy tables, and plots.

## Use

```
mkdir out && tar -xJf calicost.tar.xz -C out
```

`port.qa.audit.score_sample(load_simulated('numcnas6.3_cnasize1e7_ploidy2_random0'), out, 'calicost', 4473.9)`
gives the row above. Scoring needs the sample, because it reads the barcodes
and planted copies from it; `sim/numcnas6.3_cnasize1e7_ploidy2_random0` is
committed, so no redraw is needed.
