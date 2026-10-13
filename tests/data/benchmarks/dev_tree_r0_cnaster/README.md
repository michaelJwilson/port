# cnaster at its pin on `dev_tree` 60 × 50 r0 (`3381575a`) (T- #833)

**TL;DR:** the outputs of `cnaster` 4adad4d, run with nothing of port's rebound
except #105's row, on the CalicoST benchmark realization. Scored by
`port.studies.benchmark_table.cnaster_row`, they give:

| tool | clone ARI | integer clone ARI | state ARI | copy ARI (phase-free) | exact altered (phase-free) | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnaster 4adad4d | 0.9599 (5) | 0.9599 (5) | 0.0871 | 0.9716 (0.9717) | 0.7324 (0.7324) | 749.1 s, 4 cores | 6.5 GB |

## The run

`port.qa.cnaster_arm`: `run_audit --sim -- --no-patch --no-plots` at port
452bc5f, with one `SWAPS` row installed, `normal_baf_bin_filter`. Pure
`cnaster` at the pin removes bins in its normal-BAF filter and then ends in an
`IndexError` in the gene-level writer, before `clone_labels.tsv` and
`cnv_seglevel.tsv` are written (#105). The row marks the removed bins' genes
`is_interval = False` and otherwise leaves the frame as `cnaster`'s. The
config is the one `port.qa.audit.drawn_config` writes for the draw.
`--no-plots` turns off `write_fig` alone. The wall is the
whole `run_audit` process, as port's row in the table is timed.

## Archive

`cnaster.tar.xz` (0.6 MB), under `cnaster/`:
- `clone_labels.tsv`, `baf_clone_labels.tsv`, `cnv_seglevel.tsv` and
  `cnv_perstate.tsv`;
- `rdrbaf_final_nstates7_smp.npz`;
- `config.yaml`, with the run's scratch root replaced by `SCRATCH`;
- `run.json`: the commit, cores, wall, peak and the driver.

The archive leaves out the gene-level table and the plots. The truth is
`../dev_tree_r0/truth.tar.xz`'s.

## Use

`run_study --benchmark-table --cnaster` adds the `\cnaster{}` row and writes
`docs/benchmark_cnaster.tex`. Without `--cnaster` the table and its file are
unchanged.
