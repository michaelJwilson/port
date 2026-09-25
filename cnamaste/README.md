# cnamaste

`cnaster` at `4adad4d`, the commit `port`'s lockfile pins, renamed `cnamaste`
and vendored so that `port`'s patches can be written into it rather than
rebound at run time (#392). `VENDOR.toml` records the source, the rename, the
modules left out and the ones a stage has rewritten. MIT, as stated by the
author on #392.

`run_cnamaste <config.yaml>` is `run_cnaster` under the new name.
