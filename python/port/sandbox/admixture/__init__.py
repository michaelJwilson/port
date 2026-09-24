"""The clone-mixture design study (#380): probes, not package code.

`probes/run_sim.sh <easy|hard> <base|mix> [pure|admixed] [oracle]` runs one
simulated sample through `tests.sim_audit` with `--sal`, optionally
`--clone-mixture`; `NF=f1,f2,f3` plants a per-clone normal fraction
(`purify(normal=...)`), and `MIX_ROW_MODE`, `MIX_STARTS` set the row
regularizer and admixture starts. `probes/stop_probe.py` stops after
`cnaster`'s normal-candidate step to score the BAF-only stage.
"""
