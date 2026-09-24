"""The clone-mixture design study (#380): the variants set aside, and its probes.

`variants` is the mixture with every option the study measured -- continuous
or integer profiles, BIC or entropy rows, a fixed or annealed mixing cap --
and the numbers that decided against each. `port.extensions.clone_mixture`
ships the one adopted.

`probes/run_sim.sh` runs one simulated sample through `tests.sim_audit` with
`--sal`, with or without `--clone-mixture`, and prints the fitted `W` and
the integer decode's tumour fractions; its header lists the options.
"""
