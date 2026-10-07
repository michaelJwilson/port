# solver_combined.png

Drawn at code `335971f+` from two records on `sim/manifests/dev_tree_1s_hard.toml`, each problem built by
`run_cnaster_port --sal` at the planted clones (`port.studies.stage`, #730, #742):

| Panel | Record | Data hash | Problems | Starts | Samplers |
| --- | --- | --- | --- | --- | --- |
| (a) spatial solvers | `port.studies.potts_stream` | `40f0c8a2` | 5 | 5 | settings file |
| (b) copy-state starts | `port.studies.copy_state_stream` | `62375f46` | 5 | 5 | settings file |

Where a sampler reads a settings file (`potts_sampler_settings.json`, `copy_sampler_settings.json`),
it was tuned before the harness, on problems the run does not solve: these panels are
untuned for the run's problems until #723 retunes them on a quiet host.
