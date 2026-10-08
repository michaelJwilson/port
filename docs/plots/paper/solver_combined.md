# solver_combined.png

Drawn at code `009cf46` from two records on `sim/manifests/study15.toml`, each problem built by
`run_cnaster_port --sal` at the planted clones (`port.studies.stage`, #730, #742):

| Panel | Record | Data hash | Problems | Starts | Samplers |
| --- | --- | --- | --- | --- | --- |
| (a) spatial solvers | `port.studies.potts_stream` | `98b6a6aa` | 10 | 1 | settings file |
| (b) copy-state starts | `port.studies.copy_state_stream` | `dbdc811c` | 10 | 1 | settings file |

Each sampler reads its settings file (`potts_sampler_settings.json`, `copy_sampler_settings.json`),
calibrated by `run_calibrate` on realizations 0-4, held out from these
panels (T- #777). (a)'s annealed samplers, Wolff included, take the schedule `sal`'s `tune_schedule`
chose from its declared ramps, raced from common random numbers and ranked after ICM. (b)'s samplers
sample the run's own Baum-Welch objective, and their
settings were ranked by the log-likelihood after it.

**The two y axes are different quantities.** (a): a run's Potts energy less TRW-S's lower bound on
its problem (nats of the clone-assignment field). (b): the log-likelihood of the run's Baum-Welch at
a start, less the best any run or the planted states reached. **Each key's two numbers** are the
percent of spots (a) or rows (b) unlike the planted ones: before and after, where after is ICM
then the colour merge in (a), and Baum-Welch in (b).
