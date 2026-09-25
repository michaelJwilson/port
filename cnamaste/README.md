# cnamaste

`cnaster` at `4adad4d`, the commit `port`'s lockfile pins, renamed `cnamaste`
and vendored so that `port`'s patches can be written into it rather than
rebound at run time (#392). `VENDOR.toml` records the source, the rename, the
modules left out and the ones a stage has rewritten. MIT, as stated by the
author on #392.

`run_cnamaste <config.yaml>` is `run_cnaster` under the new name.

## Tests

`pytest cnamaste/tests` (its own configuration, `pyproject.toml`). The tests
know nothing of `cnaster` or `snakes_and_ladders`: `tests/sim` plants an
instance with `numpy` alone, writes it as the files `run_cnamaste` reads, and
reads the run's tables back against what it planted.

| File | Referee |
| --- | --- |
| `test_sim.py` | the generator against its own planted moments |
| `test_emissions.py` | `scipy.stats`, and normalization |
| `test_lattice.py` | brute-force enumeration of every path |
| `test_unsegment_round_trip.py`, `test_tmp_input_round_trip.py` | the planted bins, bitwise, back through the loader and the binner |
| `test_clone_assignment.py` | brute-force distances and `scipy.stats`; the planted labels; the solver's local optimality |
| `test_figures.py` | the data drawn against the data given; a written figure's bytes against its content |
| `test_shift.py` | brute-force `logsumexp` and the pin's loop; scale invariance and normalization |
| `test_run.py` | the planted clones, states and allele fractions, through a whole run |
| `test_benchmarks.py` | a baseline; `--benchmark-enable -m "benchmark or release"` measures |
