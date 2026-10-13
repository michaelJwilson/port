"""Studies: measurements run by hand, outside the suite (#489, #492).

A study drives `snakes_and_ladders`' own machinery -- its solvers, its starts,
its benchmark seam -- or `port`'s set-aside code to measure it on problems
port captured or drew. It referees nothing about `cnaster`, so its `sal`
imports are not the oracle surface `tests/test_coverage_scope.py` counts,
which scans the suite's modules and not the package. Each module states what
it measured and the document its numbers are in; `run_study --<study>` runs
the ones with a `main` (`port.qa.scripts.run_study.STUDIES`), and `stream`
holds the harness the two solver streams share. Moved from `tests/studies/`
(T- #673 G5).
"""
