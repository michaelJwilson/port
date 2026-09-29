"""Studies: measurements run by hand, outside the suite (#489, #492).

A study drives `snakes_and_ladders`' own machinery -- its solvers, its starts,
its benchmark seam -- to measure it on problems port captured. It referees
nothing about `cnaster`, so its `sal` imports are not the oracle surface
`tests/test_coverage_scope.py` counts, which scans the suite's modules at the
top of `tests/` and not this package. Each module states what it measured and
the document its numbers are in.
"""
