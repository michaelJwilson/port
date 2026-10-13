"""`python -m port.sandbox.np_merge [run_cnaster_port arguments]`: the entry point with the merge installed (#497).

Ticket: #497 -- the `run_cnaster_port` entry point with the Neyman-Pearson
  merge installed, set aside with the merge at the owner's request.
Measurement: none recorded; see `port.sandbox.np_merge.merge`.
Exit: graduates or retires with `port.sandbox.np_merge.merge`.
"""

import sys

from port.sandbox.np_merge import np_merge
from port.scripts.run_cnaster import main

with np_merge():
    status = main(sys.argv[1:])

raise SystemExit(status)
