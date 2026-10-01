"""`python -m port.sandbox.split_state [run_cnaster_port arguments]`: the entry point with the split installed (#481).

Ticket: #481 -- the `run_cnaster_port` entry point with one state split by
  depth and refitted, set aside with the split.
Measurement: see `port.sandbox.split_state.split`.
Exit: graduates or retires with `port.sandbox.split_state.split`.
"""

import sys

from port.sandbox.split_state import split_state
from port.scripts.run_cnaster import main

with split_state():
    status = main(sys.argv[1:])

raise SystemExit(status)
