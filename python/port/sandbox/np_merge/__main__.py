"""`python -m port.sandbox.np_merge [run_cnaster_port arguments]`: the entry point with the merge installed (#497)."""

import sys

from port.sandbox.np_merge import np_merge
from port.scripts.run_cnaster import main

with np_merge():
    status = main(sys.argv[1:])

raise SystemExit(status)
