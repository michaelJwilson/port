"""Regenerate the committed figures under `docs/plots/`.

Run as `python -m tests.generate_plots`. It drives the same round trip
`tests/test_run_cnaster_round_trip.py` drives, on the dev instance, and copies
what `run_cnaster` wrote into the repository.

**The figures are committed because a plot is a result.** #87 asks for them
beside the code rather than in a run directory nobody keeps, so a change to
the pipeline shows up as a change to a picture in a diff.

They are not a referee. Nothing here compares a figure against a previous
one, and a matplotlib PDF carries a creation timestamp, so two runs of this
script differ byte for byte with nothing having changed. Byte reproduction of
figures is the standard `CLAUDE.md` names and reaching it needs the timestamp
pinned first; #103 owns that.
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from tests.fixtures import dev_instance
from tests.test_run_cnaster_round_trip import _run

PLOTS = Path(__file__).resolve().parent.parent / "docs" / "plots"
"""Where the figures live in the repository."""

STATES = 5
"""What the run fits, against the ten the dev instance plants.

Five because ten does not fit: the kernel kills the run at 15 GB (#90). At
five it completes in 31 s at a peak of 5.89 GB.
"""


def main() -> None:
    """Run the pipeline and copy its figures into `docs/plots/`."""
    root = Path(tempfile.mkdtemp())
    output = _run(dev_instance(), root, max_iter_outer=1, max_iter=3, n_states=STATES)

    PLOTS.mkdir(parents=True, exist_ok=True)
    for stale in PLOTS.glob("*.pdf"):
        stale.unlink()

    figures = sorted(output.rglob("*.pdf"))
    for figure in figures:
        shutil.copy(figure, PLOTS / figure.name)

    print(f"wrote {len(figures)} figures to {PLOTS}")


if __name__ == "__main__":
    main()
