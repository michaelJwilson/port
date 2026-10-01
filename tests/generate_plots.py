"""Regenerate the dev instance's figures, by default under `.cache/plots/`.

Run as `python -m tests.generate_plots [--out DIR]`. It writes the dev
instance's inputs through `tests.run_config.run_written`, runs
**`run_cnaster_port`** on them, and copies what it wrote into `DIR`
(default `tests.plots_dir.PLOTS`, untracked). `--cnaster` runs plain
`cnaster` instead, for a comparison.

**Two sets.** `DIR` is the dev instance as planted by default; `DIR/lattice/`
is the same genome with `copy_lattice=True`, whose states are integer allele
copies `(A, B)` (`tests.fixtures.COPY_LATTICE`), so its copy-number figures
can be read against a truth that is integer (#313).

**CI runs this on every pull request and uploads the result** as a workflow
artifact (`.github/workflows/figures.yml`), so the figures a pull request's
code draws can be read beside it. They are not committed: `docs/plots/`
tracks no PNG (`tests/test_ci_entry.py`).

They are not a referee. Nothing here compares a figure against a previous
one. **They are PNG** (#452): a matplotlib PDF carries a creation timestamp,
so two runs differ byte for byte with nothing having changed. The run still
writes its PDFs; `run_cnaster_port --png-copies` has `write_fig` write a PNG
beside each, without metadata, and those are what is copied. `--cnaster`
runs `cnaster`'s own `write_fig`, so it copies PDFs, for a local comparison.
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

from port.extensions.combined_figure import (
    Recorded,
    combined_figure,
    genomic_figure,
    page_style,
    recording,
    spatial_figure,
)

from tests.fixtures import COPY_LATTICE, CoreInferenceTruth, dev_instance
from tests.he_slide import mock_he, write_he_slide
from tests.plots_dir import PLOTS
from tests.run_config import run_written

STATES = 8
"""What the run fits: the eight states each instance uses of those it plants.

The dev instance plants ten states and its clones use eight; the copy-lattice
instance plants nine and uses eight. At five (#90's figure, set when ten
did not fit) the five planted amplifications could not be separated and
chr7's two events were decoded as one state (#313).
"""


def _write_combined(
    recorded: Recorded, truth: CoreInferenceTruth, root: Path, output: Path
) -> None:
    """The genomic and spatial figures, and both on one page (#309, #339).

    The slide is mocked from the planted labels and read back through
    `cnaster.he.get_he_image`, as `run_cnaster` reads one. It is written
    beside the run's inputs rather than into them: `load_input_data` would
    otherwise find it and refine the initial clones by it, and the figures
    would stop being the ones the dev instance's run draws.
    """
    from cnaster.he import get_he_image
    from port.patch.utils import write_fig
    from port.pipeline import FIGURE_DPI

    # NB what the run's `FIGURE_SWAPS` row and `--png-copies` bind (#517).
    PAGE: dict[str, Any] = {
        "bbox_inches": None,
        "dpi": FIGURE_DPI,
        "group_rasters": True,
        "png_copy": True,
    }

    slide = mock_he(truth.labels, truth.lattice, seed=truth.seed)
    write_he_slide(slide, root / "slide")
    frame = get_he_image(str(root / "slide"), res="hires", pos=None)

    plots = next(output.rglob("clones_spatial.pdf")).parent
    # NB at its declared size, not a tight box: the page is drawn at the text
    #    width and included at 1:1, so a box that grows past it is rescaled.
    # NB written as drawn: the run has set seaborn's theme, which a page
    #    written under it would follow where a style is read at draw time.
    with page_style():
        write_fig(str(plots / "genomic.pdf"), genomic_figure(recorded), **PAGE)
        write_fig(
            str(plots / "spatial.pdf"),
            spatial_figure(recorded, frame),
            **PAGE,
        )
        write_fig(
            str(plots / "combined.pdf"),
            combined_figure(recorded, frame),
            **PAGE,
        )


def main() -> None:
    """Run the pipeline and copy its figures into `--out`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cnaster",
        action="store_true",
        help="run plain cnaster rather than run_cnaster_port",
    )
    parser.add_argument(
        "--out", type=Path, default=PLOTS, help="where the figures go (untracked)"
    )
    arguments = parser.parse_args()

    sets = (
        (dev_instance(), arguments.out),
        (
            dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True),
            arguments.out / "lattice",
        ),
    )

    for truth, destination in sets:
        root = Path(tempfile.mkdtemp())

        if arguments.cnaster:
            output = run_written(
                truth, root, port=False, max_iter_outer=1, max_iter=3, n_states=STATES
            )
        else:
            # NB in process, through `port.scripts.run_cnaster.main`, with its
            #    defaults: the figures are the ones a user of the entry point gets.
            with recording() as recorded:
                output = run_written(
                    truth,
                    root,
                    port=True,
                    max_iter_outer=1,
                    max_iter=3,
                    n_states=STATES,
                    flags=("--png-copies",),
                )

            _write_combined(recorded, truth, root, output)

        destination.mkdir(parents=True, exist_ok=True)
        # NB PDFs are not kept (#452). PNGs are overwritten by name rather
        #    than globbed away: `realizations.png` and `umi_grow_*.png` beside
        #    them are written by other scripts.
        for stale in destination.glob("*.pdf"):
            stale.unlink()

        suffix = "*.pdf" if arguments.cnaster else "*.png"
        figures = sorted(output.rglob(suffix))
        for figure in figures:
            shutil.copy(figure, destination / figure.name)

        print(f"wrote {len(figures)} figures to {destination}")


if __name__ == "__main__":
    main()
