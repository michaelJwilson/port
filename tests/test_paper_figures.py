"""`docs/plots/paper/` and the figures `port.studies.paper_figures` adds (#624)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from port.studies.paper_figures import KEY_STUDIES, OUT, QUESTIONS, Compared

from tests import ROOT


@pytest.mark.infra
def test_the_paper_readme_lists_exactly_the_committed_files() -> None:
    """`docs/plots/paper/README.md`'s table names every tracked file under
    `docs/plots/paper/` but itself, once, and `QUESTIONS` with `KEY_STUDIES` names the same."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", OUT.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    prefix = OUT.relative_to(ROOT).as_posix() + "/"
    committed = sorted(p.removeprefix(prefix) for p in tracked)
    committed.remove("README.md")
    listed = re.findall(r"^\| `([^`]+)` \|", (OUT / "README.md").read_text(), re.M)

    assert len(listed) == len(set(listed))
    assert sorted(listed) == committed
    assert sorted(QUESTIONS | KEY_STUDIES) == committed


def _tiny() -> Compared:
    """3 planted clones on a 4 x 6 hex lattice, and 6 bins on 2 chromosomes.

    Clone 1 plants (2, 0) over bins 0-2, decoded (0, 2) on bins 0-1 (swapped)
    and (1, 1) on bin 2 (wrong); clone 2 plants (2, 2) over bins 3-4, decoded
    exactly; every other bin is (1, 1), decoded exactly.
    """
    rows, columns = np.indices((4, 6))
    coords = np.stack([rows.ravel(), 2 * columns.ravel() + rows.ravel() % 2], 1)
    planted = np.repeat([0, 1, 2], 8)
    fitted = planted.copy()
    fitted[0] = -1
    fitted[[2, 9]] = 3
    truth = np.ones((6, 3, 2), dtype=np.int64)
    truth[0:3, 1] = (2, 0)
    truth[3:5, 2] = (2, 2)
    decoded = truth.copy()
    decoded[0:2, 1] = (0, 2)
    decoded[2, 1] = (1, 1)
    edges = np.array([0.0, 30.0, 60.0])
    start = np.arange(6) * 10.0
    return Compared(
        coords=coords.astype(np.float64),
        planted=planted,
        fitted=fitted,
        names=("$m_N$", "$m_1$", "$m_2$"),
        clone_of={0: 0, 1: 1, 2: 2},
        ari=0.8125,
        start=start,
        end=start + 10.0,
        edges=edges,
        truth=truth,
        decoded=decoded,
    )


def _texts(figure: Any) -> list[str]:
    return [t.get_text() for t in figure.findobj(lambda a: hasattr(a, "get_text"))]


@pytest.mark.snapshot
def test_the_compare_figures_draw_what_their_inputs_hold(tmp_path: Path) -> None:
    """Figures 14-17 on a tiny truth and fit, no pipeline run: the matching in
    14 and no title (T- #660: the ARI is the README's), the confusion's shares in 15, one mark per swapped or
    wrong bin in 16, the per-class shares in 17, each written unstamped (#743)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.studies import paper_figures as pf

    c = _tiny()

    labels = pf.labels_figure(c)
    assert labels._suptitle is None
    texts = _texts(labels)
    assert "$m_1$ $\\leftrightarrow$ fit 1" in texts
    assert "fit 3, unmatched" in texts
    plt.close(labels)

    confusion = pf.confusion_figure(c)
    cells = _texts(confusion.axes[0])
    # NB (1, 1): 2 * 6 + 4 + 1 bins decoded (1, 1), one of them planted (2, 0)
    assert sorted(t for t in cells if re.fullmatch(r"\d\.\d{3}", t)) == [
        "0.333", "0.667", "1.000", "1.000",
    ]  # fmt: skip
    plt.close(confusion)

    genomic = pf.genomic_compare_figure(c)
    marks = [
        line.get_xdata().size
        for ax in genomic.axes
        for line in ax.get_lines()
        if line.get_marker() == "|"
    ]
    assert sum(marks) == 3
    assert "2 swapped, 1 wrong of 6" in _texts(genomic.axes[1])
    plt.close(genomic)

    shares = pf.exact_by_class(c)
    assert shares["loh"] == (0.0, 2 / 3, 3)
    assert shares["balanced_gain"] == (1.0, 1.0, 2)
    assert np.isnan(shares["unbalanced_gain"][0])
    assert shares["neutral"] == (1.0, 1.0, 13)

    written = pf.compare_figures(c, tmp_path)
    assert [p.name for p in written] == list(pf.FIGURES)
    assert all(p.stat().st_size > 0 for p in written)


@pytest.mark.infra
def test_the_solver_panel_draws_only_the_tables_solvers() -> None:
    """T- #660: a stream holding a solver `potts_plot.TABLE` dropped draws
    none of its runs; the kept solver's runs are all drawn."""
    from port.studies import potts_plot

    rows = [
        {"problem": 0, "solver": solver, "seed": seed, "seconds": 1.0, "energy": 5.0,
         "polish_seconds": 0.1, "polished": 4.0, "both_seconds": 0.2, "both": 4.0}
        for solver in ("sal:icm", "port:icm") for seed in range(3)
    ]  # fmt: skip
    record = {
        "rows": rows,
        "done": [0],
        "problems": {0: {"truth_energy": 3.0, "bound": 1.0}},
    }

    drawn = potts_plot.frame(record)

    assert "port:icm" not in potts_plot.NUMBER
    assert sorted(set(drawn.solver)) == ["sal:icm"]
    assert len(drawn) == 3
