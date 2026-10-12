"""combined.png drawn against what it draws: each panel's artists read back and compared with their source (T- #836 F1-F3).

`test_smoke.py` checks the page's geometry. Here the drawing is decoded and judged against what the run wrote and,
for copy numbers, the planted truth, so a panel that draws the wrong segment, state, boundary or spot fails:

- **Profile:** each rectangle's extent, fill and hatch read back to per-bin `(A, B)` per clone: `cnv_seglevel.tsv`'s.
- **Tracks:** the fitted level lines are the runs of each clone's decoded state at `exp(log_mu)` and `p_binom`; the
  state key's shares are `cnv_seglevel.tsv`'s (major, minor) shares.
- **Axis:** every genomic panel's chromosome boundaries are the lineage's re-binned level's, and each name sits
  inside its chromosome.
- **Truth:** the drawn `(A, B)` score against the planted profile exactly as `cnv_seglevel.tsv` does, and as the
  ledger's row for the fixture prints.
- **Clones:** each spot drawn in a clone's colour is a spot `clone_labels.tsv` gives that clone.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from cnamaste import ledger
from cnamaste.audit.scoring import at_truth, copy_rows
from cnamaste.layout import plot_combined_page
from cnamaste.plot_copy_number_profile import Hatch, hatched_palette, swatch
from cnamaste.utils import cast_clone_label

from test_smoke import figure_inputs


def page(ctx: Any) -> tuple[dict[str, Any], Any]:
    """The staged run's figure inputs and its combined page, drawn once per session."""
    d = figure_inputs(ctx)

    def make() -> Any:
        assignment = pd.Series([f"clone {x}" for x in np.asarray(d["res"]["new_assignment"])])
        return plot_combined_page(d["lengths"], d["X"], d["base"], d["total"], d["seg"], d["res"], d["coords"], assignment,
                                  sample_list=d["sample_list"], sample_ids=d["sample_ids"])  # fmt: skip

    return d, ctx.once("pages/combined", make)


def clone_ids(d: dict[str, Any]) -> np.ndarray:
    return np.unique(np.asarray(d["res"]["new_assignment"]))


def genomic(fig: Any, d: dict[str, Any]) -> tuple[Any, list[Any]]:
    """The profile and the tracks, in drawing order: every axes on bins `[0, n_bins)`."""
    axes = [ax for ax in fig.get_axes() if ax.get_xlim() == (0.0, float(len(d["seg"])))]
    assert len(axes) == 1 + 2 * clone_ids(d).size, f"{len(axes)} genomic axes"
    return axes[0], axes[1:]


def drawn_profile(fig: Any, d: dict[str, Any]) -> dict[int, np.ndarray]:
    """Per clone id, the (bins, 2) `(A, B)` the profile draws, each bin read from the one fill that covers it."""
    profile, _ = genomic(fig, d)
    ids = clone_ids(d)
    style = hatched_palette()
    copies = {tuple(np.round(swatch(style, c), 9)): c for c in range(8)}
    h = 1.0 / ids.size
    out = {int(c): np.full((len(d["seg"]), 2), -1) for c in ids}
    hatches = {id(c.fill): c for c in profile.collections if isinstance(c, Hatch)}
    for patch in profile.patches:
        if not isinstance(patch, Rectangle) or patch.get_facecolor()[3] == 0.0:
            continue
        k = int(np.floor((1.0 - patch.get_y()) / h))
        x0, x1 = int(round(patch.get_x())), int(round(patch.get_x() + patch.get_width()))
        a = copies[tuple(np.round(patch.get_facecolor()[:3], 9))]
        hatch = hatches.get(id(patch))
        if hatch is None:
            assert a == 1, f"an unhatched fill of copy {a}: only the normal (1, 1) is plain"
            b = 1
        else:
            b = copies[tuple(np.round(hatch.get_edgecolor()[0][:3], 9))]
            assert hatch.orientation == (1 if a >= b else -1), f"hatch direction at bins {x0}-{x1}"
        row = out[int(ids[k])]
        assert np.all(row[x0:x1] == -1), f"bins {x0}-{x1} of clone {ids[k]} drawn twice"
        row[x0:x1] = (a, b)
    return out


def _runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end, state) of each run of one state."""
    edges = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, states.size]
    return [(int(s), int(e), int(states[s])) for s, e in zip(edges[:-1], edges[1:])]


def _levels(ax: Any) -> list[np.ndarray]:
    """The fitted level lines on a track: its zorder-2 line collections, solid first."""
    return [np.asarray(c.get_segments()) for c in ax.collections if isinstance(c, LineCollection) and c.get_zorder() == 2]


def _boundaries(ax: Any) -> list[float]:
    """The x of each full-height vertical line (`axvline`) on `ax`."""
    return sorted(float(l.get_xdata()[0]) for l in ax.get_lines()
                  if isinstance(l, Line2D) and len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1] and list(l.get_ydata()) == [0, 1])  # fmt: skip


def test_the_profile_draws_the_seglevel_copies(ctx: Any) -> None:
    """Every bin of every clone drawn once, its `(A, B)` `cnv_seglevel.tsv`'s (7+ drawn as 7).

    Fails where a fill is misplaced, mis-coloured, unhatched or hatched the wrong way, or a clone's row is another's."""
    d, fig = page(ctx)
    drawn = drawn_profile(fig, d)
    for cid, row in drawn.items():
        want = np.minimum(d["seg"][[f"clone{cid} A", f"clone{cid} B"]].fillna(1).to_numpy(dtype=int), 7)
        assert np.all(row >= 0), f"clone {cid}: {(row < 0).any(axis=1).sum()} bins undrawn"
        assert np.array_equal(row, want), f"clone {cid}: {(row != want).any(axis=1).sum()} bins differ"


def test_the_tracks_draw_the_fit(ctx: Any) -> None:
    """Each clone's level lines are the runs of its decoded state, at `exp(log_mu)` (RDR) and `p_binom` and
    `1 - p_binom` (BAF); its state key's shares are `cnv_seglevel.tsv`'s (major, minor) shares, to 0.1%.

    Fails where a track draws another clone's states, another state's level, or keys shares the table does not hold."""
    d, fig = page(ctx)
    _, tracks = genomic(fig, d)
    res, seg = d["res"], d["seg"]
    log_mu, p = np.asarray(res["new_log_mu"]), np.asarray(res["new_p_binom"])
    pred, n_obs = np.asarray(res["pred_cnv"]), len(seg)
    for k, cid in enumerate(clone_ids(d)):
        states = (pred[:, k] if pred.ndim == 2 and pred.shape[1] > 1 else pred.reshape(-1)[k * n_obs:(k + 1) * n_obs]) % log_mu.size
        runs = _runs(states)
        rdr, baf = _levels(tracks[2 * k]), _levels(tracks[2 * k + 1])
        assert np.array_equal(rdr[0], [[(s, np.exp(log_mu[l])), (e, np.exp(log_mu[l]))] for s, e, l in runs]), f"clone {cid}: RDR levels"
        assert np.array_equal(baf[0], [[(s, p[l]), (e, p[l])] for s, e, l in runs]), f"clone {cid}: BAF levels"
        assert np.array_equal(baf[1], [[(s, 1.0 - p[l]), (e, 1.0 - p[l])] for s, e, l in runs]), f"clone {cid}: BAF mirror levels"
        a, b = seg[f"clone{cid} A"].to_numpy(), seg[f"clone{cid} B"].to_numpy()
        pairs = pd.Series(list(zip(np.maximum(a, b), np.minimum(a, b))))
        want = {f"{100.0 * share:.1f}% ({int(m)}, {int(n)})" for (m, n), share in pairs.value_counts(normalize=True).items()}
        keyed = {t.get_text() for t in tracks[2 * k].get_legend().get_texts()}
        assert keyed == want, f"clone {cid}: keyed {sorted(keyed)}, the table's {sorted(want)}"


def test_the_genomic_axis_is_the_lineages(ctx: Any) -> None:
    """Every genomic panel's chromosome boundaries are where the lineage's re-binned level changes contig, and each
    chromosome name sits inside its chromosome.

    Fails where the figure's bins are not the run's re-binned bins, or a name labels a neighbouring chromosome."""
    d, fig = page(ctx)
    profile, tracks = genomic(fig, d)
    contig = np.asarray(ctx.request.getfixturevalue("lineage").levels["rebinned"].contig)
    assert contig.size == len(d["seg"]), "one re-binned bin per cnv_seglevel.tsv row"
    starts = [float(x) for x in np.flatnonzero(contig[1:] != contig[:-1]) + 1]
    for ax in [profile, *tracks]:
        assert _boundaries(ax) == starts, "chromosome boundaries"
    names = [t for t in tracks[-1].texts if t.get_text().isdigit()]
    assert names, "the chromosomes are named"
    for t in names:
        assert str(contig[int(np.floor(t.get_position()[0]))]) == t.get_text(), f"chr {t.get_text()} placed off its chromosome"


@pytest.mark.end2end
def test_the_drawn_copies_score_as_the_ledger(ctx: Any) -> None:
    """The drawn `(A, B)`, at the matched planted clones, score as `cnv_seglevel.tsv` does: the phase-free copy ARI
    and exact altered rate equal, and equal to the ledger's latest row for the fixture as it prints them.

    Fails where the profile shows copies other than the ones the run is scored on."""
    d, fig = page(ctx)
    truth = ctx.request.getfixturevalue("truth")
    barcodes = np.asarray(ctx.sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    fitted = at_truth(barcodes, np.asarray(d["res"]["new_assignment"]), truth)
    seg = d["seg"].copy()
    for cid, row in drawn_profile(fig, d).items():
        seg[f"clone{cid} A"], seg[f"clone{cid} B"] = row[:, 0], row[:, 1]
    drawn, table = copy_rows(fitted, seg, truth), copy_rows(fitted, d["seg"], truth)
    print(f"drawn {drawn}; table {table}")
    assert drawn == table
    row = ledger.latest(ledger.read())[ctx.sim.config["fixture_hash"]]
    assert {k: f"{v:.4f}" for k, v in drawn.items()} == {k: row[k] for k in drawn}, "the ledger's row"


def test_the_clones_panel_draws_the_written_labels(ctx: Any) -> None:
    """Each spot drawn in a clone's keyed colour is one `clone_labels.tsv` gives that clone, every spot once.

    Fails where a spot is drawn in another clone's colour or at another position, or the key names a colour wrongly."""
    d, fig = page(ctx)
    clones_ax = [ax for ax in fig.get_axes() if ax.get_aspect() == 1.0][-1]
    key = clones_ax.get_legend()
    name_of = {tuple(np.round(to_rgb(h.get_markerfacecolor()), 6)): t.get_text() for h, t in zip(key.legend_handles, key.get_texts())}
    drawn: dict[str, set[tuple[float, float]]] = {}
    for c in clones_ax.collections:
        if not isinstance(c, PathCollection):
            continue
        colours = np.unique(np.round(c.get_facecolor()[:, :3], 6), axis=0)
        assert len(colours) == 1, "one colour per clone's scatter"
        spots = {(float(x), float(-y)) for x, y in c.get_offsets()}
        drawn.setdefault(name_of[tuple(colours[0])], set()).update(spots)
    labels = pd.read_csv(io.BytesIO(ctx.sim.file("clone_labels.tsv")), sep="\t")
    want = {cast_clone_label(f"clone {k}"): {(float(x), float(y)) for x, y in g[["x", "y"]].to_numpy()} for k, g in labels.groupby("clone_label")}
    assert sum(len(s) for s in drawn.values()) == len(labels), "every spot drawn once"
    assert drawn == want
