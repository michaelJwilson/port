"""Per-function rows for the outputs and `run_cnamaste`'s own glue between stages.

Inputs: the recorded stage inputs and outputs and the written files
(`sim.file`), so the glue `run_cnamaste` computes between two recorded calls is
recomputed from the first and compared with the second.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cnamaste.io import construct_df_clone_label
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix
from cnamaste.utils import write_tsv

from audit.fn import Replay, Row, run, table, unchanged

GLUE = "scripts.run_cnamaste:run_cnamaste"


def written(ctx: Any, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(ctx.sim.file(name)), sep="\t")


# --- oracle --------------------------------------------------------------------


def _labels(ctx: Any) -> dict[str, Any]:
    return {"in": ctx.sim.stored("09_outputs/construct_df_clone_label/in"), "out": ctx.sim.stored("09_outputs/construct_df_clone_label/out")}


def _clone_label(d: dict[str, Any]) -> None:
    barcodes, coords, labels, tumor = d["in"]["args"]
    frame = unchanged(construct_df_clone_label, barcodes, coords, labels, tumor)
    by_barcode = pd.Series(np.asarray(labels), index=np.asarray(barcodes))
    assert frame["clone_label"].to_numpy().tolist() == by_barcode[frame.index].tolist(), "each barcode keeps its label"
    key = list(zip(frame["sample_id"], frame["x"], frame["y"], strict=True))
    assert key == sorted(key), "sorted by (sample_id, x, y)"
    assert set(frame.index) == set(np.asarray(barcodes)) and "tumor_proportion" not in frame
    pd.testing.assert_frame_equal(frame, d["out"])


def _baf_profiles(ctx: Any) -> dict[str, Any]:
    return {"merged": ctx.sim.stored("05_baf/merge_by_minspots/out/1"), "profiles": ctx.sim.stored("06_normal/determine_normal_candidates/in/args/2"),
            "n_states": ctx.config.hmm.n_states}


def _profiles(d: dict[str, Any]) -> None:
    res, n_states = d["merged"], d["n_states"]
    pred = np.argmax(np.asarray(res["log_gamma"]), axis=0)
    n_clones = np.unique(np.asarray(res["new_assignment"])).size
    pred = pred.reshape(n_clones, -1)
    p = np.asarray(res["new_p_binom"])[pred % n_states, 0]
    assert np.array_equal(np.where(pred < n_states, p, 1 - p), d["profiles"]), "the merged BAF clones' model BAF per bin, as run_cnamaste builds it"


def _decoder_inputs(ctx: Any) -> dict[str, Any]:
    f = ctx("08_rdr/run_core_inference/in")
    return {"res": ctx.sim.result("08_rdr/reindex_clones/out/0"), "single_X": f["args"][0], "base": f["args"][2], "total": f["args"][3],
            "calls": [ctx.sim.stored(f"09_outputs/integer_copy_{k}/in")["args"] for k in range(4)]}


def _decoder_glue(d: dict[str, Any]) -> None:
    res = d["res"]
    labels = np.asarray(res["new_assignment"])
    ids = np.unique(labels)
    _, base, _, _ = merge_pseudobulk_by_index_mix(d["single_X"], d["base"], d["total"], [np.where(labels == c)[0] for c in ids], None)
    for s, args in enumerate(d["calls"]):
        log_mu, clone_base, p, pred = (np.asarray(a) for a in args)
        assert np.array_equal(log_mu, np.asarray(res["new_log_mu"])[:, 0]) and np.array_equal(p, np.asarray(res["new_p_binom"])[:, 0])
        assert np.array_equal(pred, np.asarray(res["pred_cnv"])[:, s]) and np.allclose(clone_base, base[:, s]), f"clone {s}'s decoder input"


def _seglevel(ctx: Any) -> dict[str, Any]:
    return {"seg": written(ctx, "cnv_seglevel.tsv"), "state": written(ctx, "cnv_perstate.tsv"), "res": ctx.sim.result("08_rdr/reindex_clones/out/0"),
            "copies": [np.asarray(ctx.sim.stored(f"09_outputs/integer_copy_{k}/out/0")) for k in range(4)],
            "bins": ctx.sim.stored("07_rebin/create_bin_ranges/out")}


def _written_copies(d: dict[str, Any]) -> None:
    seg, state, res = d["seg"], d["state"], d["res"]
    pred = np.asarray(res["pred_cnv"])
    for s, copies in enumerate(d["copies"]):
        assert np.array_equal(seg[f"clone{s} Z"], pred[:, s])
        assert np.array_equal(seg[f"clone{s} A"], copies[pred[:, s], 0]) and np.array_equal(seg[f"clone{s} B"], copies[pred[:, s], 1])
        assert np.allclose(seg[f"clone{s} logmu"], np.asarray(res["new_log_mu"])[pred[:, s], 0])
        assert np.array_equal(state[f"clone{s} A"], copies[:, 0]) and np.array_equal(state[f"clone{s} B"], copies[:, 1])
    bins = d["bins"][d["bins"]["bin_id"].notna()].groupby("bin_id")
    assert np.array_equal(seg["START"], bins["START"].first()) and np.array_equal(seg["END"], bins["END"].last())


ORACLE: list[Row] = table(
    "oracle",
    ("io:construct_df_clone_label", "09_outputs/construct_df_clone_label in and out", _labels, _clone_label, "one row per barcode with its label, sorted by (sample, x, y); no input mutation"),
    (GLUE, "05_baf/merge_by_minspots out, 06_normal/determine_normal_candidates in", _baf_profiles, _profiles, "the glue's merged BAF profiles are each clone's MAP state BAF, phase applied"),
    (GLUE, Replay("08_rdr/run_core_inference in, 08_rdr/reindex_clones out, 09_outputs/integer_copy in"), _decoder_inputs, _decoder_glue,
     "each decoder call gets its clone's log_mu, p, path and pseudobulk baseline"),
    (GLUE, "cnv_seglevel.tsv, cnv_perstate.tsv, decoder outs", _seglevel, _written_copies, "the written Z, A, B and log mu are the decoded copies along each clone's path"),
)


# --- invariants ---------------------------------------------------------------------


def _tsv(tmp: Any) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=["r0", "r1"])
    write_tsv(str(tmp / "t.tsv"), frame, index=True, index_label="row")
    back = pd.read_csv(tmp / "t.tsv", sep="\t", index_col="row")
    pd.testing.assert_frame_equal(back.rename_axis(None), frame)
    write_tsv(str(tmp / "e.tsv"))
    assert (tmp / "e.tsv").read_text().strip() == '""' or (tmp / "e.tsv").read_text().strip() == ""


INVARIANT: list[Row] = table(
    "invariant",
    ("utils:write_tsv", "synthetic: a 2-row frame", lambda c: c.tmp_path, _tsv, "tab-separated, round trips with its index; no frame writes an empty file"),
)


@pytest.mark.parametrize("row", ORACLE, ids=[r.id for r in ORACLE])
def test_oracle(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """The glue between recorded calls, recomputed."""
    run(row, ctx, request)


@pytest.mark.parametrize("row", INVARIANT, ids=[r.id for r in INVARIANT])
def test_invariant(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Writers round trip."""
    run(row, ctx, request)
