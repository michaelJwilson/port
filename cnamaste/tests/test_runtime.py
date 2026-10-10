"""Runtime goals: one placeholder row per speedup port measured, not yet ported to cnamaste.

Each row names the cnamaste function, the port issue, port's measured ratio
and the size it was measured at, and how cnamaste would measure its own.
The goal is CLAUDE.md's bar, >= 2x over the current code at a stress size;
a row port measured below it (1.83x, 1.90x) would land as a simplification,
on its evidence of equivalence alone. Every row skips: no timing loop runs
until a port lands, and the row then becomes its benchmark.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest


class Goal(NamedTuple):
    function: str
    issue: str
    measured: str
    size: str
    method: str

    @property
    def below_bar(self) -> bool:
        ratio = self.measured.split("x")[0].split("-")[0]
        try:
            return float(ratio) < 2.0
        except ValueError:
            return False


GOALS: list[Goal] = [
    Goal("omics.form_gene_snp_table", "#190", "18.3x", "CalicoST easy, 3,000 spots", "pytest-benchmark on the replayed 01_genes input"),
    Goal("omics.assign_initial_blocks", "#190", "10.5x", "CalicoST easy, 3,000 spots", "pytest-benchmark on the replayed 02_blocks input"),
    Goal("spatial.best_equal_partition", "#190", "8.5x", "CalicoST easy, 3,000 spots", "pytest-benchmark on the loaded coordinates"),
    Goal("the preprocessing chain, load_input_data to summarize_counts_for_bins", "#190", "33.8x", "CalicoST easy, 3,000 spots", "wall of stages 00-04 replayed"),
    Goal("normal_spot.normal_baf_bin_filter", "#174", "80.7x", "CalicoST easy", "pytest-benchmark on the replayed 06_normal input"),
    Goal("omics.summarize_counts_for_blocks", "#198 / #569", "3.50x", "37,636 spots (memory 3.99 -> 2.86 GB)", "pytest-benchmark and peak RSS at a stress size"),
    Goal("omics.summarize_counts_for_bins", "#198 / #569", "1.90x", "37,636 spots", "pytest-benchmark; below 2x: a simplification"),
    Goal("io.load_input_data, sparse counts", "#186 / #188 / PR #499", "2.18x", "780 -> 545 MB; 5.18 -> 2.72 GB at 6,000 spots", "wall and peak RSS on a 6,000-spot draw"),
    Goal("recomb.get_sitewise_transmat", "#438", "13.8x", "33,000 blocks", "pytest-benchmark on a 33,000-block table"),
    Goal("pseudobulk.merge_pseudobulk_by_index_mix", "#488", "1.83x", "dev_tree at 60 x 50", "pytest-benchmark; below 2x: a simplification"),
    Goal("count_encoder.CountEncoder.decode_array", "#799", "3.1x", "the fit's codes", "pytest-benchmark of decode against an index gather"),
    Goal("hmrf.compute_loglike_spot_assignment, strided", "#59", "3.9-7.2x", "the HMM/spatial boundary", "pytest-benchmark over the field"),
    Goal("hmrf.pipeline_clone_assignment, fused", "#206", "2.08x", "30,000 x 5,000 (two-step is OOM-killed)", "wall and peak RSS at 30,000 x 5,000"),
    Goal("utils.write_fig and the plotting", "#195", "5.3x", "renderer 8,287 -> 1,036 MB (memory half in PR1, T- #692)", "wall of a run with figures on"),
    Goal("run_cnamaste, the whole run", "#836", "ledger", "each SUPPORTED fixture", "wall against cnaster's ledger row"),
]


@pytest.mark.parametrize("goal", GOALS, ids=[g.function for g in GOALS])
def test_a_runtime_goal(goal: Goal) -> None:
    """A placeholder: the goal and how it would be measured."""
    bar = "below the 2x bar: a simplification" if goal.below_bar else "goal >= 2x at stress size"
    pytest.skip(f"runtime goal, not yet ported: port {goal.issue} measured {goal.measured} at {goal.size}; {bar}; {goal.method}")
