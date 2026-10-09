"""Integer copy sets from the fit's error bars, against planted lattice pairs (#353)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from port.extensions.copy_likelihood import Captured
from port.qa.errors import pinned_errors, pseudobulk
from port.scripts.run_cnaster import main


@pytest.mark.warning
def test_the_copy_decode_without_the_shift_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--no-shift` with the copy rows on is refused at the command line (#576)."""

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")

    for argv in ([str(config), "--no-shift"], [str(config), "--sal", "--no-shift"]):
        with pytest.raises(SystemExit):
            main(argv)
        assert "add --no-copy-cap" in capsys.readouterr().err


@pytest.mark.bug
@pytest.mark.parametrize("tau", [1e5, 1e12])
def test_copy_errors_refuse_a_fit_where_jax_hmms_beta_binomial_is_unstable(
    tau: float,
) -> None:
    """T- #599: at `tau >= JAX_TAU_LIMIT` `--copy-errors` refuses the fit."""

    result = {
        "new_log_mu": np.zeros((3, 1)),
        "new_p_binom": np.full((3, 1), 0.5),
        "new_alphas": np.array([[0.01]]),
        "new_taus": np.array([[tau]]),
    }
    captured = Captured(
        np.zeros((4, 2, 1)), np.array([4]), np.ones((4, 1)), np.ones((4, 1)), result
    )

    with pytest.raises(ValueError, match="T- #599"):
        pinned_errors(captured)


@pytest.mark.bug
def test_the_pseudobulk_refuses_a_clone_with_no_spots() -> None:
    """A pseudobulk with an empty clone is refused (#749 WP0)."""

    def captured(assignment: list[int]) -> Captured:
        result = {
            "new_assignment": np.array(assignment),
            "pred_cnv": np.zeros((4, 3), dtype=int),
        }
        n_spots = len(assignment)
        single_x = np.arange(4 * 2 * n_spots, dtype=float).reshape(4, 2, n_spots)
        return Captured(
            single_x,
            np.array([4]),
            np.ones((4, n_spots)),
            np.ones((4, n_spots)),
            result,
        )

    with pytest.raises(ValueError, match=r"clones \[1\]"):
        pseudobulk(captured([0, 2, 2]))

    stacked = pseudobulk(captured([0, 1, 2]))
    assert stacked["counts_nb"].shape == (12,)
    assert int(stacked["n_clones"]) == 3
