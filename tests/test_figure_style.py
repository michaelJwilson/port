"""One face for every figure, stated in `pyproject.toml` (`port.extensions.figure_style`)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.infra
def test_the_shipped_default_is_the_stated_face() -> None:
    """`DEFAULT`, used where no `pyproject.toml` is found, equals `[tool.port.figures]`."""
    from port.extensions.figure_style import DEFAULT, stated

    assert stated() == DEFAULT


@pytest.mark.infra
def test_the_stated_face_is_found_without_fallback() -> None:
    """matplotlib resolves the face to its own file, not to its fallback face."""
    import matplotlib.font_manager as fm
    from port.extensions.figure_style import stated

    face = stated()["font"]
    path = fm.findfont(fm.FontProperties(family=face), fallback_to_default=False)

    assert fm.FontProperties(fname=path).get_name() == face


@pytest.mark.infra
def test_a_face_matplotlib_cannot_find_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misspelt face raises, naming it, rather than drawing in DejaVu."""
    from port.extensions import figure_style

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.port.figures]\nfamily = "serif"\nfont = "No Such Face"\nmathtext = "stix"\n'
    )
    monkeypatch.setattr(figure_style, "PYPROJECT", pyproject)

    with pytest.raises(ValueError, match="No Such Face"):
        figure_style.figure_rc()


@pytest.mark.infra
def test_the_run_and_the_combined_figure_draw_in_the_stated_face() -> None:
    """Text drawn under `figure_font` and under `page_style` resolves to the stated
    face, after `cnaster.plotting` and seaborn have set their own."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]
    from port.extensions.combined_figure import page_style
    from port.extensions.figure_style import figure_font, stated

    face = stated()["font"]

    for context in (figure_font, page_style):
        with mpl.rc_context():
            sns.set_style("ticks")

            with context():
                figure, axis = plt.subplots()
                label = axis.set_xlabel("$m_N$ copy number")
                figure.canvas.draw()
                name = label.get_fontname()
                plt.close(figure)

        assert name == face, context.__name__


@pytest.mark.analytic
def test_each_page_share_is_its_fraction_of_the_text_block_less_its_caption() -> None:
    """`page_size` against `llncs`'s 122 by 193 mm block, by hand: "full" is the
    block less 1.5 in, the shares 1/3, 1/2 and 3/4 of it, and `columns`
    figures on a row split the 122 mm (T- #733)."""
    from port.extensions.figure_style import page_size

    room = 193.0 / 25.4 - 1.5

    assert page_size() == pytest.approx((122.0 / 25.4, room))
    for page, share in [("third", 1 / 3), ("half", 1 / 2), ("three_quarters", 3 / 4)]:
        assert page_size(page)[1] == pytest.approx(share * room)  # type: ignore[arg-type]

    width, height = page_size("third", columns=2)
    assert 2 * width == pytest.approx(122.0 / 25.4)
    assert height == pytest.approx(room / 3)


@pytest.mark.infra
def test_a_share_or_a_row_page_size_cannot_draw_is_refused() -> None:
    """An unnamed share and a row of no figures raise, naming the value."""
    from port.extensions.figure_style import page_size

    with pytest.raises(ValueError, match="'quarter'"):
        page_size("quarter")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns 0"):
        page_size("half", columns=0)
