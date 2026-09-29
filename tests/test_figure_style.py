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
