from dejargonizer.prompt import LoadingBar, _render_progress_bar


def test_render_progress_bar_empty():
    line = _render_progress_bar(0.0, 0.0)
    assert "░" * 32 in line
    assert "  0.0%" in line


def test_render_progress_bar_half():
    line = _render_progress_bar(0.5, 1.0)
    assert "█" * 16 in line
    assert "░" * 16 in line
    assert " 50.0%" in line


def test_render_progress_bar_full():
    line = _render_progress_bar(1.0, 2.5)
    assert "█" * 32 in line
    assert "100.0%" in line


def test_render_progress_bar_clamps():
    assert "100.0%" in _render_progress_bar(1.5, 0.0)
    assert "  0.0%" in _render_progress_bar(-1.0, 0.0)


def test_loading_bar_finishes_at_100_percent(capsys):
    with LoadingBar() as bar:
        bar.bump(0.5)
    line = capsys.readouterr().err
    assert "100.0%" in line


def test_loading_bar_stops_quickly():
    import time

    started = time.perf_counter()
    with LoadingBar():
        pass
    assert time.perf_counter() - started < 2.0
