"""Test the progress bar."""

from __future__ import annotations

from rich.progress import Progress

from dascore.config import set_config
from dascore.utils.progress import get_progress_instance, track


class TestProgressBar:
    """Tests for the rich progress bar."""

    def test_progressbar_shows(self):
        """Undo debug patch to progress bar shows."""
        with set_config(debug=False):
            for _ in track([1, 2, 3], "testing_tracker"):
                pass

    def test_get_basic_progress(self):
        """Ensure we can return a basic progress bar."""
        pbar = get_progress_instance("basic")
        assert isinstance(pbar, Progress)

    def test_basic_progress_refresh_rate_comes_from_config(self, monkeypatch):
        """The basic progress bar should honor the configured refresh rate."""
        seen = {}

        class DummyProgress:
            def __init__(self, *_args, **kwargs):
                seen.update(kwargs)

        monkeypatch.setattr("dascore.utils.progress.Progress", DummyProgress)
        with set_config(progress_basic_refresh_per_second=0.5):
            get_progress_instance("basic")
        assert seen["refresh_per_second"] == 0.5
