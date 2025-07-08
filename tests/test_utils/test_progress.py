"""Test the progress bar."""

from __future__ import annotations

from rich.progress import Progress

import dascore as dc
from dascore.utils.progress import get_progress_instance, track


class TestProgressBar:
    """Tests for the rich progress bar."""

    def test_progressbar_shows(self, monkeypatch):
        """Undo debug patch to progress bar shows."""
        monkeypatch.setattr(dc, "_debug", False)
        for _ in track([1, 2, 3], "testing_tracker"):
            pass

    def test_get_basic_progress(self):
        """Ensure we can return a basic progress bar."""
        pbar = get_progress_instance("basic")
        assert isinstance(pbar, Progress)
