"""Test the progress bar."""
from __future__ import annotations

import dascore as dc
from dascore.utils.progress import track


class TestProgressBar:
    """Tests for the rich progress bar."""

    def test_progressbar_shows(self, monkeypatch):
        """A test for when the progress bar shows to run progress logic."""
        # Undo debug patch so progress bar shows.
        monkeypatch.setattr(dc, "_debug", False)
        for _ in track([1, 2, 3], "testing_tracker"):
            pass
