"""
Test the progress bar.
"""
import dascore as dc
from dascore.utils.progress import track


class TestProgressBar:
    """Tests for the rich progress bar."""

    def test_progressbar_shows(self, monkeypatch):
        """Undo debug patch to progress bar shows."""
        monkeypatch.setattr(dc, "_debug", False)
        for _ in track([1, 2, 3], "testing_tracker"):
            pass
