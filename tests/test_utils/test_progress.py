"""Test the progress bar."""

from __future__ import annotations

import pytest
from rich.progress import Progress

from dascore.config import config_context
from dascore.utils.progress import get_progress_instance, track


class TestProgressBar:
    """Tests for the rich progress bar."""

    @pytest.mark.concurrency
    def test_progressbar_shows(self):
        """Undo debug patch to progress bar shows."""
        with config_context(debug=False):
            for _ in track([1, 2, 3], "testing_tracker"):
                pass

    def test_unsized_iterable(self):
        """Unsized iterables without a length just skip the progress bar."""
        with config_context(debug=False):
            assert list(track(iter([1, 2, 3]), "unsized_tracker")) == [1, 2, 3]

    def test_given_length_is_the_reported_total(self, monkeypatch):
        """A length passed in is the bar's total; the sequence is not measured."""
        seen = {}

        class Unmeasurable(list):
            """A sequence which fails if anything asks for its length."""

            def __len__(self):
                raise AssertionError("the sequence should not be measured")

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def track(self, sequence, total=None, **_):
                seen["total"] = total
                yield from sequence

        monkeypatch.setattr(
            "dascore.utils.progress.get_progress_instance", lambda _: DummyProgress()
        )
        with config_context(debug=False):
            sequence = Unmeasurable([1, 2, 3])
            assert list(track(sequence, "length_tracker", length=10)) == [1, 2, 3]
        assert seen["total"] == 10

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
        with config_context(progress_basic_refresh_per_second=0.5):
            get_progress_instance("basic")
        assert seen["refresh_per_second"] == 0.5
