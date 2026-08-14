"""Test the progress bar."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from rich.progress import Progress

import dascore as dc
from dascore.config import config_context
from dascore.exceptions import ParameterError
from dascore.utils.progress import get_progress_instance, get_track_length, track


class TestGetTrackLength:
    """Tests for deciding the total a progress bar reports."""

    def test_sized_sequence_is_measured(self):
        """A sequence with no length given is measured."""
        assert get_track_length([1, 2, 3], None, 1) == 3

    def test_given_length_is_not_remeasured(self):
        """A length passed in is used as-is; the sequence is left alone."""

        class Unmeasurable(list):
            """A sequence which fails if anything asks for its length."""

            def __len__(self):
                raise AssertionError("the sequence should not be measured")

        assert get_track_length(Unmeasurable([1, 2, 3]), 10, 1) == 10

    def test_unsized_sequence_gets_no_bar(self):
        """A generator has no length to report."""
        assert get_track_length(iter([1, 2, 3]), None, 1) == 0

    def test_sized_but_unmeasurable_gets_no_bar(self):
        """A 0-d array is Sized but len() still raises on it."""
        assert get_track_length(np.array(5), None, 1) == 0

    def test_shorter_than_min_length_gets_no_bar(self):
        """A sequence below the minimum is not worth a bar."""
        assert get_track_length([1, 2], None, 5) == 0


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

    def test_get_basic_progress(self):
        """Ensure we can return a basic progress bar."""
        pbar = get_progress_instance("basic")
        assert isinstance(pbar, Progress)

    def test_none_disables_the_bar(self):
        """None is the off switch, and iteration is unaffected."""
        with config_context(debug=False):
            assert list(track([1, 2, 3], "off_tracker", None)) == [1, 2, 3]

    @pytest.mark.parametrize("bad", [False, True, "quiet"])
    def test_a_value_outside_the_levels_raises(self, bad):
        """Anything else would fall through to the standard bar."""
        with pytest.raises(ParameterError, match="progress must be one of"):
            list(track([1, 2, 3], "bad_tracker", bad))

    def test_an_empty_spool_still_refuses_a_bad_level(self):
        """Acceptance must not depend on there being data to track."""
        client = ThreadPoolExecutor()
        try:
            with pytest.raises(ParameterError, match="progress must be one of"):
                dc.spool([]).map(lambda patch: patch, client=client, progress=False)
        finally:
            client.shutdown()

    def test_a_progress_instance_is_accepted(self):
        """A caller's own bar is not a level, and is still allowed."""
        with config_context(debug=False):
            assert list(track([1, 2, 3], "own_tracker", Progress())) == [1, 2, 3]

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
