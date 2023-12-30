"""Tests for example fetching."""
from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnknownExampleError
from dascore.utils.time import to_float


class TestGetExamplePatch:
    """Test suite for `get_example_patch`."""

    def test_default(self):
        """Ensure calling get_example_patch with no args returns patch."""
        patch = dc.get_example_patch()
        assert isinstance(patch, dc.Patch)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example patch"):
            dc.get_example_patch("NotAnExampleRight????")

    def test_example_1(self):
        """Ensure example 1 returns a Patch."""
        out = dc.get_example_patch("example_event_1")
        assert isinstance(out, dc.Patch)

    def test_example_2(self):
        """Ensure example 2 returns a Patch."""
        out = dc.get_example_patch("example_event_2")
        assert isinstance(out, dc.Patch)

    def test_sin_wav(self):
        """Ensure the sin wave example can be loaded. See issee 229."""
        out = dc.get_example_patch("sin_wav")
        assert isinstance(out, dc.Patch)


class TestGetExampleSpool:
    """Test suite for `get_example_spool`."""

    def test_default(self):
        """Ensure calling get_example_spool with no args returns a Spool."""
        patch = dc.get_example_spool()
        assert isinstance(patch, dc.BaseSpool)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example spool"):
            dc.get_example_spool("NotAnExampleRight????")


class TestRickerMoveout:
    """Tests for Ricker moveout patch."""

    def test_moveout(self):
        """Ensure peaks of ricker wavelet line up with expected moveout."""
        velocity = 100
        patch = dc.get_example_patch("ricker_moveout", velocity=velocity)
        argmaxes = np.argmax(patch.data, axis=0)
        peak_times = patch.get_coord("time").values[argmaxes]
        moveout = to_float(peak_times - np.min(peak_times))
        distances = patch.get_coord("distance").values
        expected_moveout = distances / velocity
        assert np.allclose(moveout, expected_moveout)
