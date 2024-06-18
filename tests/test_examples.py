"""Tests for example fetching."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.examples import EXAMPLE_PATCHES
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

    def test_data_file_name(self):
        """Ensure get_example_spool works on a datafile."""
        spool = dc.get_example_spool("dispersion_event.h5")
        assert isinstance(spool, dc.BaseSpool)

    @pytest.mark.parametrize("name", EXAMPLE_PATCHES)
    def test_load_example_patch(self, name):
        """Ensure the registered example patches can all be loaded."""
        patch = dc.get_example_patch(name)
        assert isinstance(patch, dc.Patch)


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

    def test_data_file_name(self):
        """Ensure get_example_spool works on a datafile."""
        spool = dc.get_example_spool("dispersion_event.h5")
        assert isinstance(spool, dc.BaseSpool)


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
