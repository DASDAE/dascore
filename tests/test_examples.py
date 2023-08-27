"""Tests for example fetching."""
from __future__ import annotations

import pytest

import dascore as dc
from dascore.exceptions import UnknownExample


class TestGetExamplePatch:
    """Test suite for `get_example_patch`."""

    def test_default(self):
        """Ensure calling get_example_patch with no args returns patch."""
        patch = dc.get_example_patch()
        assert isinstance(patch, dc.Patch)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExample, match="No example patch"):
            dc.get_example_patch("NotAnExampleRight????")

    def test_example_1(self):
        """Ensure example 1 returns a Patch."""
        out = dc.get_example_patch("example_event_1")
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
        with pytest.raises(UnknownExample, match="No example spool"):
            dc.get_example_spool("NotAnExampleRight????")
