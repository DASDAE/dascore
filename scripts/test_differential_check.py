"""Tests for the differential check script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from differential_check import check_dascore_path, compare, digest, get_calls

import dascore as dc


@pytest.fixture(scope="module")
def patch():
    """An example patch."""
    return dc.get_example_patch()


class TestDigest:
    """Tests for fingerprinting a patch."""

    def test_stable(self, patch):
        """The same patch gives the same fingerprint."""
        assert digest(patch) == digest(patch.new())

    def test_detects_tiny_data_change(self, patch):
        """A change too small to see is still a change."""
        other = patch.new(data=np.asarray(patch.data) + 1e-12)
        assert digest(patch) != digest(other)

    def test_detects_coord_change(self, patch):
        """So is a change to a coordinate."""
        other = patch.update_coords(distance=patch.get_array("distance") + 1)
        assert digest(patch) != digest(other)

    def test_ignores_history(self, patch):
        """History records the call, not the answer."""
        assert digest(patch) == digest(patch.update_attrs(history=["hello"]))


class TestCompare:
    """Tests for comparing two sets of fingerprints."""

    def test_no_report_when_equal(self, patch):
        """Identical results have nothing to report."""
        assert compare({"a": digest(patch)}, {"a": digest(patch)}) == []

    def test_reports_changed_field(self, patch):
        """A changed result names the field which changed."""
        other = patch.new(data=np.asarray(patch.data) + 1)
        report = compare({"a": digest(patch)}, {"a": digest(other)})
        assert report and "data_hash" in report[0]

    def test_reports_missing_call(self, patch):
        """A call which only one side has is reported."""
        assert compare({"a": digest(patch)}, {}) == ["a: only in before"]


class TestCalls:
    """Tests for the calls which get compared."""

    def test_calls_defined(self):
        """The comparison covers the converted patch functions."""
        calls = get_calls()
        assert len(calls) > 50
        assert {"add_scalar", "agg_mean", "pad_tuple"}.issubset(set(calls))

    def test_calls_run(self):
        """Every call runs and returns something which can be fingerprinted."""
        for name, call in get_calls().items():
            assert digest(call()), f"{name} returned nothing"


class TestDascorePath:
    """Tests for the guard against measuring the wrong dascore."""

    def test_accepts_the_worktree(self):
        """The dascore inside the worktree is the one wanted."""
        check_dascore_path(str(Path("/tmp/tree/dascore")), Path("/tmp/tree"))

    def test_rejects_another(self):
        """Any other one is a mistake, not a comparison."""
        with pytest.raises(RuntimeError, match="expected the dascore"):
            check_dascore_path("/somewhere/else/dascore", Path("/tmp/tree"))
