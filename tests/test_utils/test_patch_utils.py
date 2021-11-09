"""
Test patch utilities.
"""
import pytest
import numpy as np

import fios
from fios.utils.time import to_timedelta64
from fios.utils.patch import get_relative_deltas
from fios.exceptions import PatchDimError, PatchAttributeError


@fios.patch_function(required_dims=("time", "distance"))
def time_dist_func(patch):
    """A dummy function to test time, distance coord requirement."""
    return patch


@fios.patch_function(required_attrs=("bob",))
def require_bob(patch):
    return patch


@fios.patch_function(required_attrs={"bob": "what about?"})
def require_bob_value(patch):
    return patch


class TestPatchFunction:
    """Tests for patch function decorator."""

    def test_no_call(self, random_patch):
        """Ensure the decorator can be used with or without parens"""

        @fios.patch_function
        def func1(patch):
            """first test func"""
            return patch

        @fios.patch_function()
        def func2(patch):
            """second test func"""
            return patch

        pa1 = func1(random_patch)
        pa2 = func2(random_patch)
        assert pa1.equals(pa2)

    def test_has_required_dims(self, random_patch):
        """Test that the required dimension check works."""
        # passes if this doesn't raise
        time_dist_func(random_patch)

    def test_doesnt_have_required_dims(self, random_patch):
        """Raises if patch doesn't have required dimensions."""
        pa = random_patch.rename(time="for_a_hug")

        with pytest.raises(PatchDimError):
            time_dist_func(pa)

    def test_required_attrs(self, random_patch):
        """Tests for requiring certain attrs."""
        with pytest.raises(PatchAttributeError):
            require_bob(random_patch)

    def test_require_attr_exists(self, random_patch):
        """ "test for requiring an attr exists but not checking value"""
        pa1 = random_patch.update_attrs(bob=1)
        pa2 = pa1.update_attrs(bob=2)
        # both these should not raise
        require_bob(pa1)
        require_bob(pa2)

    def test_require_attr_and_value(self, random_patch):
        """Tests for requiring attribute and specific value."""
        pa1 = random_patch.update_attrs(bob="what about?")
        pa2 = random_patch.update_attrs(bob="lightening")
        require_bob_value(pa1)  # should not raise
        with pytest.raises(PatchAttributeError, match="lightening"):
            require_bob_value(pa2)


class TestHistory:
    """Tests for tracking patch processing history."""

    def test_simple_history(self, random_patch):
        """just make sure function is logged."""
        out = time_dist_func(random_patch)
        history = out.attrs["history"]
        assert len(history) == 1
        assert "time_dist_func" in history[0]

    def test_original_history_unchanged(self, random_patch):
        """Ensure logging history only occurs on new Patch"""
        first_history = list(random_patch.attrs["history"])
        _ = time_dist_func(random_patch)
        last_history = list(random_patch.attrs["history"])
        assert first_history == last_history


class TestGetRelativeDeltas:
    """Tests for getting times relative to start/stop."""

    t1 = np.datetime64("2020-01-01")
    t2 = np.datetime64("2021-01-01")

    def test_positive_numbers(self):
        """Tsets for using raw numbers to indicate relative times."""
        out = get_relative_deltas((1, 2), self.t1, self.t2)
        tds = to_timedelta64([1, 2])
        expected = slice(tds[0], tds[1])
        assert str(expected) == str(out)

    def test_null(self):
        """Tsets for using raw numbers to indicate relative times."""
        out = get_relative_deltas((1, None), self.t1, self.t2)
        expected = slice(to_timedelta64(1), None)
        assert str(expected) == str(out)

    def test_negative_numbers(self):
        """Negative number should reference end of window."""
        # single endeded test
        out = get_relative_deltas((-1, None), self.t1, self.t2)
        expected_dt = (self.t2 - to_timedelta64(1)) - self.t1
        assert str(out) == str(slice(expected_dt, None, None))
        # double ended test
        out = get_relative_deltas((-2, -1), self.t1, self.t2)
        expected_1 = (self.t2 - to_timedelta64(2)) - self.t1
        expected_2 = (self.t2 - to_timedelta64(1)) - self.t1
        assert str(out) == str(slice(expected_1, expected_2, None))

    def test_datetime64(self):
        """Ensure datetime64 works."""
        dt1 = np.datetime64("2020-01-01T00:00:02")
        dt2 = np.datetime64("2020-01-01T00:00:04")
        out = get_relative_deltas((dt1, dt2), self.t1, self.t2)
        expected_1 = to_timedelta64(2)
        expected_2 = to_timedelta64(4)
        assert str(slice(expected_1, expected_2)) == str(out)

    def test_no_time_raises_on_time_index(self):
        """Ensure when no time is given a ValueError is raised."""
        d1 = np.datetime64("2020-01-01")
        with pytest.raises(ValueError):
            get_relative_deltas((None, d1), None, None)
