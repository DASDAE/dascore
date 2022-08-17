"""
Test patch utilities.
"""
import numpy as np
import pandas as pd
import pytest

import dascore
from dascore.exceptions import PatchAttributeError, PatchDimError
from dascore.utils.patch import _AttrsCoordsMixer


@dascore.patch_function(required_dims=("time", "distance"))
def time_dist_func(patch):
    """A dummy function to test time, distance coord requirement."""
    return patch


@dascore.patch_function(required_attrs=("bob",))
def require_bob(patch):
    """Require bob attribute"""
    return patch


@dascore.patch_function(required_attrs={"bob": "what about?"})
def require_bob_value(patch):
    """Require bob attribute with specific value."""
    return patch


class TestPatchFunction:
    """Tests for patch function decorator."""

    def test_no_call(self, random_patch):
        """Ensure the decorator can be used with or without parens"""

        @dascore.patch_function
        def func1(patch):
            """first test func"""
            return patch

        @dascore.patch_function()
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

    def test_access_original_function(self, random_patch):
        """Ensure the original function is still accessible."""
        hist1 = ",".join(random_patch.attrs["history"])
        # this calls the original function so it should bypass history logging
        out = require_bob.func(random_patch)
        hist2 = ",".join(out.attrs["history"])
        assert hist1 == hist2


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


class TestAttrsCoordsMixer:
    """Tests for handling complex interaction between attrs and coords."""

    @pytest.fixture()
    def attrs(self, random_patch):
        """return an attrs from a patch"""
        return random_patch.attrs

    @pytest.fixture()
    def coords(self, random_patch):
        """return an attrs from a patch"""
        return random_patch.coords

    @pytest.fixture()
    def mixer(self, attrs, coords, random_patch):
        """Return a mixer instance."""
        return _AttrsCoordsMixer(attrs, coords, random_patch.dims)

    def test_original_attrs_unchanged(self, mixer, attrs):
        """ensure the original attrs don't change."""
        t1 = attrs["time_min"]
        td = np.timedelta64(1, "s")
        mixer.update_attrs(time_min=t1 + td)
        new_attr, _ = mixer()
        assert new_attr is not attrs
        assert attrs["time_min"] + td == new_attr["time_min"]

    def test_original_coords_unchanged(self, mixer, coords, attrs):
        """ensure the original coords don't change."""
        t1 = attrs["time_min"]
        td = np.timedelta64(10, "s")
        mixer.update_attrs(time_min=t1 + td)
        _, new_coords = mixer()
        assert new_coords is not coords
        np.all(np.equal(coords["time"] + td, new_coords["time"]))

    def test_coords_unchanged(self, coords, attrs, random_patch):
        """ensure the original coords don't change."""
        # this was added to track down some mutation issues
        assert coords["time"].min() == attrs["time_min"]

    def test_starttime_updates_endtime(self, mixer, attrs, coords):
        """Ensure the end time gets updated when setting time_min"""
        t1 = attrs["time_min"]
        t_new = t1 + np.timedelta64(10_000_000, "s")
        mixer.update_attrs(time_min=t_new)
        attr, coords = mixer()
        assert attr["time_min"] == t_new
        # make sure time min was updated in coords
        time = coords["time"]
        assert np.min(time) == t_new

    def test_endtime_updates_starttime(self, mixer, attrs):
        """Ensure the start time gets updated when setting time_max."""
        tdist1 = attrs["time_max"] - attrs["time_min"]
        t2 = attrs["time_max"]
        t_new = t2 - np.timedelta64(10_000_000, "s")
        mixer.update_attrs(time_max=t_new)
        attr, coords = mixer()
        assert attr["time_max"] == t_new
        # make sure time min was updated in coords
        time = coords["time"]
        assert np.max(time) == t_new
        # ensure distance between start/end is the same
        tdist2 = attr["time_max"] - attr["time_min"]
        assert tdist2 == tdist1

    def test_coords_updates_times_and_distance(self, mixer, coords, attrs):
        """
        Ensure updating coords also updates attributes in attrs.
        """
        td = np.timedelta64(10, "s")
        dx = 10
        new_coords_kwarg = {
            "time": coords["time"] + td,
            "distance": coords["distance"] + dx,
        }

        mixer.update_coords(**new_coords_kwarg)
        new_attrs, new_coords = mixer()
        # first ensure coords actually updated
        assert np.all(new_coords["time"] == new_coords_kwarg["time"])
        assert np.all(new_coords["distance"] == new_coords_kwarg["distance"])
        # check attrs time are updated
        assert attrs["time_min"] + td == new_attrs["time_min"]
        assert attrs["time_max"] + td == new_attrs["time_max"]
        # check distance
        assert attrs["distance_min"] + dx == new_attrs["distance_min"]
        assert attrs["distance_max"] + dx == new_attrs["distance_max"]

    def test_relative_time_coord_no_absolute_time(self):
        """Ensure a time axis is still obtained with relative time and no start"""
        coords = dict(
            time=np.arange(10) * 0.1,
            distance=np.arange(10) * 10,
        )
        attrs = {}
        mixer = _AttrsCoordsMixer(attrs, coords, ("time", "distance"))
        new_attrs, new_coords = mixer()
        assert not np.any(pd.isnull(new_coords["time"]))

    def test_update_startttime_string(self, mixer, coords, attrs):
        """check start/end time relationship when starttime is a string."""
        new_start = np.datetime64("2000-01-01")
        duration = attrs["time_max"] - attrs["time_min"]
        mixer.update_attrs(time_min=str(new_start))
        new_attrs, _ = mixer()
        assert new_attrs["time_min"] == new_start
        assert new_attrs["time_max"] == new_start + duration

    def test_update_time_delta(self, mixer, coords, attrs):
        """Updating the time delta should also update endtimes."""
        one_sec = np.timedelta64(1, "s")
        td_old = attrs["d_time"]
        td_new = td_old * 2
        mixer.update_attrs(d_time=td_new)
        new_attrs, new_coords = mixer()
        assert new_attrs["d_time"] == td_new
        # first ensure new time coords are approximately new time delta
        new_time = new_coords["time"].values
        tdiff = (new_time[1:] - new_time[:-1]) / one_sec
        assert np.allclose(tdiff, td_new / one_sec)
        # also ensure the endtime has increased proportionately.
        old_duration = attrs["time_max"] - attrs["time_min"]
        new_duration = new_attrs["time_max"] - new_attrs["time_min"]
        assert np.isclose(old_duration / new_duration, td_old / td_new)
