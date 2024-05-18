"""Test patch utilities."""
from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import (
    IncompatiblePatchError,
    PatchAttributeError,
    PatchDimError,
)
from dascore.utils.patch import (
    align_patch_coords,
    get_dim_value_from_kwargs,
    merge_compatible_coords_attrs,
)


@dc.patch_function(required_dims=("time", "distance"))
def time_dist_func(patch):
    """A dummy function to test time, distance coord requirement."""
    return patch


@dc.patch_function(required_attrs=("bob",))
def require_bob(patch):
    """Require bob attribute."""
    return patch


@dc.patch_function(required_attrs={"bob": "what about?"})
def require_bob_value(patch):
    """Require bob attribute with specific value."""
    return patch


class TestPatchFunction:
    """Tests for patch function decorator."""

    def test_no_call(self, random_patch):
        """Ensure the decorator can be used with or without parens."""

        @dc.patch_function
        def func1(patch):
            """First test func."""
            return patch

        @dc.patch_function()
        def func2(patch):
            """Second test func."""
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
        pa = random_patch.rename_coords(time="for_a_hug")

        with pytest.raises(PatchDimError):
            time_dist_func(pa)

    def test_required_attrs(self, random_patch):
        """Tests for requiring certain attrs."""
        with pytest.raises(PatchAttributeError):
            require_bob(random_patch)

    def test_require_attr_exists(self, random_patch):
        """Test for requiring an attr exists but not checking value."""
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
        """Just make sure function is logged."""
        out = time_dist_func(random_patch)
        history = out.attrs["history"]
        assert len(history) == 1
        assert "time_dist_func" in history[0]

    def test_original_history_unchanged(self, random_patch):
        """Ensure logging history only occurs on new Patch."""
        first_history = list(random_patch.attrs["history"])
        _ = time_dist_func(random_patch)
        last_history = list(random_patch.attrs["history"])
        assert first_history == last_history


class TestMergePatches:
    """Tests for merging patches together."""

    def test_deprecated(self, random_patch):
        """Ensure deprecation warning is raised."""
        from dascore.utils.patch import merge_patches

        with pytest.warns(DeprecationWarning, match="merge_patches is deprecated"):
            merge_patches(random_patch)


class TestGetDimValueFromKwargs:
    """Tests for getting dimensional values."""

    def test_raises_no_overlap(self, random_patch):
        """Test that an exception is raised when key doesn't exist."""
        kwargs = {}
        with pytest.raises(PatchDimError):
            get_dim_value_from_kwargs(random_patch, kwargs)


class TestAlignPatches:
    """Tests for aligning patches."""

    def test_align_self(self, random_patch):
        """A patch should align with itself."""
        out1, out2 = align_patch_coords(random_patch, random_patch)
        assert out1 == out2 == random_patch

    def test_different_dims(self, random_patch):
        """Ensure alignment works for different dimensions."""
        no_time = random_patch.select(time=0, samples=True).squeeze()
        out1, out2 = align_patch_coords(no_time, random_patch)
        assert out1.dims == out2.dims
        assert out1.ndim == out2.ndim
        eq_or_q = [x == y or 1 in {x, y} for x, y in zip(out1.shape, out2.shape)]
        assert all(eq_or_q)

    def test_subset(self, random_patch):
        """Ensure a subset of a patch slices output."""
        sub = random_patch.select(time=(1, 15), samples=True)
        out1, out2 = align_patch_coords(sub, random_patch)
        assert out1.shape == out2.shape
        assert out1.coords == out2.coords
        # The data should all be the same since this is a sub-patch.
        assert np.all(out1.data == out2.data)

    def test_align_no_overlap(self, random_patch):
        """Alignment with no overlap should return null."""
        dist = random_patch.get_coord("distance")
        new = random_patch.update_coords(distance_min=dist.max() + 10)
        out1, out2 = align_patch_coords(new, random_patch)
        assert out1.shape == out2.shape
        assert out1.coords == out2.coords
        # Patches should be degenerate.
        assert 0 in set(out1.shape)

    def test_no_common_dims_raises(self, random_patch):
        """Patches with no common dims should not be align-able."""
        new = random_patch.rename_coords(time="money", distance="gold")
        msg = "align patches with no shared dimensions."
        with pytest.raises(PatchDimError, match=msg):
            align_patch_coords(new, random_patch)

    def test_new_dims_each_patch(self, random_patch):
        """Tests for when there are new dimensions on each patch."""
        small = random_patch.select(time=(0, 10), distance=(0, 12), samples=True)
        p1 = small.rename_coords(time="money")
        p2 = small.rename_coords(time="about")
        out1, out2 = align_patch_coords(p1, p2)
        assert out1.dims == out2.dims
        assert out1.ndim == out2.ndim

    def test_len_1_non_coord_patch(self, random_patch):
        """Tests for when there are non-coordinate dimensions of len 1."""
        non_patch = random_patch.first("time")
        # Coords should be expanded in out2.
        out1, out2 = align_patch_coords(non_patch, random_patch)


class TestMergeCompatibleCoordsAttrs:
    """Tests for merging compatible attrs, coords."""

    def test_simple(self, random_patch):
        """Simple merge test."""
        coords, attrs = merge_compatible_coords_attrs(random_patch, random_patch)
        assert coords == random_patch.coords
        assert attrs == random_patch.attrs

    def test_incompatible_dims(self, random_patch):
        """Ensure incompatible dims raises."""
        new = random_patch.rename_coords(time="money")
        match = "not compatible for merging"
        with pytest.raises(IncompatiblePatchError, match=match):
            merge_compatible_coords_attrs(random_patch, new)

    def test_incompatible_coords(self, random_patch):
        """Ensure an incompatible error is raised for coords that dont match."""
        new_time = random_patch.attrs.time_max
        new = random_patch.update_attrs(time_min=new_time)
        match = "coordinates are not equal"
        with pytest.raises(IncompatiblePatchError, match=match):
            merge_compatible_coords_attrs(new, random_patch)

    def test_incompatible_attrs(self, random_patch):
        """Ensure if attrs are off an Error is raised."""
        new = random_patch.update_attrs(network="TA")
        match = "attributes are not equal"
        with pytest.raises(IncompatiblePatchError, match=match):
            merge_compatible_coords_attrs(new, random_patch)

    def test_extra_coord(self, random_patch, random_patch_with_lat_lon):
        """Extra coords on both patch should end up in the merged patch."""
        new_coord = np.ones(random_patch.coord_shapes["time"])
        pa1 = random_patch.update_coords(new_time=("time", new_coord))
        pa2 = random_patch_with_lat_lon.update_coords(new_time=("time", new_coord))
        expected = set(pa1.coords.coord_map) | set(pa2.coords.coord_map)
        coords, attrs = merge_compatible_coords_attrs(pa1, pa2)
        assert set(coords.coord_map) == expected
        assert set(attrs.coords) == expected

    def test_extra_attrs(self, random_patch):
        """Ensure extra attributes are added to patch."""
        patch = random_patch.update_attrs(new_attr=10)
        coords, attrs = merge_compatible_coords_attrs(patch, random_patch)
        assert attrs.get("new_attr") == 10

    def test_different_dims(self, random_patch):
        """Ensure we can merge patches which share some dims but not all."""
        patch1 = random_patch
        patch2 = random_patch.rename_coords(time="money")
        coord, attrs = merge_compatible_coords_attrs(
            patch1, patch2, dim_intersection=True
        )
        assert set(coord.dims) == (set(patch1.dims) | set(patch2.dims))
