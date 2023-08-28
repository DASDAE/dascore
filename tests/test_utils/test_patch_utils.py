"""Test patch utilities."""
from __future__ import annotations

import pytest

import dascore as dc
from dascore.exceptions import (
    PatchAttributeError,
    PatchDimError,
)
from dascore.utils.patch import (
    get_dim_value_from_kwargs,
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
