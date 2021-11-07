"""
Test patch utilities.
"""
import pytest

import fios
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
