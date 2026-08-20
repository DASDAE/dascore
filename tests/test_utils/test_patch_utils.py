"""Test patch utilities."""

from __future__ import annotations

from typing import Annotated, Literal, get_type_hints

import numpy as np
import pandas as pd
import pydantic
import pytest
from pydantic import Field

import dascore as dc
from dascore.config import config_context
from dascore.constants import PatchType
from dascore.exceptions import (
    CoordError,
    IncompatiblePatchError,
    ParameterError,
    PatchAttributeError,
    PatchCoordinateError,
)
from dascore.units import percent
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import (
    _force_patch_merge,
    _spool_up,
    align_patch_coords,
    check_dims,
    check_kind,
    concatenate_patches,
    get_dim_axis_value,
    get_patch_kind,
    get_patch_names,
    get_patch_window_size,
    get_window_axis_step,
    merge_compatible_coords_attrs,
    merge_patches,
    patch_function,
    patches_to_df,
    stack_patches,
    swap_kwargs_dim_to_axis,
)


@dc.patch_function(required_dims=("time", "distance"))
def time_dist_dims(patch):
    """A dummy function to test time, distance dim requirement."""
    return patch


@dc.patch_function()
def add_one(patch):
    """A simple function which modifies the patch."""
    return patch.new(data=patch.data + 1)


@dc.patch_function(required_coords=("time", "distance"))
def time_dist_coord(patch):
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
        time_dist_dims(random_patch)

    def test_doesnt_have_required_dims(self, random_patch):
        """Raises if patch doesn't have required dimensions."""
        pa = random_patch.rename_coords(time="for_a_hug")

        with pytest.raises(PatchCoordinateError):
            time_dist_dims(pa)

    def test_has_required_coords(self, random_patch, random_dft_patch):
        """Test that the required coord check works."""
        # if the coords are dimensions it should work
        time_dist_coord(random_patch)
        # or if they are only coordinates it should work
        time_dist_coord(random_dft_patch)

    def test_doesnt_have_required_coords(self, random_patch):
        """Raises if patch doesn't have required coords."""
        pa = random_patch.rename_coords(time="for_a_hug")
        with pytest.raises(PatchCoordinateError):
            time_dist_coord(pa)

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

    def test_type_checking(self, random_patch):
        """Ensure the type-checking for validate_call works."""

        @patch_function(validate_call=True)
        def some_func(
            patch: PatchType,
            some_int: int,
            specific_float: Annotated[float, Field(ge=0, le=1)],
            lit_str: Literal["bob", "bill", "marely"] = "bill",
        ):
            """A test function for type checking"""
            return patch

        patch = dc.get_example_patch()
        ok = some_func(patch, some_int=23, specific_float=0.2, lit_str="bob")
        assert isinstance(ok, dc.Patch)
        assert ok is patch

        with pytest.raises(pydantic.ValidationError):
            some_func(patch, some_int=10, specific_float=20.0)

    def test_data_type(self, random_patch):
        """Ensure the decorator can set the output data_type."""

        @patch_function(data_type="strain_rate")
        def some_func(patch):
            """A test function for setting the output data_type."""
            return patch.new(data=patch.data + 1)

        out = some_func(random_patch)

        assert out.attrs.data_type == "strain_rate"

    def test_data_type_overwrites_returned_patch_attr(self, random_patch):
        """Ensure the decorator data_type takes precedence."""

        @patch_function(data_type="strain_rate")
        def some_func(patch):
            """A test function with a conflicting output data_type."""
            return patch.new(data=patch.data + 1, attrs={"data_type": "velocity"})

        out = some_func(random_patch)

        assert out.attrs.data_type == "strain_rate"

    def test_data_type_none_preserves_existing_behavior(self, random_patch):
        """Ensure the default data_type argument leaves attrs unchanged."""

        @patch_function()
        def some_func(patch):
            """A test function without decorator-managed data_type."""
            return patch.new(data=patch.data + 1, attrs={"data_type": "velocity"})

        out = some_func(random_patch)

        assert out.attrs.data_type == "velocity"

    def test_data_type_none_preserves_inherited_data_type(self, random_patch):
        """Ensure the default data_type argument preserves inherited attrs."""

        @patch_function()
        def some_func(patch):
            """A test function without decorator-managed data_type."""
            return patch.new(data=patch.data + 1)

        patch = random_patch.update_attrs(data_type="velocity")
        out = some_func(patch)

        assert out.attrs.data_type == "velocity"

    def test_empty_data_type_clears_returned_patch_attr(self, random_patch):
        """Ensure the decorator can clear the output data_type."""

        @patch_function(data_type="")
        def some_func(patch):
            """A test function for clearing data_type."""
            return patch.new(data=patch.data + 1, attrs={"data_type": "velocity"})

        out = some_func(random_patch)

        assert out.attrs.data_type == ""

    def test_data_type_and_history_use_one_attr_update(self, random_patch, monkeypatch):
        """Ensure decorator-managed attrs are updated together."""
        update_count = 0
        original_update_attrs = dc.Patch.update_attrs

        def update_attrs(self, **attrs):
            nonlocal update_count
            update_count += 1
            return original_update_attrs(self, **attrs)

        @patch_function(data_type="strain_rate")
        def some_func(patch):
            """A test function for setting data_type and history."""
            return patch.new(data=patch.data + 1)

        monkeypatch.setattr(dc.Patch, "update_attrs", update_attrs)
        out = some_func(random_patch)

        assert out.attrs.data_type == "strain_rate"
        assert len(out.attrs.history) == len(random_patch.attrs.history) + 1
        assert update_count == 1


class TestHistory:
    """Tests for tracking patch processing history."""

    def test_simple_history(self, random_patch):
        """Just make sure function is logged."""
        out = add_one(random_patch)
        history = out.attrs["history"]
        assert len(history) == 1
        assert "add_one" in history[0]

    def test_original_history_unchanged(self, random_patch):
        """Ensure logging history only occurs on new Patch."""
        first_history = list(random_patch.attrs["history"])
        _ = add_one(random_patch)
        last_history = list(random_patch.attrs["history"])
        assert first_history == last_history

    def test_no_history_unchanged_patch(self, random_patch):
        """
        Ensure no new history is created by a function that returns the same patch.
        """
        out = time_dist_dims(random_patch)
        assert out.attrs.history == random_patch.attrs.history
        # No new patch should have been created.
        assert out is random_patch

    def test_patch_argument_truncated(self, random_patch):
        """
        Ensure patch arguments are truncated in history (issue #529).
        """

        @patch_function()
        def func_with_patch_arg(patch, other_patch):
            """Test function with patch argument."""
            return patch.new(data=patch.data + 1)

        other = random_patch.mean("time")
        result = func_with_patch_arg(random_patch, other_patch=other)
        history_str = result.attrs.history[0]

        # History should contain truncated representation, not full patch string
        assert "Patch..." in history_str
        # Should not contain the full multi-line patch representation
        assert "\n➤ Coordinates" not in history_str
        assert "\n➤ Data" not in history_str
        # Check it's reasonably short (less than 100 chars)
        assert len(history_str) < 100

    def test_history_disabled_for_patch_function(self, random_patch):
        """Decorator-based history should respect disabled config."""
        with config_context(patch_history="disabled"):
            out = add_one(random_patch)
        assert out.attrs.history == random_patch.attrs.history

    def test_disabled_preserves_existing_history(self, random_patch):
        """Disabling history should stop new entries without clearing old ones."""
        patch = random_patch.pass_filter(time=(10, 20))
        original_history = patch.attrs.history
        with config_context(patch_history="disabled"):
            out = add_one(patch)
        assert out.attrs.history == original_history

    def test_concatenate_history_disabled(self, random_patch):
        """Concatenate should use the shared history helper."""
        with config_context(patch_history="disabled"):
            out = concatenate_patches([random_patch, random_patch], time=None)
        assert len(out) == 1
        assert out[0].attrs.history == random_patch.attrs.history

    def test_stack_history_disabled(self, random_patch):
        """Stack should use the shared history helper."""
        with config_context(patch_history="disabled"):
            out = stack_patches([random_patch, random_patch])
        assert out.attrs.history == random_patch.attrs.history


class TestPatchMergeWorkflow:
    """Tests for the supported patch merge workflow."""

    def test_spool_chunk_replacement(self, random_patch):
        """Ensure spool.chunk remains the supported merge path."""
        out = dc.spool([random_patch]).chunk(time=None)
        assert len(out) == 1

    def test_merge_compatible_coords_attrs_ignores_private_attrs(self, random_patch):
        """Private attrs should not make otherwise compatible patches fail."""
        patch_1 = random_patch.update_attrs(_source_patch_key="one")
        patch_2 = random_patch.update_attrs(_source_patch_key="two")
        coords, attrs = merge_compatible_coords_attrs(patch_1, patch_2)
        assert coords == random_patch.coords
        assert attrs["_source_patch_key"] == "one"

    def test_binary_ops_ignore_private_attrs(self, random_patch):
        """Binary ops should work on patches with different private ids."""
        patch_1 = random_patch.update_attrs(_source_patch_key="one")
        patch_2 = random_patch.update_attrs(_source_patch_key="two")
        out = patch_1 + patch_2
        assert isinstance(out, dc.Patch)
        assert out.attrs["_source_patch_key"] == "one"


class TestGetDimAxisValue:
    """Tests for getting the dimension name, axis, and value."""

    def test_raises_no_overlap(self, random_patch):
        """Test that an exception is raised when key doesn't exist."""
        kwargs = {}
        msg = "You must specify"
        with pytest.raises(ParameterError, match=msg):
            get_dim_axis_value(random_patch, kwargs=kwargs)

    def test_raises_extra(self, random_patch):
        """Ensure extra args/kwargs raise."""
        msg = "not found in the patch"
        with pytest.raises(PatchCoordinateError, match=msg):
            get_dim_axis_value(random_patch, kwargs={"bob": 10, "time": None})
        with pytest.raises(PatchCoordinateError, match=msg):
            get_dim_axis_value(random_patch, args=("bob", "distance"))

    def test_get_single(self, random_patch):
        """Ensure a single kwarg works."""
        value = 10
        for axis, dim in enumerate(random_patch.dims):
            # test kwargs
            out = get_dim_axis_value(random_patch, kwargs={dim: value})
            assert len(out) == 1
            assert out[0] == (dim, axis, value)
            # test args
            out = get_dim_axis_value(random_patch, args=(dim,))
            assert len(out) == 1
            assert out[0] == (dim, axis, None)

    def test_multiple(self, random_patch):
        """Ensure multiple kwargs works."""
        kwargs = {x: 10 for x in random_patch.dims}
        # Allow multiple should return a tuple.
        out = get_dim_axis_value(random_patch, kwargs=kwargs, allow_multiple=True)
        assert len(out) == len(random_patch.dims)
        # But if not it should raise
        msg = "You must specify"
        with pytest.raises(ParameterError, match=msg):
            get_dim_axis_value(random_patch, kwargs=kwargs, allow_multiple=False)


class TestPatchesToDF:
    """Test for getting metadata from patch into a dataframe."""

    def test_spool_input(self, random_spool):
        """A spool should return its contents with its patches embedded."""
        df = patches_to_df(random_spool)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(random_spool)
        # the "patch" column carries the actual patches, not None
        assert "patch" in df.columns
        assert all(isinstance(x, dc.Patch) for x in df["patch"])

    def test_list_of_patches_input(self, random_spool):
        """A plain sequence of patches is scanned and the patches embedded."""
        patches = list(random_spool)
        df = patches_to_df(patches)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(patches)
        assert list(df["patch"]) == patches

    def test_empty_list_input(self):
        """An empty sequence produces an empty frame with the right columns."""
        df = patches_to_df([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "patch" in df.columns

    def test_dataframe_input(self, random_spool):
        """The function should be idempotent."""
        df = random_spool.get_contents()
        out = patches_to_df(df)
        out2 = patches_to_df(out)
        eq = out == out2
        is_null = pd.isnull(out) & pd.isnull(out2)
        assert np.all(eq | is_null)

    def test_history_added(self, random_spool):
        """Ensure the history column gets added."""
        df = random_spool.get_contents().drop(columns="history", errors="ignore")
        out = patches_to_df(df)
        assert "history" in out.columns

    def test_patch_column_added(self, random_spool):
        """Ensure the patch column gets added when dataframe input lacks it."""
        df = random_spool.get_contents().drop(columns="patch", errors="ignore")
        out = patches_to_df(df)
        assert "patch" in out.columns
        assert len(out["patch"]) == len(df)
        assert out["patch"].isnull().all()


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

    def test_align_no_overlap_raises(self, random_patch):
        """Sharing a dimension but none of its values is a conflict."""
        dist = random_patch.get_coord("distance")
        new = random_patch.update_coords(distance_min=dist.max() + 10)
        with pytest.raises(PatchCoordinateError, match="share no values"):
            align_patch_coords(new, random_patch)

    def test_no_common_dims_raises(self, random_patch):
        """Patches with no common dims should not be align-able."""
        new = random_patch.rename_coords(time="money", distance="gold")
        msg = "align patches with no shared dimensions."
        with pytest.raises(PatchCoordinateError, match=msg):
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
        desired_shape = tuple(max(x, y) for x, y in zip(out1.shape, out2.shape))
        # This will raise if the shapes aren't broadcastable
        np.broadcast_shapes(out1.shape, out2.shape, desired_shape)


class TestCheckKind:
    """Tests for deciding whether two patches are the same kind."""

    def test_kind_is_the_configured_attrs(self, random_patch):
        """The kind holds exactly the config's attrs, in order."""
        kind = get_patch_kind(random_patch)
        names = dc.get_config().patch_kind_attrs
        assert tuple(kind) == names
        assert kind["tag"] == random_patch.attrs.tag
        # Attrs work as well as patches.
        assert get_patch_kind(random_patch.attrs) == kind

    def test_processing_keeps_kind(self, random_patch):
        """Coordinates, history, units, and ids never enter the kind."""
        changed = (
            random_patch.pass_filter(time=(None, 10))
            .select(time=(1, 20), samples=True)
            .set_units("m/s")
        )
        assert check_kind(random_patch, changed)

    @pytest.mark.parametrize(
        "update",
        [
            {"acquisition_key": "TA.R2D1..RAW"},
            {"data_category": "DTS"},
            {"tag": "other"},
        ],
    )
    def test_kind_attr_conflicts(self, random_patch, update):
        """A conflicting kind attr raises, warns, or quietly reports False."""
        random_patch = random_patch.update_attrs(
            acquisition_key="DAS.R2D1..RAW", data_category="DAS", tag="a"
        )
        other = random_patch.update_attrs(**update)
        msg = "not the same kind"
        with pytest.raises(IncompatiblePatchError, match=msg):
            check_kind(random_patch, other)
        with pytest.warns(UserWarning, match=msg):
            assert not check_kind(random_patch, other, check_behavior="warn")
        with suppress_warnings(action="error"):
            assert not check_kind(random_patch, other, check_behavior="ignore")

    def test_message_names_the_attr(self, random_patch):
        """The error says which attribute differs and how."""
        other = random_patch.update_attrs(tag="other")
        with pytest.raises(IncompatiblePatchError, match="'tag'") as info:
            check_kind(random_patch, other)
        assert "patch_kind_attrs" in str(info.value)

    def test_custom_attr_is_not_kind_unless_configured(self, random_patch):
        """Only the configured names count; config can add to them."""
        pa1 = random_patch.update_attrs(foo="a")
        pa2 = random_patch.update_attrs(foo="b")
        assert check_kind(pa1, pa2)
        with config_context(patch_kind_attrs=("foo",)):
            with pytest.raises(IncompatiblePatchError, match="'foo'"):
                check_kind(pa1, pa2)
            # and the default names no longer matter
            assert check_kind(pa1, pa1.update_attrs(tag="other"))

    def test_missing_attr_matches_anything(self, random_patch):
        """A missing value conflicts with nothing; only known values can."""
        with config_context(patch_kind_attrs=("foo",)):
            assert get_patch_kind(random_patch)["foo"] is None
            assert check_kind(random_patch, random_patch.new())
            assert check_kind(random_patch, random_patch.update_attrs(foo="a"))
            with pytest.raises(IncompatiblePatchError, match="'foo'"):
                check_kind(
                    random_patch.update_attrs(foo="a"),
                    random_patch.update_attrs(foo="b"),
                )

    def test_data_type_is_not_kind(self, random_patch):
        """Processing changes data_type, so it must not gate arithmetic."""
        vel = random_patch.update_attrs(data_type="velocity")
        env = vel.envelope("time")
        assert env.attrs.data_type == "envelope"
        assert check_kind(vel, env)
        assert (vel / env).attrs.data_type == "velocity"

    def test_empty_string_is_absent(self, random_patch):
        """An empty attr (a legacy reader's network="") is the same as none."""
        legacy = random_patch.update_attrs(network="", station="")
        assert get_patch_kind(legacy)["network"] is None
        assert check_kind(legacy, random_patch)

    def test_array_valued_kind_attr(self, random_patch):
        """Array-valued kind attrs compare as a whole rather than raising."""
        pa1 = random_patch.update_attrs(foo=np.array([1, 2]))
        pa2 = random_patch.update_attrs(foo=np.array([1, 2]))
        pa3 = random_patch.update_attrs(foo=np.array([1, 3]))
        with config_context(patch_kind_attrs=("foo",)):
            assert check_kind(pa1, pa2)
            assert not check_kind(pa1, pa3, check_behavior="ignore")

    def test_retired_check_behavior_raises(self, random_patch):
        """The behavior argument is validated up front."""
        with pytest.raises(ParameterError, match="behavior must be one of"):
            check_kind(random_patch, random_patch, check_behavior=None)


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
        """Ensure an incompatible error is raised for coords that don't match."""
        new_time = random_patch.get_coord("time").max()
        new = random_patch.update_coords(time_min=new_time)
        match = "coordinates are not equal"
        with pytest.raises(IncompatiblePatchError, match=match):
            merge_compatible_coords_attrs(new, random_patch)

    def test_incompatible_attrs(self, random_patch):
        """A conflicting kind attr raises."""
        new = random_patch.update_attrs(tag="other")
        match = "not the same kind"
        with pytest.raises(IncompatiblePatchError, match=match):
            merge_compatible_coords_attrs(new, random_patch)

    def test_kind_is_the_union(self, random_patch):
        """A value only one patch knows is carried by the result."""
        keyed = random_patch.update_attrs(acquisition_key="TA.R2D1..RAW")
        for first, second in [(random_patch, keyed), (keyed, random_patch)]:
            _, attrs = merge_compatible_coords_attrs(first, second)
            assert attrs.acquisition_key == "TA.R2D1..RAW"

    def test_other_attrs_fold_to_first(self, random_patch):
        """Attrs outside the kind never refuse a merge; the first wins."""
        pa1 = random_patch.update_attrs(foo="a", data_units="m")
        pa2 = random_patch.update_attrs(foo="b", data_units="s", bar="b")
        _, attrs = merge_compatible_coords_attrs(pa1, pa2)
        assert attrs.foo == "a"
        assert dc.get_quantity(attrs.data_units) == dc.get_quantity("m")
        # An attr only the second has is added.
        assert attrs.bar == "b"

    def test_first_without_units_stays_unitless(self, random_patch):
        """A first patch without units must not inherit the second's."""
        pa1 = random_patch.set_units(None)
        pa2 = random_patch.set_units("m")
        _, attrs = merge_compatible_coords_attrs(pa1, pa2)
        assert dc.get_quantity(attrs.data_units) is None

    def test_extra_coord(self, random_patch, random_patch_with_lat_lon):
        """Extra coords on both patch should end up in the merged patch."""
        new_coord = np.ones(random_patch.coord_shapes["time"])
        pa1 = random_patch.update_coords(new_time=("time", new_coord))
        pa2 = random_patch_with_lat_lon.update_coords(new_time=("time", new_coord))
        expected = set(pa1.coords.coord_map) | set(pa2.coords.coord_map)
        coords, attrs = merge_compatible_coords_attrs(pa1, pa2)
        assert set(coords.coord_map) == expected
        assert "coords" not in attrs.model_dump()

    def test_extra_attrs(self, random_patch):
        """Ensure extra attributes are added to patch."""
        patch = random_patch.update_attrs(new_attr=10)
        _, attrs = merge_compatible_coords_attrs(patch, random_patch)
        assert attrs.get("new_attr") == 10

    def test_different_dims(self, random_patch):
        """Ensure we can merge patches which share some dims but not all."""
        patch1 = random_patch
        patch2 = random_patch.rename_coords(time="money")
        coord, _ = merge_compatible_coords_attrs(patch1, patch2, dim_intersection=True)
        assert set(coord.dims) == (set(patch1.dims) | set(patch2.dims))


class TestConcatenate:
    """Tests for concatenating spools."""

    def test_deprecated_merge_patches_uses_spool_chunk(self, random_patch):
        """The deprecated merge helper should still merge compatible patches."""
        patch_1 = random_patch.update_attrs(history=[])
        time = patch_1.get_coord("time")
        patch_2 = patch_1.update_coords(time_min=time.max() + time.step).update_attrs(
            history=[]
        )

        with pytest.warns(DeprecationWarning, match="merge_patches is deprecated"):
            out = merge_patches([patch_1, patch_2])

        assert len(out) == 1
        assert out[0].get_coord("time").max() == patch_2.get_coord("time").max()

    def test_different_dims_raises(self, random_patch):
        """Patches can't be concated when they have different dims."""
        p1 = random_patch
        p2 = random_patch.rename_coords(time="money")
        msg = "Cannot concatenate"
        with pytest.raises(PatchCoordinateError, match=msg):
            concatenate_patches([p1, p2], time=None)

    def test_different_kind_skipped(self, random_patch):
        """A patch of another kind is left out per check_behavior."""
        other = random_patch.update_attrs(tag="other")
        msg = "not the same kind"
        with pytest.warns(UserWarning, match=msg):
            out = concatenate_patches([random_patch, other], time=None)
        assert len(out) == 1
        assert out[0].shape == random_patch.shape
        assert out[0].attrs.tag == random_patch.attrs.tag
        with pytest.raises(IncompatiblePatchError, match=msg):
            concatenate_patches(
                [random_patch, other], time=None, check_behavior="raise"
            )

    def test_kind_accumulates(self, random_patch):
        """A value a kept patch knows binds the patches after it."""
        blank = random_patch.update_attrs(tag="")
        a, b = random_patch.update_attrs(tag="a"), random_patch.update_attrs(tag="b")
        with pytest.warns(UserWarning, match="'tag': \\('a', 'b'\\)"):
            out = concatenate_patches([blank, a, b], time=None)
        assert out[0].shape[1] == 2 * random_patch.shape[1]
        assert out[0].attrs.tag == "a"

    def test_rejected_patch_binds_nothing(self, random_patch):
        """A patch dropped for its coordinates must not set the run's kind."""
        blank = random_patch.update_attrs(tag="")
        bad = random_patch.update_attrs(tag="a").update_coords(distance_min=5)
        good = random_patch.update_attrs(tag="b")
        with pytest.warns(UserWarning, match="not equal"):
            out = concatenate_patches([blank, bad, good], time=None)
        assert out[0].shape[1] == 2 * random_patch.shape[1]
        assert out[0].attrs.tag == "b"

    def test_kind_is_per_output(self, random_patch):
        """Each output's kind comes from its own members only."""
        blank = random_patch.update_attrs(tag="")
        a = random_patch.update_attrs(tag="a")
        out = concatenate_patches([blank, a], time=1)
        assert [x.attrs.tag for x in out] == ["", "a"]

    def test_nan_kind_value_is_filled(self, random_patch):
        """A NaN kind value is missing and takes the known value."""
        with config_context(patch_kind_attrs=("foo",)):
            nan, known = (
                random_patch.update_attrs(foo=np.nan),
                random_patch.update_attrs(foo="a"),
            )
            out = concatenate_patches([nan, known], time=None)
            assert out[0].attrs.foo == "a"

    def test_different_kind_skipped_before_dims(self, random_patch):
        """A different-kind patch is skipped whatever its dimensions."""
        other = random_patch.update_attrs(tag="other").rename_coords(time="money")
        with suppress_warnings(action="error"):
            out = concatenate_patches(
                [random_patch, other], time=None, check_behavior="ignore"
            )
        assert len(out) == 1
        assert out[0].shape == random_patch.shape

    def test_other_attrs_do_not_gate(self, random_patch):
        """Units differing still concatenate, silently."""
        other = random_patch.set_units("m/s").update_attrs(
            history=random_patch.attrs.history
        )
        with suppress_warnings(action="error"):
            out = concatenate_patches([random_patch, other], time=None)
        assert out[0].shape[1] == 2 * random_patch.shape[1]

    def test_history_divergence_warns(self, random_patch):
        """Splicing processed beside unprocessed data warns but proceeds."""
        other = random_patch.pass_filter(time=(None, 10))
        with pytest.warns(UserWarning, match="histories differ"):
            out = concatenate_patches([random_patch, other], time=None)
        assert out[0].shape[1] == 2 * random_patch.shape[1]
        assert not any("pass_filter" in x for x in out[0].attrs.history)

    def test_duplicate_patches_existing_dim(self, random_patch):
        """Ensure equal (but distinct) patches are concatenated together.

        Note: the same patch *instance* twice would collapse to one entry
        (spools have set semantics by patch identity); patch.new() mints
        a distinct instance with equal contents.
        """
        spool = dc.spool([random_patch, random_patch.new()])
        out = concatenate_patches(spool, time=None)
        assert len(out) == 1
        patch = out[0]
        time_coord = patch.get_coord("time")
        old_coord = random_patch.get_coord("time")
        assert len(time_coord) == 2 * len(old_coord)
        # Also ensure the new coord is just the old one repeated twice.
        val1, val2 = patch.get_array("time"), random_patch.get_array("time")
        assert np.all(val1[: len(val2)] == val2)
        assert np.all(val1[len(val2) :] == val2)

    def test_concatenate_ignores_private_attrs(self, random_patch):
        """Private ids should not block concatenation of compatible patches."""
        patch_1 = random_patch.update_attrs(_source_patch_key="one")
        shifted = (
            random_patch.coords.get_array("time") + random_patch.get_coord("time").step
        )
        patch_2 = random_patch.update_coords(time=shifted).update_attrs(
            _source_patch_key="two"
        )
        out = concatenate_patches([patch_1, patch_2], time=None)
        assert len(out) == 1
        assert out[0].attrs["_source_patch_key"] == "one"

    def test_different_lens(self, random_patch):
        """Ensure different lengths can be chunked on same dim."""
        patches = [random_patch] * 6
        out = concatenate_patches(patches, time=2)
        assert len(out) == 3

    def test_new_dim(self, random_patch):
        """Ensure we create a new dimension."""
        patches = [random_patch, random_patch]
        out = concatenate_patches(patches, zoolou=None)
        assert len(out) == 1
        patch = out[0]
        assert "zoolou" in patch.dims
        coord = patch.get_coord("zoolou")
        assert len(coord) == len(patches)

    def test_concat_chunk_to_new_dimension(self, random_patch):
        """Ensure the new dimension can be chunked by an int value."""
        # When new_dim = 1 it should only add a new dimension to each patch
        # and not change the original shape.
        # Distinct instances: identical instances would dedup to one.
        spool = dc.spool([random_patch] + [random_patch.new() for _ in range(5)])
        # Test for single values along new dimension
        new = spool.concatenate(new_dim=1)
        assert len(new) == len(spool)
        for patch in new:
            coord = patch.get_coord("new_dim")
            assert len(coord) == 1
        # Test for concatenating two patches together
        new = spool.concatenate(new_dim=2)
        assert len(new) == len(spool) // 2
        for patch in new:
            coord = patch.get_coord("new_dim")
            assert len(coord) == 2

    def test_spool_up(self, random_patch):
        """Ensure a patch is returned if the wrapper is used."""
        func = _spool_up(concatenate_patches)
        out = func([random_patch] * 3, time=None)
        assert isinstance(out, dc.BaseSpool)

    def test_new_dim_spool(self, random_patch):
        """Ensure a patch with new dim can be retrieved from spool."""
        spool = dc.spool([random_patch, random_patch])
        spool_concat = spool.concatenate(wave_rank=None)
        assert len(spool_concat) == 1
        patch = spool_concat[0]
        assert "wave_rank" in patch.dims
        assert len(patch.get_coord("wave_rank")) == len(spool)

    def test_patch_with_gap(self, random_patch):
        """Ensure a patch with a time gap still concats."""
        # Create a spool with patches that have a large gap
        time = random_patch.get_coord("time")
        one_hour = dc.to_timedelta64(3600)
        patch2 = random_patch.update_coords(time_min=time.max() + one_hour)
        spool = dc.spool([random_patch, patch2])
        # chunk rightfully wouldn't merge these patches, but concatenate will.
        merged = spool.concatenate(time=None)
        assert len(merged) == 1
        assert isinstance(merged[0], dc.Patch)

    def test_bad_kwargs(self, random_patch):
        """Ensure bad number of keywords raise."""
        msg = "Exactly one keyword argument"
        with pytest.raises(ParameterError, match=msg):
            concatenate_patches([random_patch])
        with pytest.raises(ParameterError, match=msg):
            concatenate_patches([random_patch], time=None, distance=None)

    def test_concat_along_non_dim(self, random_patch):
        """Ensure we can concat along non dims."""
        patches = [random_patch.mean("time") for _ in range(10)]
        out = concatenate_patches(patches, time=None)
        assert len(out) == 1
        patch = out[0]
        coord = patch.get_coord("time")
        assert len(coord) == 10

    def test_concat_different_sizes(self, random_patch):
        """Ensure coordinates with different sizes (along concat axis) work."""
        p1 = random_patch.select(time=(0, 10), samples=True)
        p2 = random_patch.select(time=(15, 20), samples=True)
        out = concatenate_patches([p1, p2], time=None)
        assert len(out) == 1
        patch = out[0]
        new_time = patch.get_array("time")
        old_times = np.concatenate([p1.get_array("time"), p2.get_array("time")])
        assert np.all(new_time == old_times)

    def test_concat_normal_with_non_dim(self, spool_with_non_coords):
        """Ensure normal and non-dim patches can be concatenated together."""
        old_arrays = [x.get_array("time") for x in spool_with_non_coords]
        old_array = np.concatenate(old_arrays)
        out = spool_with_non_coords.concatenate(time=None)
        assert len(out) == 1
        patch = out[0]
        new_array = patch.get_array("time")
        # The coordinates should be the same length
        assert sum([len(x) for x in old_arrays]) == len(new_array)
        # The array values should either both be NaN or nearly equal
        both_nan = pd.isnull(old_array) & pd.isnull(new_array)
        try:
            nearly_eq = np.isclose(old_array, new_array)
        except TypeError:
            nearly_eq = old_array == new_array
        assert np.all(both_nan | nearly_eq)

    def test_private_coords_dropped(self, random_patch):
        """Ensure private coords don't interfere with concat along new dim."""
        pa1 = random_patch.update_coords(_private_1=(None, np.array([1, 2, 3])))
        pa2 = random_patch.update_coords(_private_1=(None, np.array([2, 2, 2])))
        spool = dc.spool([pa1, pa2])
        out = spool.concatenate(time_new=None)
        assert len(out) == 1
        assert out[0].shape[-1] == 2

    def test_concat_dropped_coord(self, random_spool):
        """Ensure patches after dropping a coordinate can be concatenated together
        and the concatenated patch can have a new dimension.
        """
        sp = random_spool
        pa_list = []
        for pa in sp:
            pa_dft = pa.dft("time")
            cm = pa_dft.coords
            pa_dft_dropped_time = pa_dft.update(coords=cm.update(time=None))
            pa_list.append(pa_dft_dropped_time)
        sp_dft = dc.spool(pa_list)
        sp_concat = sp_dft.concatenate(time_min=None)
        pa_concat = sp_concat[0]
        assert pa_concat.shape[-1] == len(sp)
        assert "time_min" in pa_concat.dims


class TestStackPatches:
    """Tests for stacking (adding) spool content."""

    def test_stack_data(self):
        """
        Try the stack method on an example spool that has repeated
        copies of the same patch with different times. Check data.
        """
        # Grab the example spool which has repeats of the same patch
        # but with different time dimensions. Stack the patches.
        spool = dc.get_example_spool()
        stack_patch = stack_patches(spool, dim_vary="time")
        # We expect the sum/stack to be same as a multiple of
        # the first patch's data.
        baseline = float(len(spool)) * spool[0].data
        assert np.allclose(baseline, stack_patch.data)

    def test_same_dim_different_shape(self, random_spool):
        """Ensure when stack dimensions have different shape an error is raised."""
        # Create a spool with two patches, each with time dim but with different
        # lengths.
        patch1, patch2 = random_spool[:2]
        patch2 = patch2.select(time=(1, 30), samples=True)
        spool = dc.spool([patch1, patch2])
        # Check that warnings/exceptions are raised.
        msg = "Patches are not compatible"
        with pytest.raises(IncompatiblePatchError, match=msg):
            stack_patches(spool, dim_vary="time", check_behavior="raise")
        # Or a warning issued.
        with pytest.warns(UserWarning, match=msg):
            stack_patches(spool, dim_vary="time", check_behavior="warn")
        # Or the incompatible patch skipped without a word -- skipped
        # being the point: a quiet policy must not stack it anyway.
        with suppress_warnings(action="error"):
            quiet = stack_patches(spool, dim_vary="time", check_behavior="ignore")
        assert np.allclose(quiet.data, patch1.data)

    def test_retired_check_behavior_raises(self, random_spool):
        """None used to mean "ignore"; one spelling of it survives."""
        patch1, patch2 = random_spool[0], random_spool[1]
        patch2 = patch2.select(time=(1, 30), samples=True)
        spool = dc.spool([patch1, patch2])
        with pytest.raises(ParameterError, match="behavior must be one of"):
            stack_patches(spool, dim_vary="time", check_behavior=None)

    def test_retired_check_behavior_raises_on_compatible_patches(self, random_patch):
        """Acceptance must not wait for data which happens to disagree."""
        with pytest.raises(ParameterError, match="behavior must be one of"):
            check_dims(random_patch, random_patch, check_behavior=None)

    def test_different_dimensions(self, random_spool):
        """Tests for when the spool has patches with different dimensions."""
        first = random_spool[1]
        new_patch = random_spool[0].rename_coords(time="money")
        spool = dc.spool([first, new_patch])
        msg = "not compatible for merging"
        with pytest.warns(UserWarning, match=msg):
            out = stack_patches(spool, dim_vary="time", check_behavior="warn")
        # The mismatched patch is left out rather than summed in.
        assert np.allclose(out.data, first.data)

    def test_bad_dim_vary(self, random_spool):
        """Ensure when dim_vary is not in patch an error is raised."""
        with pytest.raises(PatchCoordinateError):
            stack_patches(random_spool, dim_vary="money")

    def test_different_kind_skipped(self, random_patch):
        """A patch of another kind is not summed in."""
        other = random_patch.update_attrs(tag="other")
        msg = "not the same kind"
        with pytest.warns(UserWarning, match=msg):
            out = stack_patches([random_patch, other], check_behavior="warn")
        assert np.allclose(out.data, random_patch.data)
        with pytest.raises(IncompatiblePatchError, match=msg):
            stack_patches([random_patch, other], check_behavior="raise")

    def test_rejected_patch_binds_nothing(self, random_patch):
        """A patch dropped for its coordinates must not set the stack's kind."""
        blank = random_patch.update_attrs(tag="")
        bad = random_patch.update_attrs(tag="a").update_coords(distance_min=5)
        good = random_patch.update_attrs(tag="b")
        with pytest.warns(UserWarning, match="not equal"):
            out = stack_patches([blank, bad, good])
        assert np.allclose(out.data, 2 * random_patch.data)
        assert out.attrs.tag == "b"

    def test_stack_coords(self):
        """
        Try the stack method on an example spool that has repeated
        copies of the same patch with different times. Check coords.
        """
        # Grab the example spool which has repeats of the same patch
        # but with different time dimensions. Stack the patches.
        spool = dc.get_example_spool()
        stack_patch = stack_patches(spool, dim_vary="time")
        # check that 'time' coordinates has same step as original patch
        time_coords = stack_patch.coords.coord_map["time"]
        orig_time_coords = spool[0].coords.coord_map["time"]
        assert time_coords.step == orig_time_coords.step
        # check that distance coordinates are the same
        dist_coords = stack_patch.coords.coord_map["distance"]
        orig_dist_coords = spool[0].coords.coord_map["distance"]
        assert dist_coords.start == orig_dist_coords.start
        assert dist_coords.stop == orig_dist_coords.stop
        assert dist_coords.step == orig_dist_coords.step
        assert dist_coords.units == orig_dist_coords.units


class TestGetPatchName:
    """Tests for getting the default name from patch sources."""

    def test_single_patch(self, random_patch):
        """Ensure the random patch can have a name."""
        out = get_patch_names(random_patch)
        assert isinstance(out, pd.Series)
        assert len(out) == 1

    def test_spool(self, random_spool):
        """Ensure names can be generated from a spool as well."""
        out = get_patch_names(random_spool)
        assert len(out) == len(random_spool)

    def test_empty(self):
        """Ensure an empty thing returns a series of the right type."""
        out = get_patch_names([])
        assert isinstance(out, pd.Series)

    def test_annotations_resolve(self):
        """
        The docs renderer resolves annotations at runtime and falls back to
        raw strings for the whole signature if any name is undefined.
        """
        assert get_type_hints(get_patch_names)

    def test_name_column_exists(self, random_spool):
        """If the name or path field already exist this should be used."""
        df = random_spool.get_contents().assign(name=lambda x: np.arange(len(x)))
        # This should convert to string type and return.
        names = get_patch_names(df)
        assert np.all(names.values == np.arange(len(df)).astype(str))

    def test_path_column(self, random_directory_spool):
        """Ensure the path column works."""
        names = get_patch_names(random_directory_spool)
        df = random_directory_spool.get_contents()
        paths = df["source_path"].str.split("/")
        expected = pd.Series([x[-1].split(".")[0] for x in paths])
        assert np.all(names == expected)

    def test_path_column_leave_extension(self, random_directory_spool):
        """If extension is false the file extension should remain."""
        names = get_patch_names(random_directory_spool, strip_extension=False)
        assert "." in names.iloc[0]

    def test_mixed_path_sources_use_metadata(self, random_spool):
        """Mixed real and memory paths consistently use metadata names."""
        df = random_spool.get_contents().iloc[:2].copy()
        df["source_path"] = ["/data/real_file.h5", "memory://registry/patch"]
        names = get_patch_names(df)
        assert names.iloc[0] != "real_file"
        assert names.iloc[1] != "patch"

    def test_multiple_coord_fields(self, random_spool):
        """Naming on more than one coordinate flattens the min/max fields."""
        # drop path so the coordinate-based naming branch is exercised
        df = random_spool.get_contents().drop(columns=["source_path"], errors="ignore")
        names = get_patch_names(df, coords=("time", "distance"))
        assert len(names) == len(df)
        assert names.str.len().gt(0).all()


class TestSwapKwargsDimToAxis:
    """Tests for swap_kwargs_dim_to_axis function."""

    def test_with_multiple_dims(self, random_patch):
        """Test swap_kwargs_dim_to_axis function with multiple dimensions."""
        # Test with list of dimensions
        kwargs = {"dim": ["time", "distance"]}
        new_kwargs = swap_kwargs_dim_to_axis(random_patch, kwargs)

        expected_axes = [
            random_patch.get_axis("time"),
            random_patch.get_axis("distance"),
        ]
        assert new_kwargs["axis"] == expected_axes
        assert "dim" not in new_kwargs

    def test_no_dim(self, random_patch):
        """Test swap_kwargs_dim_to_axis function with no dim parameter."""
        # Test with no dim parameter
        kwargs = {"other": "value"}
        new_kwargs = swap_kwargs_dim_to_axis(random_patch, kwargs)

        # Should be unchanged
        assert new_kwargs == kwargs

    def test_with_none_dim(self, random_patch):
        """Test swap_kwargs_dim_to_axis function with dim=None."""
        # Test with None dim parameter
        kwargs = {"dim": None}
        new_kwargs = swap_kwargs_dim_to_axis(random_patch, kwargs)

        # Should remove dim and not add axis
        assert "dim" not in new_kwargs
        assert "axis" not in new_kwargs

    def test_single_string_dim(self, random_patch):
        """Test swap_kwargs_dim_to_axis with single string dimension."""
        kwargs = {"dim": "time", "dtype": None}
        new_kwargs = swap_kwargs_dim_to_axis(random_patch, kwargs)

        expected_axis = random_patch.get_axis("time")
        assert new_kwargs["axis"] == expected_axis
        assert "dim" not in new_kwargs
        assert new_kwargs["dtype"] is None

    def test_unknown_dim_raises(self, random_patch):
        """Bad dimension should raise ParameterError."""
        # Patch has no dimension foo.
        kwargs = {"dim": "foo"}

        with pytest.raises(ParameterError, match="Dimension 'foo' not found"):
            swap_kwargs_dim_to_axis(random_patch, kwargs)

    def test_unknown_dim_in_list_raises(self, random_patch):
        """Bad dimension in list should raise ParameterError."""
        # One valid, one invalid dimension
        kwargs = {"dim": ["time", "invalid_dim"]}

        with pytest.raises(ParameterError, match="Dimension 'invalid_dim' not found"):
            swap_kwargs_dim_to_axis(random_patch, kwargs)


class TestGetPatchWindowSize:
    """Tests for the get_patch_window_size function."""

    @pytest.fixture()
    def simple_patch(self):
        """Create a simple patch for testing."""
        patch = dc.get_example_patch()
        return patch.update_coords(time_step=0.2)  # Make windows reasonable

    def test_basic_window_size(self, simple_patch):
        """Test basic window size calculation."""
        size = get_patch_window_size(simple_patch, {"time": 0.6})
        assert isinstance(size, tuple)
        assert len(size) == simple_patch.data.ndim
        # Find which axis corresponds to time
        time_axis = simple_patch.dims.index("time")
        distance_axis = simple_patch.dims.index("distance")
        assert size[time_axis] > 1  # time dimension should have window > 1
        assert size[distance_axis] == 1  # distance dimension should be 1

    def test_multiple_dimensions(self, simple_patch):
        """Test window size with multiple dimensions."""
        size = get_patch_window_size(simple_patch, {"time": 0.6, "distance": 3.0})
        time_axis = simple_patch.dims.index("time")
        distance_axis = simple_patch.dims.index("distance")
        assert size[time_axis] > 1  # time dimension
        assert size[distance_axis] > 1  # distance dimension

    def test_samples_true(self, simple_patch):
        """Test with samples=True parameter."""
        size = get_patch_window_size(simple_patch, {"time": 5}, samples=True)
        time_axis = simple_patch.dims.index("time")
        assert size[time_axis] == 5

    def test_require_odd_true_samples_false(self, simple_patch):
        """Test require_odd=True with samples=False adjusts even sizes."""
        # Use a value that would give even samples
        coord = simple_patch.get_coord("time")
        step = coord.step
        even_value = step * 4  # Should give 4 samples

        size = get_patch_window_size(
            simple_patch, {"time": even_value}, samples=False, require_odd=True
        )
        # Should be adjusted to 5 (next odd number)
        time_axis = simple_patch.dims.index("time")
        assert size[time_axis] % 2 == 1

    def test_require_odd_true_samples_true_even_raises(self, simple_patch):
        """Test require_odd=True with samples=True raises for even sizes."""
        with pytest.raises(ParameterError, match="windows must be odd"):
            get_patch_window_size(
                simple_patch, {"time": 4}, samples=True, require_odd=True
            )

    def test_require_odd_true_samples_true_odd_passes(self, simple_patch):
        """Test require_odd=True with samples=True passes for odd sizes."""
        size = get_patch_window_size(
            simple_patch, {"time": 5}, samples=True, require_odd=True
        )
        time_axis = simple_patch.dims.index("time")
        assert size[time_axis] == 5

    def test_min_samples_validation(self, simple_patch):
        """Test minimum samples validation."""
        with pytest.raises(ParameterError, match="at least 3 samples"):
            get_patch_window_size(
                simple_patch, {"time": 2}, samples=True, min_samples=3
            )

    def test_warn_above_warning(self, simple_patch):
        """Test warning for large window sizes."""
        with pytest.warns(UserWarning, match="Large window size.*may result in slow"):
            get_patch_window_size(
                simple_patch, {"time": 15}, samples=True, warn_above=10
            )

    def test_no_warning_under_threshold(self, simple_patch):
        """Test no warning for window sizes under threshold."""
        with suppress_warnings(action="error"):
            # This should not raise (no warning)
            size = get_patch_window_size(
                simple_patch, {"time": 5}, samples=True, warn_above=10
            )
            time_axis = simple_patch.dims.index("time")
            assert size[time_axis] == 5

    def test_empty_kwargs(self, simple_patch):
        """Test with empty kwargs returns all ones."""
        size = get_patch_window_size(simple_patch, {})
        assert all(s == 1 for s in size)
        assert len(size) == simple_patch.data.ndim

    def test_invalid_dimension_raises(self, simple_patch):
        """Test invalid dimension name raises error."""
        with pytest.raises(ParameterError):
            get_patch_window_size(simple_patch, {"invalid_dim": 5})

    def test_non_evenly_sampled_raises(self, simple_patch):
        """Test non-evenly sampled coordinate raises error."""
        # Create a non-evenly sampled coordinate
        time_size = simple_patch.data.shape[simple_patch.dims.index("time")]
        time_vals = np.array([0.0, 0.1, 0.3, 0.7, 1.5])  # Non-uniform spacing
        # Take enough values to match the patch size
        if len(time_vals) < time_size:
            # Extend with more irregular values
            extra_vals = np.linspace(2.0, 10.0, time_size - len(time_vals))
            time_vals = np.concatenate([time_vals, extra_vals])
        irregular_patch = simple_patch.update_coords(time=time_vals[:time_size])

        with pytest.raises(CoordError):
            get_patch_window_size(irregular_patch, {"time": 0.5})


class TestGetWindowAxisStep:
    """Tests for getting window size, axis, and step."""

    window = 16
    step = 8

    def test_apply_with_overlap(self, random_patch):
        """Ensure overlap is converted to the correct step size."""
        coord = random_patch.get_coord("distance")
        window = self.window * coord.step
        overlap = (self.window - self.step) * coord.step
        out = get_window_axis_step(random_patch, distance=window, overlap=overlap)
        assert out == (self.window, random_patch.get_axis("distance"), self.step)

    def test_apply_with_percent_overlap(self, random_patch):
        """Ensure percent overlap is supported."""
        coord = random_patch.get_coord("distance")
        window = self.window * coord.step
        out = get_window_axis_step(random_patch, distance=window, overlap=50 * percent)
        assert out == (self.window, random_patch.get_axis("distance"), self.step)

    def test_apply_with_percent_overlap_and_samples(self, random_patch):
        """Ensure percent overlap works when samples=True."""
        out = get_window_axis_step(
            random_patch, distance=self.window, overlap=50 * percent, samples=True
        )
        assert out == (self.window, random_patch.get_axis("distance"), self.step)

    def test_negative_overlap_raises(self, random_patch):
        """Ensure negative overlap is rejected."""
        step = random_patch.get_coord("distance").step
        msg = "overlap must be non-negative"
        with pytest.raises(ParameterError, match=msg):
            get_window_axis_step(
                random_patch, distance=self.window * step, overlap=-step
            )

    def test_invalid_percent_overlap_raises(self, random_patch):
        """Ensure percent overlap must be between 0 and 100."""
        msg = "Percentage must be between 0 and 100"
        with pytest.raises(ParameterError, match=msg):
            get_window_axis_step(
                random_patch, distance=self.window, overlap=101 * percent, samples=True
            )

    def test_complete_overlap_raises(self, random_patch):
        """Ensure complete overlap is rejected."""
        msg = "Window step must be greater than zero"
        with pytest.raises(ParameterError, match=msg):
            get_window_axis_step(
                random_patch, distance=self.window, overlap=100 * percent, samples=True
            )

    def test_overlap_larger_than_window_raises(self, random_patch):
        """Ensure overlap larger than window is rejected."""
        msg = "Window step must be greater than zero"
        with pytest.raises(ParameterError, match=msg):
            get_window_axis_step(
                random_patch,
                distance=self.window,
                overlap=self.window + 1,
                samples=True,
            )

    def test_step_and_overlap_raises(self, random_patch):
        """Ensure step and overlap are mutually exclusive."""
        step = random_patch.get_coord("distance").step
        msg = "step and overlap are mutually exclusive"
        with pytest.raises(ParameterError, match=msg):
            get_window_axis_step(
                random_patch,
                distance=self.window * step,
                step=self.step * step,
                overlap=50 * percent,
            )

    def test_none_overlap_matches_default(self, random_patch):
        """Ensure overlap=None preserves default window/step behavior."""
        step = random_patch.get_coord("distance").step
        out = get_window_axis_step(
            random_patch, distance=self.window * step, overlap=None
        )
        assert out == (self.window, random_patch.get_axis("distance"), None)


class TestForcePatchMergeOverlap:
    """_force_patch_merge tolerates complete-envelope overlap (keep first)."""

    def test_complete_overlap_keeps_first(self, random_patch):
        """Identical envelopes merge to the first patch."""
        twin = random_patch.new()
        infos = []
        for patch in (random_patch, twin):
            info = patch.coords._get_dim_summary()
            info["patch"] = patch
            info["dims"] = ",".join(patch.dims)
            infos.append(info)
        out = _force_patch_merge(infos, merge_kwargs={})
        assert len(out) == 1
        assert out[0]["patch"] is random_patch


class TestDottedPatchNames:
    """A acquisition_key puts dots in the name it generates."""

    def test_name_round_trips_through_a_file(self, tmp_path, random_patch):
        """Writing a patch under its own name and re-reading keeps the name."""
        patch = random_patch.update_attrs(acquisition_key="DAS.R2D1..RAW")
        name = patch.get_patch_name()
        assert "DAS.R2D1..RAW" in name
        patch.io.write(tmp_path / f"{name}.h5", "dasdae")
        assert get_patch_names(dc.spool(tmp_path).update()).iloc[0] == name
