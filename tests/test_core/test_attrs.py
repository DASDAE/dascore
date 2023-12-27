"""Tests for Patch attrs modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.attrs import (
    PatchAttrs,
    combine_patch_attrs,
    decompose_attrs,
    merge_compatible_coords_attrs,
)
from dascore.core.coords import CoordSummary, get_coord
from dascore.exceptions import (
    AttributeMergeError,
    IncompatiblePatchError,
)
from dascore.utils.downloader import fetcher
from dascore.utils.misc import register_func

MORE_COORDS_ATTRS = []


@pytest.fixture(scope="class")
def random_summary(random_patch) -> PatchAttrs:
    """Return the summary of the random patch."""
    return PatchAttrs.model_validate(random_patch.attrs.model_dump())


@pytest.fixture(scope="class")
def random_attrs(random_patch) -> PatchAttrs:
    """Return the summary of the random patch."""
    return random_patch.attrs


@pytest.fixture(scope="session")
@register_func(MORE_COORDS_ATTRS)
def attrs_coords_1() -> PatchAttrs:
    """Add non-standard coords to attrs."""
    attrs = {"depth_min": 10.0, "depth_max": 12.0, "another_name": "FooBar"}
    out = PatchAttrs(**attrs)
    assert "depth" in out.coords
    return out


@pytest.fixture(scope="session")
@register_func(MORE_COORDS_ATTRS)
def attrs_coords_2() -> PatchAttrs:
    """Add non-standard coords to attrs."""
    coords = {"depth": {"min": 10.0, "max": 12.0}}
    attrs = {"coords": coords, "another_name": "FooBar"}
    return PatchAttrs(**attrs)


@pytest.fixture(scope="session", params=MORE_COORDS_ATTRS)
def more_coords_attrs(request) -> PatchAttrs:
    """
    Meta fixture for attributes with extra coords.
    These are initiated in different ways but should be identical.
    """
    return request.getfixturevalue(request.param)


class TestPatchAttrs:
    """Basic tests on patch attributes."""

    def test_get(self, random_attrs):
        """Ensure get returns existing values."""
        out = random_attrs.get("time_min")
        assert out == random_attrs.time_min

    def test_get_existing_key(self, random_attrs):
        """Ensure get returns existing values."""
        out = random_attrs.get("time_min")
        assert out == random_attrs.time_min

    def test_get_no_key(self, random_attrs):
        """Ensure missing keys return default value."""
        out = random_attrs.get("not_a_key", 1)
        assert out == 1

    def test_immutable(self, random_attrs):
        """Ensure random_attrs is faux-immutable."""
        with pytest.raises(ValidationError, match="Instance is frozen"):
            random_attrs.bob = 1
        with pytest.raises(ValidationError, match="Instance is frozen"):
            random_attrs["bob"] = 1

    def test_coords_with_coord_keys(self):
        """Ensure coords with base keys work."""
        coords = {"distance": get_coord(data=np.arange(100))}
        out = PatchAttrs(**{"coords": coords})
        assert out.coords
        assert "distance" in out.coords
        for _name, val in out.coords.items():
            assert isinstance(val, CoordSummary)

    def test_coords_with_coord_manager(self, random_patch):
        """Ensure coords with a coord manager works."""
        cm = random_patch.coords
        out = PatchAttrs(**{"coords": cm})
        assert out.coords
        assert set(cm.coord_map) == set(out.coords)
        assert out.dims == ",".join(cm.dims)

    def test_coords_are_coord_summary(self, more_coords_attrs):
        """All the coordinates should be Coordinate Summarys not dict."""
        for _, coord_sum in more_coords_attrs.coords.items():
            assert isinstance(coord_sum, CoordSummary)

    def test_access_min_max_step_etc(self, more_coords_attrs):
        """Ensure min, max, step etc can be accessed for new coordinate."""
        expected_attrs = ["_min", "_max", "_step", "_units", "_dtype"]
        for eat in expected_attrs:
            attr_name = "depth" + eat
            assert hasattr(more_coords_attrs, attr_name)
            val = getattr(more_coords_attrs, attr_name)
            if eat == "min":
                assert val == 10.0
            if eat == "max":
                assert val == 12.0

    def test_deprecated_d_set(self):
        """Ensure setting attributes d_whatever is deprecated."""
        with pytest.warns(DeprecationWarning):
            PatchAttrs(time_min=1, time_max=10, d_time=10)

    def test_deprecated_d_get(self, random_attrs):
        """Access attr.d_{whatever} is deprecated."""
        with pytest.warns(DeprecationWarning):
            _ = random_attrs.d_time

    def test_access_coords(self, more_coords_attrs):
        """Ensure coordinates can be accessed as well."""
        assert "depth" in more_coords_attrs.coords
        assert more_coords_attrs.coords["depth"].min == more_coords_attrs.depth_min

    def test_extra_attrs_not_in_dump(self, more_coords_attrs):
        """When using the extra attrs, they shouldn't show up in the dump."""
        dump = more_coords_attrs.model_dump()
        not_expected = {"depth_min", "depth_max", "depth_step"}
        assert not_expected.isdisjoint(set(dump))

    def test_extra_attrs_not_in_dump_random_attrs(self, random_attrs):
        """When using the extra attrs, they shouldn't show up in the dump."""
        dump = random_attrs.model_dump()
        not_expected = {"time_min", "time_max", "time_step"}
        assert not_expected.isdisjoint(set(dump))

    def test_supports_extra_attrs(self):
        """The attr dict should allow extra attributes."""
        out = PatchAttrs(bob="doesnt", bill_min=12, bob_max="2012-01-12")
        assert out.bob == "doesnt"
        assert out.bill_min == 12

    def test_flat_dump(self, more_coords_attrs):
        """Ensure flat dump flattens out the coords."""
        out = more_coords_attrs.flat_dump()
        expected = {
            "depth_min",
            "depth_max",
            "depth_step",
            "depth_units",
            "depth_dtype",
        }
        assert set(out).issuperset(expected)

    def test_flat_dump_coords(self, more_coords_attrs):
        """Ensure flat dim with dim_tuple works."""
        attrs = more_coords_attrs
        out = attrs.flat_dump(dim_tuple=True)
        assert "depth" in out
        depth = attrs.coords["depth"]
        dep_min, dep_max = depth.min, depth.max
        assert out["depth"] == (dep_min, dep_max)

    def test_coords_to_coord_summary(self):
        """Coordinates included in coords should be converted to coord summary."""
        out = {
            "station": "01",
            "coords": {
                "time": get_coord(start=0, stop=10, step=1),
                "distance": get_coord(start=10, stop=100, step=1, units="m"),
            },
        }
        attr = dc.PatchAttrs(**out)
        assert attr.dims == ",".join(("time", "distance"))
        for name, coord in attr.coords.items():
            assert isinstance(coord, CoordSummary)

    def test_items(self, random_patch):
        """Ensure items works like a dict."""
        attrs = random_patch.attrs
        out = dict(attrs.items())
        assert out == attrs.model_dump()


class TestSummaryAttrs:
    """Tests for summarizing a schema."""

    def test_attrs_reconstructed(self, random_patch, random_summary):
        """Ensure all the expected attrs are extracted."""
        summary1 = dict(random_summary)
        attrs = dict(random_patch.attrs)
        common_keys = set(summary1) & set(attrs)
        for key in common_keys:
            assert summary1[key] == attrs[key]

    def test_can_jsonize(self, random_summary):
        """Ensure the summary can be converted to json."""
        json = random_summary.model_dump_json()
        assert isinstance(json, str)

    def test_can_roundrip(self, random_summary):
        """Ensure json can be round-tripped."""
        json = random_summary.model_dump_json()
        random_summary2 = PatchAttrs.model_validate_json(json)
        assert random_summary2 == random_summary

    def test_from_dict(self, random_attrs):
        """Test new method for more intuitive init."""
        out = PatchAttrs.from_dict(random_attrs)
        assert out == random_attrs
        new_dict = dict(random_attrs)
        new_dict["data_units"] = "m/s"
        out = PatchAttrs.from_dict(new_dict)
        assert isinstance(out, PatchAttrs)


class TestRenameDimension:
    """Ensure rename dimension works."""

    def test_simple_rename(self, random_attrs):
        """Ensure renaming a dimension works."""
        attrs = random_attrs
        new_name = "money"
        time_ind = attrs.dim_tuple.index("time")
        out = attrs.rename_dimension(time=new_name)
        assert new_name in out.dims
        assert out.dim_tuple[time_ind] == new_name
        assert len(out.dim_tuple) == len(attrs.dim_tuple)

    def test_empty_rename(self, random_attrs):
        """Passing no kwargs should return same attrs."""
        attrs = random_attrs.rename_dimension()
        assert attrs == random_attrs


class TestDropPrivate:
    """Tests for dropping private attrs."""

    def test_simple_drop(self):
        """Ensure private attrs are removed after operation."""
        attrs = PatchAttrs(_private1=1, extra_attr=2).drop_private()
        attr_dict = dict(attrs)
        assert "_private1" not in attr_dict
        assert "extra_attr" in attr_dict


class TestMisc:
    """Misc small tests."""

    def test_schema_deprecated(self):
        """Ensure schema module emits deprecation warning."""
        with pytest.warns(DeprecationWarning):
            from dascore.core.schema import PatchAttrs  # noqa

    def test_get_attrs_non_dim_coordinates(self, random_patch_with_lat_lon):
        """
        Ensure only dims show up in dims even when coord manager has many
        coordinates.
        """
        patch = random_patch_with_lat_lon
        cm = patch.coords
        attrs = dc.PatchAttrs(coords=cm)
        assert attrs.dim_tuple == cm.dims


class TestUpdateAttrs:
    """Tests for updating attributes."""

    def test_attrs_can_update(self, random_attrs):
        """Ensure attributes can update coordinates."""
        attrs = random_attrs.update(distance_units="miles")
        expected = dc.get_quantity("miles")
        assert dc.get_quantity(attrs.coords["distance"].units) == expected


class TestMergeAttrs:
    """Tests for merging patch attrs."""

    def test_empty(self):
        """Empty PatchAttrs should work in all cases."""
        pa1, pa2 = PatchAttrs(), PatchAttrs()
        assert isinstance(combine_patch_attrs([pa1, pa2]), PatchAttrs)
        out = combine_patch_attrs([pa1, pa2], drop_attrs="history")
        assert isinstance(out, PatchAttrs)

    def test_empty_with_coord_raises(self):
        """Attributes which don't have specified coord should raise."""
        pa1, pa2 = PatchAttrs(), PatchAttrs()
        match = "Failed to combine"
        with pytest.raises(AttributeMergeError, match=match):
            combine_patch_attrs([pa1, pa2], "time")
        with pytest.raises(AttributeMergeError, match=match):
            combine_patch_attrs([pa1, pa2], "distance")

    def test_simple_merge(self):
        """Happy path, simple merge."""
        pa1 = PatchAttrs(distance_min=1, distance_max=10)
        pa2 = PatchAttrs(distance_min=11, distance_max=20)
        merged = combine_patch_attrs([pa1, pa2], "distance")
        # the names of attrs should be identical before/after merge
        assert set(dict(merged)) == set(dict(pa1))
        assert merged.distance_max == pa2.distance_max
        assert merged.distance_min == pa1.distance_min

    def test_drop(self):
        """Ensure drop_attrs does its job."""
        pa1 = PatchAttrs(history=["a", "b"])
        pa2 = PatchAttrs()
        msg = "the following non-dim attrs are not equal"
        with pytest.raises(AttributeMergeError, match=msg):
            combine_patch_attrs([pa1, pa2])
        out = combine_patch_attrs([pa1, pa2], drop_attrs="history")
        assert isinstance(out, PatchAttrs)

    def test_conflicts(self):
        """Ensure when non-dim fields aren't equal merge raises."""
        pa1 = PatchAttrs(tag="bob", another=2, same=42)
        pa2 = PatchAttrs(tag="bob", another=2, same=42)
        pa3 = PatchAttrs(another=1, same=42, different=10)
        msg = "the following non-dim attrs are not equal"
        with pytest.raises(AttributeMergeError, match=msg):
            combine_patch_attrs([pa1, pa2, pa3])

    def test_missing_coordinate(self):
        """When one model has a missing coord it should just get dropped."""
        pa1 = PatchAttrs(bob_min=10, bob_max=12)
        pa2 = PatchAttrs(bob_min=13)
        out = combine_patch_attrs([pa1, pa2], coord_name="bob")
        assert out == pa1

    def test_drop_conflicts(self, random_patch):
        """Ensure when non-dim fields aren't equal, but are defined, they drop."""
        pa1 = random_patch.attrs.update(tag="bill", station="UU")
        pa2 = random_patch.attrs.update(station="TA", tag="bob")
        out = combine_patch_attrs([pa1, pa2], coord_name="time", conflicts="drop")
        defaults = PatchAttrs()
        assert isinstance(out, PatchAttrs)
        assert out.tag == defaults.tag
        assert out.station == defaults.station

    def test_keep_disjoint_values(self, random_patch):
        """Ensure when disjoint values should be kept they are."""
        random_attrs = random_patch.attrs
        attrs1 = random_attrs.update(jazz_hands=1984)
        out = combine_patch_attrs([attrs1, random_attrs], conflicts="keep_first")
        assert out.jazz_hands == 1984

    def test_unequal_raises(self, random_attrs):
        """When attrs have unequal coords it should raise an error."""
        attr1 = random_attrs
        attr2 = attr1.update(distance_units="miles")
        match = "Cant merge patch attrs"
        with pytest.raises(AttributeMergeError, match=match):
            combine_patch_attrs([attr1, attr2], coord_name="distance")


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
        match = "their dimensions are not equal"
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
        """Extra coords on both patch should end up in the merged."""
        new_coord = np.ones(random_patch.coord_shapes["time"])
        pa1 = random_patch.update_coords(new_time=("time", new_coord))
        pa2 = random_patch_with_lat_lon
        expected = set(pa1.coords.coord_map) & set(pa2.coords.coord_map)
        coords, attrs = merge_compatible_coords_attrs(pa1, pa2)
        assert set(coords.coord_map) == expected
        assert set(attrs.coords) == expected

    def test_extra_attrs(self, random_patch):
        """Ensure extra attributes are added to patch."""
        patch = random_patch.update_attrs(new_attr=10)
        coords, attrs = merge_compatible_coords_attrs(patch, random_patch)
        assert attrs.get("new_attr") == 10


class TestDecomposeAttrs:
    """Tests for decomposing attributes."""

    @pytest.fixture(scope="class")
    def scanned_attrs(self):
        """Scan all the attrs in the dascore cache, return them."""
        return dc.scan(fetcher.path)

    @pytest.fixture(scope="class")
    def decomposed_attrs(self, scanned_attrs):
        """Decompose attrs into dicts of lists."""
        return decompose_attrs(scanned_attrs)

    def test_only_datetime(self, decomposed_attrs):
        """Ensure all times are datetime64."""
        time_dtypes = decomposed_attrs["dims"]["time"]
        assert set(time_dtypes) == {"datetime64"}


class TestGetAttrSummary:
    """Test getting dataframe of summary info."""

    def test_summary(self, random_attrs):
        """Ensure a dataframe is returned."""
        out = random_attrs.get_summary_df()
        assert isinstance(out, pd.DataFrame)
