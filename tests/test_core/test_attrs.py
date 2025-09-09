"""Tests for Patch attrs modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.attrs import (
    PatchAttrs,
)
from dascore.core.coords import CoordSummary, get_coord
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

    def test_dims_match_attrs(self, random_patch):
        """Ensure the dims from patch attrs matches patch dims."""
        pat = random_patch.rename_coords(distance="channel")
        assert pat.dims == pat.attrs.dim_tuple


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

    def test_simple_drop_private(self):
        """Ensure private attrs are removed after operation."""
        attrs = PatchAttrs(_private1=1, extra_attr=2).drop_private()
        attr_dict = dict(attrs)
        assert "_private1" not in attr_dict
        assert "extra_attr" in attr_dict


class TestDrop:
    """Tests for dropping attrs."""

    def test_simple_drop(self):
        """Ensure a single attr can be dropped."""
        attrs = PatchAttrs(bob=1, bill=2, sue="Z")
        new = dict(attrs.drop("bob", "bill"))
        assert "bob" not in new and "bill" not in new
        assert "sue" in new


class TestMisc:
    """Misc small tests."""

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

    def test_update_from_coords(self, random_patch):
        """Ensure attrs.update updates dims from coords."""
        attrs = random_patch.attrs
        new_patch = random_patch.rename_coords(distance="channel")
        new = attrs.update(coords=new_patch.coords)
        assert new.dim_tuple == new_patch.dims

    def test_update_coord_summary(self, random_patch):
        """Ensure updating attrs updates the summary."""
        coords, _ = random_patch.coords.drop_coords("distance")
        attr = random_patch.attrs.update(coords=coords)
        coord_summary = attr.coords
        # Distance should have been dropped from the coord summary.
        assert set(coord_summary) == set(coords.coord_map)


class TestGetAttrSummary:
    """Test getting dataframe of summary info."""

    def test_summary(self, random_attrs):
        """Ensure a dataframe is returned."""
        out = random_attrs.get_summary_df()
        assert isinstance(out, pd.DataFrame)
