"""Tests for PatchAttrs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coords import get_coord
from dascore.exceptions import PatchAttributeError


@pytest.fixture(scope="class")
def random_summary(random_patch) -> PatchAttrs:
    """Return attrs reconstructed from pure patch attrs."""
    return PatchAttrs.model_validate(random_patch.attrs.model_dump())


@pytest.fixture(scope="class")
def random_attrs(random_patch) -> PatchAttrs:
    """Return patch attrs view."""
    return random_patch.attrs


class TestPatchAttrs:
    """Basic tests on patch attributes."""

    def test_get_existing_key(self, random_attrs):
        """Ensure get returns existing values."""
        assert random_attrs.get("tag") == random_attrs.tag

    def test_get_no_key(self, random_attrs):
        """Ensure missing keys return default value."""
        assert random_attrs.get("not_a_key", 1) == 1

    def test_immutable(self):
        """PatchAttrs instances remain frozen."""
        attrs = PatchAttrs(tag="bob")
        with pytest.raises(ValidationError, match="Instance is frozen"):
            attrs.tag = "bill"

    def test_model_validate_existing_instance(self):
        """Model validation should accept existing PatchAttrs instances."""
        attrs = PatchAttrs(tag="bob")
        assert PatchAttrs.model_validate(attrs) == attrs

    def test_model_validate_non_mapping(self):
        """Non-mapping inputs should pass through to normal model validation."""
        with pytest.raises(ValidationError):
            PatchAttrs.model_validate(1)

    def test_rejects_coord_keys(self):
        """Unknown flat keys remain allowed at raw PatchAttrs construction."""
        out = PatchAttrs(time_min=1, time_max=10)
        assert out.time_min == 1
        assert out.time_max == 10

    def test_rejects_coords_mapping(self):
        """Nested coords are rejected as well."""
        with pytest.raises(ValueError, match="no longer accepts coordinate metadata"):
            PatchAttrs(coords={"time": {"min": 0, "max": 1}})

    def test_rejects_coords_key(self):
        """String coords values are allowed as plain extra attrs."""
        out = PatchAttrs(coords="not-valid")
        assert out.coords == "not-valid"

    def test_rejects_coord_manager(self, random_patch):
        """CoordManager input should be rejected."""
        with pytest.raises(ValueError, match="no longer accepts coordinate metadata"):
            PatchAttrs(coords=random_patch.coords)

    def test_ignores_dims(self):
        """Dimensions are ignored during raw construction normalization."""
        out = PatchAttrs(dims="time,distance")
        assert "dims" not in out.model_dump()

    def test_extra_attrs_supported(self):
        """Non-coordinate extras should still be supported."""
        out = PatchAttrs(bob="doesnt", bill_min=12)
        assert out.bob == "doesnt"
        assert out.bill_min == 12

    def test_extra_attrs_not_in_dump_random_attrs(self, random_attrs):
        """Coord summaries should not be present in attrs dumps."""
        dump = random_attrs.model_dump()
        not_expected = {"time_min", "time_max", "time_step"}
        assert not_expected.isdisjoint(set(dump))

    def test_flat_dump_matches_model_dump(self, random_patch):
        """Patch summaries flatten attrs and coordinate summaries."""
        out = random_patch.summary.flat_dump()
        assert out["time_min"] == random_patch.coords.min("time")
        assert out["distance_max"] == random_patch.coords.max("distance")

    def test_items(self, random_patch):
        """Ensure items works like a dict."""
        attrs = random_patch.attrs
        assert dict(attrs.items()) == attrs.model_dump()

    def test_dims_live_on_patch(self, random_patch):
        """Dimensions remain available on patch after renaming coords."""
        pat = random_patch.rename_coords(distance="channel")
        assert pat.dims == ("channel", "time")
        assert "dims" not in pat.attrs.model_dump()


class TestSummaryAttrs:
    """Tests for summarizing a schema."""

    def test_attrs_reconstructed(self, random_patch, random_summary):
        """Ensure all the expected attrs are extracted."""
        summary1 = dict(random_summary)
        attrs = dict(random_patch.attrs)
        for key in set(summary1) & set(attrs):
            assert summary1[key] == attrs[key]

    def test_can_jsonize(self, random_summary):
        """Ensure the summary can be converted to json."""
        assert isinstance(random_summary.model_dump_json(), str)

    def test_can_roundtrip(self, random_summary):
        """Ensure json can be round-tripped."""
        json = random_summary.model_dump_json()
        assert PatchAttrs.model_validate_json(json) == random_summary

    def test_from_dict(self, random_attrs):
        """from_dict should accept attrs views and mappings."""
        out = PatchAttrs.from_dict(random_attrs)
        assert isinstance(out, PatchAttrs)
        new_dict = dict(random_attrs)
        new_dict["data_units"] = "m/s"
        out = PatchAttrs.from_dict(new_dict)
        assert isinstance(out, PatchAttrs)

    def test_from_dict_none(self):
        """from_dict should normalize None to an empty attrs instance."""
        out = PatchAttrs.from_dict(None)
        assert isinstance(out, PatchAttrs)
        assert out.model_dump() == PatchAttrs().model_dump()

    def test_from_dict_drops_dims(self):
        """from_dict should continue dropping dims during normalization."""
        out = PatchAttrs.from_dict({"tag": "bob", "dims": "time,distance"})
        assert out.tag == "bob"
        assert "dims" not in out.model_dump()

    def test_from_dict_model_dump_provider(self, random_patch):
        """from_dict should accept non-PatchAttrs objects with model_dump."""

        class ModelDumpProvider:
            """Simple object that only exposes model_dump."""

            def model_dump(self):
                return {"tag": random_patch.attrs.tag}

        out = PatchAttrs.from_dict(ModelDumpProvider())
        assert isinstance(out, PatchAttrs)


class TestDropPrivate:
    """Tests for dropping private attrs."""

    def test_simple_drop_private(self):
        """Ensure private attrs are removed after operation."""
        attrs = PatchAttrs(_private1=1, extra_attr=2).drop_private()
        assert "_private1" not in dict(attrs)
        assert "extra_attr" in dict(attrs)

    def test_flat_dump_alias(self, random_attrs):
        """flat_dump should stay an alias for model_dump."""
        assert random_attrs.flat_dump() == random_attrs.model_dump()


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

    def test_patch_summary_exposes_coord_summaries(self, random_patch_with_lat_lon):
        """Patch summary should expose coordinate summary accessors."""
        summary = random_patch_with_lat_lon.summary
        assert summary.get_coord_summary(
            "time"
        ).min == random_patch_with_lat_lon.coords.min("time")
        assert (
            random_patch_with_lat_lon.summary.get_coord_summary("time")
            == random_patch_with_lat_lon.coords.to_summary_dict()["time"]
        )


class TestUpdateAttrs:
    """Tests for updating attributes."""

    def test_attrs_can_update_non_coord_fields(self, random_attrs):
        """Non-coordinate updates still work."""
        attrs = PatchAttrs.from_dict(random_attrs).update(tag="miles")
        assert attrs.tag == "miles"

    def test_update_rejects_coord_like_fields(self, random_attrs):
        """Flat coordinate-like fields should go through update_coords instead."""
        with pytest.raises(PatchAttributeError, match="update_coords"):
            PatchAttrs.from_dict(random_attrs).update(time_min=1)

    def test_update_rejects_nested_coords(self, random_patch):
        """Passing coords directly should fail."""
        with pytest.raises(PatchAttributeError, match="coordinate metadata"):
            PatchAttrs.from_dict(random_patch.attrs).update(coords=random_patch.coords)

    def test_update_ignores_dims(self, random_attrs):
        """Passing dimensions directly should be ignored by normalization."""
        out = PatchAttrs.from_dict(random_attrs).update(dims=("time", "distance"))
        assert "dims" not in out.model_dump()


class TestGetAttrSummary:
    """Test getting dataframe of summary info."""

    def test_summary(self, random_attrs):
        """Ensure a dataframe is returned."""
        out = random_attrs.get_summary_df()
        assert isinstance(out, pd.DataFrame)


class TestSeparateConstruction:
    """Tests for constructing coords separately from attrs."""

    def test_patch_accepts_coords_and_attrs(self):
        """Patch construction should keep attrs pure while coords carry summaries."""
        coord = get_coord(start=0, stop=10, step=1)
        patch = dc.Patch(
            data=np.asarray([[1] * 10]),
            coords={"time": coord, "distance": [0]},
            dims=("distance", "time"),
        )
        assert patch.summary.get_coord_summary("time").min == coord.min()
