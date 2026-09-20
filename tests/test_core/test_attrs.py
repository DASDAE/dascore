"""Tests for PatchAttrs."""

from __future__ import annotations

import uuid
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.constants import INVENTORY_ATTRS, VALID_DATA_TYPES, max_lens
from dascore.core.attrs import PatchAttrs, scalar_attrs
from dascore.core.coords import get_coord
from dascore.core.inventory import Acquisition, Interrogator
from dascore.exceptions import (
    InvalidInventoryError,
    ParameterError,
    PatchAttributeError,
)
from dascore.units import get_quantity
from dascore.utils.misc import validate_acquisition_key


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

    def test_supports_extra_attrs(self):
        """The attr dict should allow extra attributes."""
        out = PatchAttrs(bob="doesnt", bill_min=12, bob_max="2012-01-12")
        assert out.bob == "doesnt"
        assert out.bill_min == 12

    def test_valid_data_types_fit_max_length(self):
        """Ensure supported data_type values fit the declared attr length."""
        max_len = max_lens["data_type"]

        for data_type in VALID_DATA_TYPES:
            assert len(data_type) <= max_len

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

    def test_update_accepts_coord_like_fields(self, random_attrs):
        """Coord-shaped names are ordinary attrs; update never touches coords."""
        out = PatchAttrs.from_dict(random_attrs).update(time_min=1, channel_step=3)
        assert out["time_min"] == 1
        assert out["channel_step"] == 3

    def test_update_rejects_nested_coords(self, random_patch):
        """Passing coords directly should fail."""
        with pytest.raises(ValueError, match="coordinate metadata"):
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


class TestAcquisitionKey:
    """The identity key which resolves a patch against an inventory."""

    valid = ("XX.R2D1.01.RAW", "XX.R2D1..RAW", "X-1.A2.00.H-Z")
    invalid = ("XX.R2D1.01", "XX.R2D1.01.RAW.EXTRA", "XX.R2D1.01.R W", "XX..01.RAW")

    @pytest.mark.parametrize("value", valid)
    def test_valid(self, value):
        """Four code tokens, of which only the location may be blank."""
        assert PatchAttrs(acquisition_key=value).acquisition_key == value

    @pytest.mark.parametrize("value", invalid)
    def test_invalid(self, value):
        """Wrong token count or illegal characters are rejected."""
        with pytest.raises(ValidationError):
            PatchAttrs(acquisition_key=value)

    def test_unset_is_empty(self):
        """An empty id means the patch has no inventory identity."""
        assert PatchAttrs().acquisition_key == ""

    @pytest.mark.parametrize("value", invalid)
    def test_inventory_agrees(self, value):
        """
        The inventory rejects exactly what the attr rejects.

        Both go through one validator so a code which is legal in a file
        header cannot be illegal in the inventory naming the same source.
        """
        with pytest.raises(InvalidInventoryError):
            validate_acquisition_key(value)

    @pytest.mark.parametrize("value", valid)
    def test_inventory_agrees_valid(self, value):
        """The shared validator accepts what the attr accepts."""
        assert validate_acquisition_key(value) == value

    def test_removed_names_become_extras(self):
        """
        The names acquisition_key replaced are ordinary extras now.

        They are no longer part of the model, so nothing in DASCore reads
        them, but a patch which sets one keeps it like any other extra.
        """
        attrs = PatchAttrs(network="XX", station="A1", instrument_id="sn-1")
        assert set(PatchAttrs.model_fields) & {"network", "station"} == set()
        assert attrs.get("network") == "XX"


class TestInventoryAttrs:
    """The vocabulary readers share with the inventory."""

    def test_names_exist_in_model(self):
        """
        Every canonical name is a field of the model it claims to mirror.

        The vocabulary is what keeps a file header and an enriched value
        the same attr; a name which no longer matches the inventory would
        quietly split them in two.
        """
        acquisition = set(Acquisition.model_fields)
        interrogator = set(Interrogator.model_fields)
        for name in INVENTORY_ATTRS:
            prefix, _, field = name.rpartition(".")
            if prefix == "interrogator":
                assert field in interrogator, name
            else:
                assert not prefix, name
                assert field in acquisition, name

    def test_excludes_data_state(self):
        """
        Data-state fields are the patch's, not the observing system's.

        Processing rewrites them, so blanket enrichment must not carry
        them even though the inventory records their as-acquired values.
        `data_category` is not one: no processing function rewrites which
        family of instrument recorded the data.
        """
        assert not set(INVENTORY_ATTRS) & {"data_type", "data_units"}
        assert "data_category" in INVENTORY_ATTRS


class TestScalarAttrs:
    """Attrs hold scalars; an array belongs on the patch as a coordinate."""

    non_scalar = (
        np.array([1.0, 2.0]),
        [1, 2],
        (1, 2),
        {1, 2},
        {"a": 1},
        pd.Series([1.0, 2.0]),
        pd.DataFrame({"a": [1.0, 2.0]}),
        get_quantity("m") * np.array([1.0, 2.0]),
        PatchAttrs(),
        (x for x in range(3)),
        np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])[0],
    )

    accepted = (
        "a string",
        b"bytes",
        True,
        3,
        1.5,
        1 + 2j,
        None,
        np.float32(1.0),
        np.datetime64("2020-01-01"),
        np.timedelta64(1, "s"),
        pd.Timestamp("2020-01-01"),
        get_quantity("10 m"),
        get_quantity("m").units,
        Path("a_file"),
        uuid.uuid4(),
        Decimal("1.5"),
        object(),
    )

    @pytest.mark.parametrize("value", non_scalar)
    def test_skipped_with_a_warning(self, value):
        """By default an attr holding more than one value is dropped."""
        with pytest.warns(UserWarning, match="Attrs hold scalars"):
            out = PatchAttrs(gauge=value)
        assert "gauge" not in dict(out)

    @pytest.mark.parametrize("value", non_scalar)
    def test_refused_on_raise(self, value):
        """Asking for a refusal gets one, naming the attr and its type."""
        with pytest.raises(PatchAttributeError, match="Attrs hold scalars"):
            PatchAttrs.from_dict({"gauge": value}, "raise")

    @pytest.mark.parametrize("value", accepted)
    def test_accepted(self, value):
        """A scalar of any kind is kept as it was given."""
        assert PatchAttrs(gauge=value).gauge is value

    def test_message_names_the_coordinate_alternative(self):
        """The error says where an array goes instead."""
        with pytest.raises(PatchAttributeError, match=r"update_coords\(gauge="):
            PatchAttrs.from_dict({"gauge": np.array([1.0, 2.0])}, "raise")

    def test_one_warning_names_them_all(self):
        """A file with several such attrs is one warning, not a stream."""
        stored = {"tag": "x", "gauge": np.array([1.0, 2.0]), "pair": (1, 2)}
        with pytest.warns(UserWarning, match="'gauge'.*'pair'") as record:
            out = PatchAttrs(**stored)
        assert len(record) == 1
        assert out.tag == "x"

    def test_ignore_is_silent(self):
        """The third mode drops the value without saying anything."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = PatchAttrs.from_dict({"gauge": np.array([1.0, 2.0])}, "ignore")
        assert "gauge" not in dict(out)

    def test_retired_mode_raises(self):
        """The mode is spelled the way every other warn level is."""
        with pytest.raises(ParameterError, match="on_non_scalar"):
            PatchAttrs.from_dict({"tag": "x"}, "drop")

    @pytest.mark.parametrize("mode", ["warn", "raise", "ignore"])
    def test_one_value_is_that_value(self, mode):
        """HDF5 spells a scalar as a length-1 array; it is one, silently."""
        stored = {
            "project": np.array(["survey"]),
            "epsg_code": np.array([4326]),
            "serial": np.array([b"XYZ123"]),
            "solo": ("only",),
            "wrapped": np.array(5.0),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = PatchAttrs.from_dict(stored, mode)
        assert out.project == "survey"
        assert out.epsg_code == 4326
        assert out.serial == "XYZ123"
        assert out.solo == "only"
        assert out.wrapped == 5.0 and not isinstance(out.wrapped, np.ndarray)

    def test_zero_dim_object_array_does_not_smuggle(self):
        """The unwrapped value is itself held to the rule."""
        wrapped = np.empty((), dtype=object)
        wrapped[()] = [1, 2]
        with pytest.raises(PatchAttributeError, match="Attrs hold scalars"):
            PatchAttrs.from_dict({"gauge": wrapped}, "raise")

    def test_history_is_still_a_tuple(self):
        """History is a declared field, and its annotation allows one."""
        assert PatchAttrs(history=["a", "b"]).history == ("a", "b")

    def test_a_subclass_keeps_its_own_fields(self):
        """The class the values are destined for decides what is declared."""

        class _Attrs(PatchAttrs):
            """Attrs with a declared collection."""

            notes: tuple[str, ...] = ()

        assert _Attrs(notes=("a", "b")).notes == ("a", "b")
        assert scalar_attrs({"notes": ("a", "b")}, "raise", _Attrs)["notes"]
        with pytest.raises(PatchAttributeError, match="Attrs hold scalars"):
            scalar_attrs({"notes": ("a", "b")}, "raise")

    def test_structural_keys_are_not_attrs(self):
        """`dims` and `coords` say how a patch is built, so they pass."""
        stored = {"dims": ("time", "distance"), "coords": {"time": 1}}
        assert scalar_attrs(stored, "raise") == stored

    def test_update_attrs_modes(self, random_patch):
        """The patch-level switch is the same word, keyword-only."""
        with pytest.warns(UserWarning, match="Attrs hold scalars"):
            warned = random_patch.update_attrs(gauge=np.array([1.0, 2.0]))
        assert "gauge" not in dict(warned.attrs)
        with pytest.raises(PatchAttributeError, match="Attrs hold scalars"):
            random_patch.update_attrs(gauge=[1, 2], on_non_scalar="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quiet = random_patch.update_attrs(gauge=[1, 2], on_non_scalar="ignore")
        assert "gauge" not in dict(quiet.attrs)

    def test_update_attrs_keeps_the_attrs_class(self, random_patch):
        """An unrelated edit leaves a reader's declared fields declared."""

        class _Attrs(PatchAttrs):
            """Attrs with a declared collection."""

            flags: tuple[bool, ...] = ()

        attrs = _Attrs(flags=(True, False))
        patch = random_patch.new(attrs=attrs)
        out = patch.update_attrs(tag="new")
        assert isinstance(out.attrs, _Attrs)
        assert out.attrs.flags == (True, False)
        assert out.attrs.tag == "new"

    def test_the_switch_is_not_an_attr(self, random_patch):
        """A stored attr of that name survives; the keyword is the switch."""
        patch = random_patch.update_attrs(on_non_scalar="ignore", tag="x")
        assert "on_non_scalar" not in dict(patch.attrs)
        stored = PatchAttrs(on_non_scalar="a stored value")
        assert stored.on_non_scalar == "a stored value"

    def test_the_switch_is_not_in_the_operation(self, random_patch):
        """The mode is how attrs were read, not what the patch became."""
        one = random_patch.update_attrs(tag="x")
        two = random_patch.update_attrs(tag="x", on_non_scalar="ignore")
        assert one.attrs.data_id == two.attrs.data_id
        assert one.attrs.history == two.attrs.history


class TestReviewThreads:
    """Edges the pull request's reviewers raised."""

    def test_a_declared_field_given_one_value_in_an_array(self):
        """A file's length-1 array for `data_type` is the name, not its repr."""
        attrs = dc.PatchAttrs.from_dict({"data_type": np.array(["strain_rate"])})
        assert attrs.data_type == "strain_rate"
        assert dc.PatchAttrs(tag=np.array([b"bob"])).tag == "bob"

    def test_history_is_not_unwrapped(self):
        """A one-entry history is still a history."""
        assert dc.PatchAttrs(history=("one",)).history == ("one",)

    def test_a_bad_mode_is_refused_for_an_instance_too(self):
        """The early return does not skip the check on the mode."""
        with pytest.raises(ParameterError, match="on_non_scalar"):
            dc.PatchAttrs.from_dict(dc.PatchAttrs(), on_non_scalar="drop")

    def test_arrays_with_matching_nans_agree(self):
        """As two null scalars do."""
        from dascore.utils.attrs import _attr_values_equal  # noqa: PLC0415

        one = np.array([1.0, np.nan])
        assert _attr_values_equal(one, one.copy())
        assert not _attr_values_equal(one, np.array([1.0, 2.0]))
        assert _attr_values_equal(np.array(["a"]), np.array(["a"]))
