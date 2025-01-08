"""Tests for Patch attrs modules."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.attrs import (
    PatchAttrs,
)
from dascore.utils.misc import register_func

MORE_COORDS_ATTRS = []


@pytest.fixture(scope="class")
def random_attrs_loaded(random_patch) -> PatchAttrs:
    """Return the random patch attrs dumped then loaded."""
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
        out = random_attrs.get("category")
        assert out == random_attrs.category

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

    def test_supports_extra_attrs(self):
        """The attr dict should allow extra attributes."""
        out = PatchAttrs(bob="doesnt", bill_min=12, bob_max="2012-01-12")
        assert out.bob == "doesnt"
        assert out.bill_min == 12

    def test_items(self, random_patch):
        """Ensure items works like a dict."""
        attrs = random_patch.attrs
        out = dict(attrs.items())
        assert out == attrs.model_dump()


class TestSummaryAttrs:
    """Tests for summarizing a schema."""

    def test_attrs_reconstructed(self, random_patch, random_attrs_loaded):
        """Ensure all the expected attrs are extracted."""
        summary1 = dict(random_attrs_loaded)
        attrs = dict(random_patch.attrs)
        common_keys = set(summary1) & set(attrs)
        for key in common_keys:
            assert summary1[key] == attrs[key]

    def test_can_jsonize(self, random_attrs_loaded):
        """Ensure the summary can be converted to json."""
        json = random_attrs_loaded.model_dump_json()
        assert isinstance(json, str)

    def test_can_roundrip(self, random_attrs_loaded):
        """Ensure json can be round-tripped."""
        json = random_attrs_loaded.model_dump_json()
        random_summary2 = PatchAttrs.model_validate_json(json)
        assert random_summary2 == random_attrs_loaded


class TestDropPrivate:
    """Tests for dropping private attrs."""

    def test_simple_drop(self):
        """Ensure private attrs are removed after operation."""
        attrs = PatchAttrs(_private1=1, extra_attr=2).drop_private()
        attr_dict = dict(attrs)
        assert "_private1" not in attr_dict
        assert "extra_attr" in attr_dict


class TestUpdateAttrs:
    """Tests for updating attributes."""

    def test_attrs_can_update(self, random_attrs):
        """Ensure attributes can update coordinates."""
        # We also test that str get converted to data units correctly.
        attrs = random_attrs.update(data_units="m/s")
        expected = dc.get_quantity("m/s")
        assert attrs.data_units == expected


class TestGetAttrSummary:
    """Test getting dataframe of summary info."""

    def test_summary(self, random_attrs):
        """Ensure a dataframe is returned."""
        out = random_attrs.get_summary_df()
        assert isinstance(out, pd.DataFrame)
