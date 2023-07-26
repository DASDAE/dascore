"""Tests for schema."""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from dascore.core.schema import PatchAttrs


@pytest.fixture(scope="class")
def random_summary(random_patch) -> PatchAttrs:
    """Return the summary of the random patch."""
    return PatchAttrs.model_validate(dict(random_patch.attrs))


@pytest.fixture(scope="class")
def random_attrs(random_patch) -> PatchAttrs:
    """Return the summary of the random patch."""
    return random_patch.attrs


class TestSummarySchema:
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
        """Ensure json can be round-tripped"""
        json = random_summary.model_dump_json()
        random_summary2 = PatchAttrs.model_validate_json(json)
        assert random_summary2 == random_summary

    def test_sub_docstrings(self):
        """Ensure the docstring for PatchSummary had params subbed in."""
        docstr = PatchAttrs.__doc__
        # data_type is one of the parameters inserted into docstring.
        assert "data_type" in docstr

    def test_from_dict(self, random_attrs):
        """Test new method for more intuitive init."""
        out = PatchAttrs.from_dict(random_attrs)
        assert out == random_attrs
        new_dict = dict(random_attrs)
        new_dict["data_units"] = "m/s"
        out = PatchAttrs.from_dict(new_dict)
        assert isinstance(out, PatchAttrs)


class TestSchemaIsDictLike:
    """Tests to insure schema behaves like a dict."""

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


class TestDropPrivate:
    """Tests for dropping private attrs."""

    def test_simple_drop(self):
        """Ensure private attrs are removed after operation."""
        attrs = PatchAttrs(_private1=1, extra_attr=2).drop_private()
        attr_dict = dict(attrs)
        assert "_private1" not in attr_dict
        assert "extra_attr" in attr_dict
