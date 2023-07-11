"""Tests for schema."""
import pydantic
import pytest

from dascore.core.schema import PatchAttrs
from dascore.utils.models import SimpleValidator


@pytest.fixture(scope="class")
def random_summary(random_patch) -> PatchAttrs:
    """Return the summary of the random patch."""
    return PatchAttrs.parse_obj(dict(random_patch.attrs))


@pytest.fixture(scope="class")
def random_attrs(random_patch) -> PatchAttrs:
    """Return the summary of the random patch."""
    return random_patch.attrs


class TestSimpleValidator:
    """Test suite for validator."""

    def test_base_validator(self):
        """Ensure the base validator does nothing."""

        class MyModel(pydantic.BaseModel):
            """A test model."""

            value: SimpleValidator = 1

        my_model = MyModel(value=2)
        assert my_model.value == 2


class TestSummarySchema:
    """Tests for summarizing a schema."""

    def test_attrs_reconstructed(self, random_patch, random_summary):
        """Ensure all the expected attrs are extracted."""
        summary1 = random_summary.dict()
        attrs = dict(random_patch.attrs)
        common_keys = set(summary1) & set(attrs)
        for key in common_keys:
            assert summary1[key] == attrs[key]

    def test_can_jsonize(self, random_summary):
        """Ensure the summary can be converted to json."""
        json = random_summary.json()
        assert isinstance(json, str)

    def test_can_roundrip(self, random_summary):
        """Ensure json can be round-tripped"""
        json = random_summary.json()
        random_summary2 = PatchAttrs.parse_raw(json)
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
        with pytest.raises(TypeError, match="is immutable"):
            random_attrs.bob = 1
        with pytest.raises(TypeError, match="is immutable"):
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
