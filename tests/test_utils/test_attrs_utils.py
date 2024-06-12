"""
Tests for attr utilities.
"""

from __future__ import annotations

import pytest

import dascore as dc
from dascore import PatchAttrs
from dascore.exceptions import AttributeMergeError
from dascore.utils.attrs import (
    combine_patch_attrs,
    decompose_attrs,
    separate_coord_info,
)
from dascore.utils.downloader import fetcher


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

    def test_unequal_raises(self, random_patch):
        """When attrs have unequal coords it should raise an error."""
        attr1 = random_patch.attrs
        attr2 = attr1.update(distance_units="miles")
        match = "Cant merge patch attrs"
        with pytest.raises(AttributeMergeError, match=match):
            combine_patch_attrs([attr1, attr2], coord_name="distance")


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


class TestSeparateCoordInfo:
    """Tests for separating coord info from attr dict."""

    def test_empty(self):
        """Empty args should return emtpy dicts."""
        out1, out2 = separate_coord_info(None)
        assert out1 == out2 == {}

    def test_meets_reqs(self):
        """Simple case for filtering out required attrs."""
        input_dict = {"coords": {"time": {"min": 10}}}
        coords, attrs = separate_coord_info(input_dict)
        assert coords == input_dict["coords"]

    def test_dict_of_coord_info(self, random_patch):
        """Passing in a dictionary of coord info should work."""
        coord_dict = random_patch.coords.to_summary_dict()
        dims = random_patch.dims
        coords, attrs = separate_coord_info(coord_dict, dims=dims)
        assert coords == coord_dict
