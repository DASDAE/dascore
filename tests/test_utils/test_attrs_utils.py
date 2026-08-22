"""Tests for attr utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from dascore import PatchAttrs
from dascore.exceptions import ParameterError
from dascore.utils.attrs import _is_missing, combine_patch_attrs


class TestMergeAttrs:
    """Tests for merging patch attrs."""

    def test_empty(self):
        """Empty PatchAttrs should work in all cases."""
        pa1, pa2 = PatchAttrs(), PatchAttrs()
        assert isinstance(combine_patch_attrs([pa1, pa2]), PatchAttrs)
        out = combine_patch_attrs([pa1, pa2], drop_attrs="history")
        assert isinstance(out, PatchAttrs)

    def test_drop(self):
        """Ensure drop_attrs does its job."""
        pa1 = PatchAttrs(history=["a", "b"])
        pa2 = PatchAttrs()
        out = combine_patch_attrs([pa1, pa2], drop_attrs="history")
        assert isinstance(out, PatchAttrs)

    def test_invalid_conflicts_raises(self):
        """An unsupported conflicts value should raise. See #804."""
        pa1, pa2 = PatchAttrs(tag="bob"), PatchAttrs(tag="bill")
        with pytest.raises(ParameterError, match="conflict must be one of"):
            combine_patch_attrs([pa1, pa2], conflict="banana")

    def test_conflicts(self):
        """Ensure when non-dim fields aren't equal merge raises."""
        pa1 = PatchAttrs(tag="bob", another=2, same=42)
        pa2 = PatchAttrs(tag="bob", another=2, same=42)
        pa3 = PatchAttrs(another=1, same=42, different=10)
        with pytest.raises(Exception, match="hold conflicting values"):
            combine_patch_attrs([pa1, pa2, pa3])

    def test_missing_is_a_value(self):
        """An attr one member never recorded conflicts with one which did."""
        pa1 = PatchAttrs(data_type="velocity", gauge=10.0)
        pa2 = PatchAttrs(data_type="", gauge=np.nan)
        for order in ([pa1, pa2], [pa2, pa1]):
            with pytest.raises(Exception, match="hold conflicting values"):
                combine_patch_attrs(order)

    def test_missing_spellings_are_one_value(self):
        """None, NaN, and "" are the same value, so they never conflict."""
        attrs = [PatchAttrs(foo=""), PatchAttrs(foo=None), PatchAttrs(foo=np.nan)]
        assert combine_patch_attrs(attrs).get("foo") is None

    def test_all_missing_is_omitted(self):
        """An attr nobody knows is left out rather than carried as ""."""
        out = combine_patch_attrs([PatchAttrs(foo=""), PatchAttrs(foo="")])
        assert out.get("foo") is None

    def test_drop_omits_missing_beside_known(self):
        """Drop omits an attr one member left empty, like any other conflict."""
        pa1 = PatchAttrs(data_type="velocity", foo="a")
        pa2 = PatchAttrs(data_type="", foo="b")
        out = combine_patch_attrs([pa1, pa2], conflict="drop")
        # data_type is a declared field, so dropping it leaves its default.
        assert _is_missing(out.get("data_type"))
        assert out.get("foo") is None

    def test_keep_first_means_the_first_member(self):
        """keep_first keeps the first member's value, empty or not."""
        pa1 = PatchAttrs(foo="", bar=None)
        pa2 = PatchAttrs(foo="x", bar=1)
        out = combine_patch_attrs([pa1, pa2], conflict="keep_first")
        assert out.get("foo") is None
        assert out.get("bar") is None
        out = combine_patch_attrs([pa2, pa1], conflict="keep_first")
        assert out.foo == "x"
        assert out.bar == 1

    def test_history_never_compared(self):
        """Histories differing is not a conflict; the first member's carries."""
        pa1 = PatchAttrs(history=("pass_filter",))
        pa2 = PatchAttrs(history=())
        assert combine_patch_attrs([pa1, pa2]).history == ("pass_filter",)
        assert combine_patch_attrs([pa2, pa1]).history == ()

    def test_drop_conflicts(self, random_patch):
        """Ensure unequal non-coordinate attrs can be dropped."""
        attrs = PatchAttrs.from_dict(random_patch.attrs)
        pa1 = attrs.update(tag="bill", acquisition_key="UU.R2D1..RAW")
        pa2 = attrs.update(tag="bob", acquisition_key="TA.R2D1..RAW")
        out = combine_patch_attrs([pa1, pa2], conflict="drop")
        defaults = PatchAttrs()
        assert out.tag == defaults.tag
        assert out.acquisition_key == defaults.acquisition_key

    def test_keep_disjoint_values(self, random_patch):
        """Ensure disjoint values can be kept."""
        random_attrs = PatchAttrs.from_dict(random_patch.attrs)
        attrs1 = random_attrs.update(jazz_hands=1984)
        out = combine_patch_attrs([attrs1, random_attrs], conflict="keep_first")
        assert out.jazz_hands == 1984

    def test_patch_input(self, random_patch):
        """Patch objects should normalize through their attrs."""
        out = combine_patch_attrs([random_patch, random_patch])
        assert isinstance(out, PatchAttrs)

    def test_mapping_input(self, random_patch):
        """Plain mapping inputs should normalize through PatchAttrs."""
        out = combine_patch_attrs(
            [random_patch.attrs.model_dump(), random_patch.attrs],
            conflict="keep_first",
        )
        assert isinstance(out, PatchAttrs)

    def test_mapping_like_input(self, random_patch):
        """Non-dict mapping inputs should normalize through from_dict."""

        class MappingLike(Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        mapping = MappingLike(random_patch.attrs.model_dump())
        out = combine_patch_attrs([mapping, random_patch.attrs], conflict="keep_first")
        assert isinstance(out, PatchAttrs)

    def test_private_attrs_are_ignored_for_merge_conflicts(self, random_patch):
        """Private attrs should not block attr merging."""
        attrs1 = random_patch.attrs.update(_source_patch_key="one")
        attrs2 = random_patch.attrs.update(_source_patch_key="two")
        out = combine_patch_attrs([attrs1, attrs2])
        assert "_source_patch_key" not in out.model_dump()
