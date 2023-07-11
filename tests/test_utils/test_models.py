"""
Tests for DASCore models and related functionality.
"""

import pytest
from pydantic import BaseModel

from dascore.core.schema import PatchAttrs
from dascore.exceptions import AttributeMergeError
from dascore.utils.models import merge_models


class _NewMod(BaseModel):
    """A model for testing different classes."""

    here: int = 0


class TestMergeModels:
    """Tests for merging attrs."""

    def test_empty(self):
        """Empty PatchAttrs should work in all casses"""
        pa1, pa2 = PatchAttrs(), PatchAttrs()
        assert isinstance(merge_models([pa1, pa2]), PatchAttrs)
        assert isinstance(merge_models([pa1, pa2], "time"), PatchAttrs)
        assert isinstance(merge_models([pa1, pa2], "distance"), PatchAttrs)
        out = merge_models([pa1, pa2], drop_attrs="history")
        assert isinstance(out, PatchAttrs)

    def test_simple_merge(self):
        """Happy path, simple merge."""
        pa1 = PatchAttrs(distance_min=1, distance_max=10)
        pa2 = PatchAttrs(distance_min=11, distance_max=20)

        merged = merge_models([pa1, pa2], "distance")
        # the names of attrs should be identical before/after merge
        assert set(dict(merged)) == set(dict(pa1))
        assert merged.distance_max == pa2.distance_max
        assert merged.distance_min == pa1.distance_min

    def test_different_classes_raises(self):
        """Different classes of models should raise."""
        pa1 = PatchAttrs()
        p2 = _NewMod()
        with pytest.raises(AttributeMergeError, match="same class"):
            merge_models([pa1, p2])

    def test_drop(self):
        """Ensure drop_attrs does its job."""
        pa1 = PatchAttrs(history=["a", "b"])
        pa2 = PatchAttrs()
        with pytest.raises(AttributeMergeError, match="not all of their non-dim"):
            merge_models([pa1, pa2])
        out = merge_models([pa1, pa2], drop_attrs="history")
        assert isinstance(out, PatchAttrs)

    def test_conflicts(self):
        """Ensure when non-dim fields aren't equal merge raises."""
        pa1 = PatchAttrs(tag="bob")
        pa2 = PatchAttrs()
        with pytest.raises(AttributeMergeError, match="not all of their non-dim"):
            merge_models([pa1, pa2])

    def test_not_all_attributes(self):
        """Ensure when non-dim fields aren't equal merge raises."""
        pa1 = PatchAttrs(bob_min=10, bob_max=12)
        pa2 = PatchAttrs(bob_min=13)
        with pytest.raises(AttributeMergeError, match="required attributes"):
            merge_models([pa1, pa2], dim="bob")

    def test_drop_conflicts(self):
        """Ensure when non-dim fields aren't equal merge raises."""
        pa1 = PatchAttrs(tag="bill")
        pa2 = PatchAttrs(station="TA")
        defaults = PatchAttrs()
        out = merge_models([pa1, pa2], dim="time", conflicts="drop")
        assert isinstance(out, PatchAttrs)
        assert out.tag == defaults.tag
        assert out.station == defaults.station

    def test_keep_disjoint_values(self, random_patch):
        """Ensure when disjoint values should be kept they are."""
        random_attrs = random_patch.attrs
        new = dict(random_attrs)
        new["jazz_hands"] = 1984
        attrs1 = PatchAttrs(**new)
        out = merge_models([attrs1, random_attrs], conflicts="keep_first")
        assert out.jazz_hands == 1984
