"""Tests for attr utilities."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

import dascore as dc
from dascore import PatchAttrs
from dascore.utils.attrs import (
    combine_patch_attrs,
    separate_coord_info,
)


def _validate_no_coords(
    obj,
    dims: tuple[str, ...] | None = None,
    coord_manager=None,
    raise_error: bool = True,
    flat_keys: bool = True,
) -> dict:
    """Test helper for the old attrs-purity behavior."""

    def _normalize(obj):
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return dict(obj)

    out = _normalize(obj)
    direct_names = set()
    if "coords" in out:
        direct_names.add("coords")
        out.pop("coords", None)
    if "dims" in out:
        direct_names.add("dims")
        out.pop("dims", None)
    known_names = set(() if dims is None else dims) | {"time", "distance"}
    if coord_manager is not None:
        dims = coord_manager.dims
        known_names |= set(coord_manager.coord_map)
    if flat_keys:
        scan_dims = tuple(known_names) if known_names else dims
        coord_info, attr_info = separate_coord_info(out, dims=scan_dims)
        coord_names = direct_names | set(coord_info)
    else:
        attr_info = out
        coord_names = direct_names
    if coord_names and raise_error:
        names = ", ".join(sorted(coord_names))
        msg = "PatchAttrs no longer accepts coordinate metadata. " f"Received: {names}."
        raise ValueError(msg)
    return attr_info


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

    def test_conflicts(self):
        """Ensure when non-dim fields aren't equal merge raises."""
        pa1 = PatchAttrs(tag="bob", another=2, same=42)
        pa2 = PatchAttrs(tag="bob", another=2, same=42)
        pa3 = PatchAttrs(another=1, same=42, different=10)
        with pytest.raises(Exception, match="non-dim attrs are not equal"):
            combine_patch_attrs([pa1, pa2, pa3])

    def test_drop_conflicts(self, random_patch):
        """Ensure unequal non-coordinate attrs can be dropped."""
        pa1 = PatchAttrs.from_dict(random_patch.attrs).update(tag="bill", station="UU")
        pa2 = PatchAttrs.from_dict(random_patch.attrs).update(station="TA", tag="bob")
        out = combine_patch_attrs([pa1, pa2], conflicts="drop")
        defaults = PatchAttrs()
        assert out.tag == defaults.tag
        assert out.station == defaults.station

    def test_keep_disjoint_values(self, random_patch):
        """Ensure disjoint values can be kept."""
        random_attrs = PatchAttrs.from_dict(random_patch.attrs)
        attrs1 = random_attrs.update(jazz_hands=1984)
        out = combine_patch_attrs([attrs1, random_attrs], conflicts="keep_first")
        assert out.jazz_hands == 1984

    def test_patch_input(self, random_patch):
        """Patch objects should normalize through their attrs."""
        out = combine_patch_attrs([random_patch, random_patch])
        assert isinstance(out, PatchAttrs)

    def test_mapping_input(self, random_patch):
        """Plain mapping inputs should normalize through PatchAttrs."""
        out = combine_patch_attrs(
            [random_patch.attrs.model_dump(), random_patch.attrs],
            conflicts="keep_first",
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
        out = combine_patch_attrs([mapping, random_patch.attrs], conflicts="keep_first")
        assert isinstance(out, PatchAttrs)


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
        assert attrs == {}

    def test_dict_of_coord_info(self, random_patch):
        """Passing in a dictionary of coord info should work."""
        coord_dict = random_patch.coords.to_summary_dict()
        dims = random_patch.dims
        coords, attrs = separate_coord_info(coord_dict, dims=dims)
        assert coords == coord_dict
        assert attrs == {}

    def test_ignores_keys_without_underscore(self):
        """Keys without separators should stay in attrs."""
        coords, attrs = separate_coord_info({"time": 1, "tag": "x"})
        assert coords == {}
        assert attrs == {"time": 1, "tag": "x"}

    def test_ignores_unknown_coord_suffix(self):
        """Unknown coord suffixes should not be parsed as coord metadata."""
        coords, attrs = separate_coord_info({"time_bob": 1, "tag": "x"})
        assert coords == {}
        assert attrs == {"time_bob": 1, "tag": "x"}

    def test_unsplittable_valid_coord_key_stays_attr(self):
        """Suffix-only coord-looking keys should not crash dim inference."""
        coords, attrs = separate_coord_info({"units": "m", "tag": "x"})
        assert coords == {}
        assert attrs == {"units": "m", "tag": "x"}

    def test_invalid_coord_like_key_ignored_in_dim_inference(self):
        """Only valid coord summary keys should participate in inferred dims."""
        obj = {"time_bob": 1, "distance_min": 0, "distance_max": 10}
        coords, attrs = separate_coord_info(obj)
        assert set(coords) == {"distance"}
        assert attrs == {
            "time_bob": 1,
        }

    def test_dim_inference_skips_unsplittable_keys(self):
        """Unsplittable keys should be ignored while inferring dims."""
        obj = {
            "distance": 1,
            "distance_bob": 2,
            "distance_min": 0,
            "distance_max": 10,
        }
        coords, attrs = separate_coord_info(obj)
        assert "distance" in coords
        assert attrs == {}

    def test_coord_level_to_summary(self):
        """Coord-level values exposing to_summary should be normalized."""

        class CoordLike:
            def to_summary(self):
                return dc.core.CoordSummary(min=0, max=1, step=1)

        coords, attrs = separate_coord_info({"coords": {"time": CoordLike()}})
        assert attrs == {}
        assert coords["time"]["min"] == 0

    def test_coord_manager_input_uses_dims_and_summary_dict(self, random_patch):
        """CoordManager inputs should use dims and to_summary_dict paths."""
        coords, attrs = separate_coord_info({"coords": random_patch.coords})
        assert attrs == {}
        assert set(coords) == set(random_patch.coords.coord_map)


class TestValidateNoCoords:
    """Tests for stripping or rejecting coordinate metadata from attrs-like input."""

    def test_raise_on_coord_fields(self):
        """Coordinate-like flat keys should raise by default."""
        with pytest.raises(ValueError, match="distance"):
            _validate_no_coords({"distance_units": "miles"})

    def test_raise_on_coords_container(self, random_patch):
        """Nested coord containers should raise by default."""
        with pytest.raises(ValueError, match="coords"):
            _validate_no_coords({"coords": random_patch.coords})

    def test_raise_false_strips_coord_fields(self):
        """raise_error=False should remove coordinate metadata and keep attrs."""
        out = _validate_no_coords(
            {"distance_units": "miles", "tag": "x"}, raise_error=False
        )
        assert out == {"tag": "x"}

    def test_raise_false_strips_fallback_coord_fields(self):
        """Flat coord-only fields left by splitting should still be removed."""
        out = _validate_no_coords(
            {"distance_units": "miles", "time_dtype": "datetime64[ns]", "tag": "x"},
            raise_error=False,
        )
        assert out == {"tag": "x"}

    def test_raise_false_strips_coords_and_dims(self, random_patch):
        """raise_error=False should remove nested structural coord metadata."""
        out = _validate_no_coords(
            {"coords": random_patch.coords, "dims": random_patch.dims, "tag": "x"},
            raise_error=False,
        )
        assert out == {"tag": "x"}
