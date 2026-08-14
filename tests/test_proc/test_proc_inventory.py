"""Tests for copying inventory metadata onto a patch."""

from __future__ import annotations

import inspect
import itertools
import os
import pickle
import re
import tempfile
import warnings
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.constants import (
    enrich_attrs_description,
    enrich_conflict_description,
    enrich_coords_description,
    enrich_on_missing_description,
)
from dascore.core._spool_inventory import InventoryRef, resolve_row_epochs
from dascore.core.inventory import (
    _SYSTEM_FACT_NAMES,
    Acquisition,
    CoordinateReferenceSystem,
    CouplingCondition,
    DistanceMap,
    FiberArray,
    FiberSegment,
    Geometry,
    Inventory,
    Network,
    OpticalMeasurement,
    OpticalPath,
    OpticalPathAnnotation,
)
from dascore.core.inventory_loader import BLESSED_NAME
from dascore.examples import get_example_patch, inventory_patch_pair
from dascore.exceptions import (
    CoordMergeError,
    InvalidInventoryError,
    InvalidSpoolError,
    InvalidSpoolQueryError,
    ParameterError,
    PatchError,
    UnitError,
    UnresolvedPatchError,
)
from dascore.units import cm, get_quantity, m


@pytest.fixture(scope="module")
def pair():
    """The example patch and the inventory which resolves it."""
    return inventory_patch_pair()


@pytest.fixture(scope="module")
def patch(pair):
    """The example patch."""
    return pair[0]


@pytest.fixture(scope="module")
def inventory(pair):
    """The example inventory."""
    return pair[1]


def _replace_acquisition(inventory, **kwargs):
    """Return the inventory with its one acquisition changed."""
    old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
    return inventory.replace(old, old.new(**kwargs))


def _replace_path(inventory, **kwargs):
    """Return the inventory with its one optical path changed."""
    old = inventory.networks[0].fiber_arrays[0].optical_paths[0]
    return inventory.replace(old, old.new(**kwargs))


class TestResolution:
    """How a patch finds its inventory entry."""

    def test_uses_patch_id(self, patch, inventory):
        """The patch's own id needs no argument."""
        out = patch.enrich(inventory, coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_explicit_id(self, patch, inventory):
        """A patch without an id can be told which entry it is."""
        bare = patch.update_attrs(acquisition_key="")
        out = bare.enrich(inventory, acquisition_key="DAS.R2D1..RAW", coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_no_id_raises(self, patch, inventory):
        """A patch naming no entry cannot be resolved."""
        bare = patch.update_attrs(acquisition_key="")
        with pytest.raises(PatchError, match="no acquisition_key"):
            bare.enrich(inventory)

    def test_disagreeing_id_raises(self, patch, inventory):
        """Two different ids are two different data sources."""
        with pytest.raises(PatchError, match="disagree"):
            patch.enrich(inventory, acquisition_key="DAS.R2D1..OTHER")

    def test_agreeing_id_is_allowed(self, patch, inventory):
        """Repeating the patch's own id is not a disagreement."""
        out = patch.enrich(inventory, acquisition_key="DAS.R2D1..RAW", coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_unknown_id_raises(self, patch, inventory):
        """An id the inventory does not contain resolves to nothing."""
        with pytest.raises(UnresolvedPatchError, match="resolves to 0"):
            patch.update_attrs(acquisition_key="XX.R2D1..RAW").enrich(inventory)

    def test_time_with_time_axis_raises(self, patch, inventory):
        """A patch with a real time axis already says when it was recorded."""
        with pytest.raises(PatchError, match="does not also accept a time"):
            patch.enrich(inventory, time="2017-09-18")

    def test_time_without_time_axis(self, patch, inventory):
        """A lag-time patch resolves at an explicit instant."""
        lags = patch.get_coord("time").values - patch.get_coord("time").min()
        relative = patch.update_coords(time=lags)
        out = relative.enrich(inventory, time="2017-09-18", coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_no_time_and_no_axis_raises(self, patch, inventory):
        """Without either, no epoch can be chosen."""
        lags = patch.get_coord("time").values - patch.get_coord("time").min()
        relative = patch.update_coords(time=lags)
        with pytest.raises(PatchError, match="no physical time coordinate"):
            relative.enrich(inventory)

    def test_straddling_epoch_raises(self, patch, inventory):
        """Acquisition metadata is scalar per patch, so a straddle has two."""
        old = inventory.networks[0].fiber_arrays[0]
        acq = old.acquisitions[0]
        middle = patch.get_coord("time").values[len(patch.get_coord("time")) // 2]
        first = acq.new(end_time=middle, gauge_length=10.0)
        second = acq.new(start_time=middle, gauge_length=20.0)
        split = inventory.replace(old, old.new(acquisitions=(first, second)))
        with pytest.raises(PatchError, match="spans a change of acquisition"):
            patch.enrich(split, coords=False)


class TestAttrs:
    """What blanket and named attr requests copy."""

    def test_blanket_copies_system_facts(self, patch, inventory):
        """The fields the inventory is authoritative for, under its names."""
        attrs = patch.enrich(inventory, coords=False).attrs
        assert attrs.gauge_length == 10.0
        assert attrs.pulse_width == 1e-8
        assert attrs.get("interrogator.serial_number") == "sn-1"
        assert attrs.get("interrogator.model") == "FI-1"

    def test_blanket_excludes_coord_redundant(self, patch, inventory):
        """The coordinates already state these, and decimating changes them."""
        decimated = patch.decimate(time=2)
        out = decimated.enrich(inventory, coords=False)
        assert "sample_rate" not in dict(out.attrs)
        assert "spatial_interval" not in dict(out.attrs)

    def test_named_coord_redundant_restores(self, patch, inventory):
        """Naming one restores the as-acquired value, as for data state."""
        out = patch.enrich(inventory, attrs=("sample_rate",), coords=False)
        assert out.attrs.sample_rate == 250.0

    def test_blanket_excludes_data_state(self, patch, inventory):
        """Processing owns the data trio; blanket enrich must not undo it."""
        processed = patch.update_attrs(data_type="strain_rate")
        out = processed.enrich(inventory, coords=False)
        assert out.attrs.data_type == "strain_rate"

    def test_named_data_state_restores(self, patch, inventory):
        """Naming one means exactly that: restore the as-acquired value."""
        processed = patch.update_attrs(data_type="strain_rate")
        out = processed.enrich(inventory, attrs=("data_type",), coords=False)
        assert out.attrs.data_type == "velocity"

    def test_attrs_false(self, patch, inventory):
        """No attrs are copied when none are wanted."""
        out = patch.enrich(inventory, attrs=False, coords=False)
        assert "gauge_length" not in dict(out.attrs)

    def test_missing_named_attr_raises(self, patch, inventory):
        """A name the inventory does not define is an error by default."""
        with pytest.raises(PatchError, match="defines no 'pulse_rate'"):
            patch.enrich(inventory, attrs=("pulse_rate",), coords=False)

    def test_missing_named_attr_null(self, patch, inventory):
        """It can instead be filled with the missing marker."""
        out = patch.enrich(
            inventory, attrs=("pulse_rate",), coords=False, on_missing="null"
        )
        assert np.isnan(out.attrs.pulse_rate)

    def test_missing_named_attr_ignore(self, patch, inventory):
        """Or omitted entirely."""
        out = patch.enrich(
            inventory, attrs=("pulse_rate",), coords=False, on_missing="ignore"
        )
        assert "pulse_rate" not in dict(out.attrs)

    def test_missing_named_attr_warn(self, patch, inventory):
        """Or omitted with a warning, the middle of the shared vocabulary."""
        with pytest.warns(UserWarning, match="defines no 'pulse_rate'"):
            out = patch.enrich(
                inventory, attrs=("pulse_rate",), coords=False, on_missing="warn"
            )
        assert "pulse_rate" not in dict(out.attrs)

    def test_missing_named_coord_warn(self, patch, inventory):
        """The coordinate half honors the same policy."""
        with pytest.warns(UserWarning, match="defines no 'nope'"):
            out = patch.enrich(
                inventory, attrs=False, coords=("nope",), on_missing="warn"
            )
        assert "nope" not in out.coords.coord_map

    @pytest.mark.parametrize("kwargs", [{"attrs": None}, {"coords": None}])
    def test_none_is_not_the_off_switch(self, patch, inventory, kwargs):
        """False turns a half off; None is no longer a second spelling."""
        with pytest.raises(ParameterError, match="pass False to copy none"):
            patch.enrich(inventory, **kwargs)

    @pytest.mark.parametrize("kwargs", [{"attrs": None}, {"coords": None}])
    def test_spool_refuses_none_when_it_is_passed(self, patch, inventory, kwargs):
        """And the spool says so now rather than on the first patch out."""
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(ParameterError, match="pass False to copy none"):
            spool.enrich(**kwargs)

    def test_adds_nothing_unasked(self, patch, inventory):
        """Enrich never leaves the patch holding inventory state."""
        out = patch.enrich(inventory, attrs=False, coords=False)
        assert dict(out.attrs.drop("history")) == dict(patch.attrs.drop("history"))

    def test_identity_and_native_untouched(self, patch, inventory):
        """acquisition_key and tag belong to the patch, not the inventory."""
        out = patch.enrich(inventory, coords=False)
        assert out.attrs.acquisition_key == patch.attrs.acquisition_key
        assert out.attrs.tag == patch.attrs.tag


class TestConflicts:
    """How disagreements between patch and inventory are settled."""

    def test_keep_first_prefers_inventory(self, patch, inventory):
        """Enrichment puts the inventory's value first, so it wins."""
        out = patch.update_attrs(gauge_length=99.0).enrich(inventory, coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_raise_names_both_values(self, patch, inventory):
        """The misresolution guard says what disagreed and how."""
        stale = patch.update_attrs(gauge_length=99.0)
        with pytest.raises(PatchError, match=r"99\.0.*10\.0"):
            stale.enrich(inventory, coords=False, conflict="raise")

    def test_drop_removes_the_attr(self, patch, inventory):
        """A dropped conflict leaves neither value behind."""
        stale = patch.update_attrs(gauge_length=99.0)
        out = stale.enrich(inventory, coords=False, conflict="drop")
        assert "gauge_length" not in dict(out.attrs)

    def test_equal_values_are_not_conflicts(self, patch, inventory):
        """Agreement is not disagreement, whatever the policy."""
        agreeing = patch.update_attrs(gauge_length=10.0)
        out = agreeing.enrich(inventory, coords=False, conflict="raise")
        assert out.attrs.gauge_length == 10.0

    def test_filling_is_not_a_conflict(self, patch, inventory):
        """An attr the patch does not have cannot disagree with one it lacks."""
        out = patch.enrich(inventory, coords=False, conflict="raise")
        assert out.attrs.gauge_length == 10.0

    def test_re_enrich_is_a_refresh(self, patch, inventory):
        """Enriching twice is not an error and changes nothing."""
        once = patch.enrich(inventory, coords=False)
        twice = once.enrich(inventory, coords=False)
        assert dict(twice.attrs.drop("history")) == dict(once.attrs.drop("history"))

    def test_bad_conflicts_raises(self, patch, inventory):
        """The flag shares chunking's vocabulary and its validation."""
        with pytest.raises(ParameterError, match="conflict"):
            patch.enrich(inventory, conflict="whatever")

    def test_bad_on_missing_raises(self, patch, inventory):
        """An unknown policy is a caller error, not a silent default."""
        with pytest.raises(ParameterError, match="on_missing"):
            patch.enrich(inventory, on_missing="whatever")


class TestChannelResolution:
    """Placing the patch's channels on the optical path."""

    def test_distance_map_maps_instrument_meters(self, patch, inventory):
        """The measured map moves the interrogator's axis onto the path."""
        out = patch.enrich(inventory, attrs=False, coords=("x",))
        assert len(out.get_coord("x")) == len(patch.get_coord("distance"))

    def test_channel_map_needs_a_channel_coord(self, patch, inventory):
        """A channel-axis map has nothing to read on a meters patch."""
        channel_map = _replace_acquisition(
            inventory, distance_map=DistanceMap(channel=(0.0,), distance=(100.0,))
        )
        with pytest.raises(PatchError, match=r"\['channel'\]"):
            patch.enrich(channel_map, attrs=False, coords=("x",))

    def test_channel_map_with_a_channel_coord(self, patch, inventory):
        """Given channel numbers, a one-point channel map resolves them."""
        channel_map = _replace_acquisition(
            inventory, distance_map=DistanceMap(channel=(0.0,), distance=(100.0,))
        )
        channels = np.arange(len(patch.get_coord("distance")))
        with_channel = patch.update_coords(channel=("distance", channels))
        out = with_channel.enrich(channel_map, attrs=False, coords=("zone",))
        assert out.get_coord("zone").values[0] == "north"

    def test_axis_follows_the_patch(self, patch, inventory):
        """A map written in both coordinates is read on the one present."""
        both = _replace_acquisition(
            inventory,
            distance_map=DistanceMap(
                channel=(0.0, 299.0),
                instrument_distance=(0.0, 299.0),
                distance=(100.0, 399.0),
            ),
        )
        by_meters = patch.enrich(both, attrs=False, coords=("zone",))
        channels = np.arange(len(patch.get_coord("distance")))
        by_channel = patch.update_coords(channel=("distance", channels)).enrich(
            both, attrs=False, coords=("zone",)
        )
        assert list(by_meters.get_coord("zone").values) == list(
            by_channel.get_coord("zone").values
        )

    def test_optical_distance_is_requestable(self, patch, inventory):
        """The path axis can be added once it does not collide."""
        renamed = patch.rename_coords(distance="instrument_distance")
        out = renamed.enrich(inventory, attrs=False, coords=("distance",))
        assert out.get_coord("distance").min() == 100.0

    def test_distance_collision_raises(self, patch, inventory):
        """Overwriting the mapped axis would break the next resolution."""
        with pytest.raises(PatchError, match="will not overwrite"):
            patch.enrich(inventory, attrs=False, coords=("distance",))


class TestCoords:
    """What the optical path projects onto the patch."""

    def test_blanket_adds_geometry_and_groups(self, patch, inventory):
        """A blanket request copies what the path says about each channel."""
        out = patch.enrich(inventory, attrs=False)
        names = set(out.coords.coord_map)
        assert {"x", "y", "z", "zone", "noisy"}.issubset(names)

    def test_coords_false(self, patch, inventory):
        """No coordinates are added when none are wanted."""
        out = patch.enrich(inventory, attrs=False, coords=False)
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)

    def test_categorical_group(self, patch, inventory):
        """A string group is single valued, with None where uncovered."""
        out = patch.enrich(inventory, attrs=False, coords=("zone",))
        values = out.get_coord("zone").values
        assert values[0] == "north" and values[-1] == "south"

    def test_membership_group(self, patch, inventory):
        """A boolean group is False where uncovered, never null."""
        out = patch.enrich(inventory, attrs=False, coords=("noisy",))
        values = out.get_coord("noisy").values
        assert values.dtype == bool
        assert not values[0] and values[len(values) // 2]

    def test_numeric_group(self, patch, inventory):
        """A numeric group carries NaN where uncovered."""
        annotations = (
            OpticalPathAnnotation(
                start_distance=100.0, end_distance=200.0, group="frost", value=1.5
            ),
        )
        inv = _replace_path(inventory, annotations=annotations)
        out = patch.enrich(inv, attrs=False, coords=("frost",))
        values = out.get_coord("frost").values
        assert values[0] == 1.5
        assert np.isnan(values[-1])

    def test_track_field(self, patch, inventory):
        """Qualified names project one field of a typed track."""
        out = patch.enrich(inventory, attrs=False, coords=("coupling.medium",))
        values = out.get_coord("coupling.medium").values
        # A string coordinate has no null, so uncovered channels are empty.
        assert values[0] == "soil" and values[-1] == ""

    def test_component_field(self, patch, inventory):
        """Component intervals come from the cumulative layout."""
        out = patch.enrich(inventory, attrs=False, coords=("optical_components.name",))
        assert out.get_coord("optical_components.name").values[0] == "cable"

    def test_coordinate_alias(self, patch, inventory):
        """A label the CRS defines resolves to its canonical axis."""
        out = patch.enrich(inventory, attrs=False, coords=("latitude", "longitude"))
        assert out.get_coord("longitude").values[0] == -117.0

    def test_unknown_alias_is_missing(self, patch, inventory):
        """A label this CRS does not define is a name with no answer.

        It goes through on_missing like any other, rather than raising an
        inventory error the policy cannot intercept.
        """
        with pytest.raises(PatchError, match="defines no 'northing'"):
            patch.enrich(inventory, attrs=False, coords=("northing",))
        out = patch.enrich(
            inventory, attrs=False, coords=("northing",), on_missing="ignore"
        )
        assert "northing" not in set(out.coords.coord_map)

    def test_depth_alias(self, patch, inventory):
        """A depth-labeled CRS resolves depth like any other label."""
        crs = CoordinateReferenceSystem(
            coordinate_labels=("longitude", "latitude", "depth"),
            units=("degree", "degree", "meter"),
        )
        inv = inventory.new(coordinate_reference_system=crs)
        out = patch.enrich(inv, attrs=False, coords=("depth",))
        assert out.get_coord("depth").values[0] == 1500.0

    def test_existing_coord_collision_raises(self, patch, inventory):
        """A group may not silently replace a coordinate the patch has."""
        collides = patch.update_coords(
            zone=("distance", np.zeros(len(patch.get_coord("distance"))))
        )
        with pytest.raises(PatchError, match="already has a 'zone'"):
            collides.enrich(inventory, attrs=False, coords=("zone",))

    def test_missing_coord_raises(self, patch, inventory):
        """A group the path does not define is an error by default."""
        with pytest.raises(PatchError, match="defines no 'nope'"):
            patch.enrich(inventory, attrs=False, coords=("nope",))

    def test_missing_coord_null(self, patch, inventory):
        """It can instead be filled with the missing marker."""
        out = patch.enrich(inventory, attrs=False, coords=("nope",), on_missing="null")
        assert np.isnan(out.get_coord("nope").values).all()

    def test_missing_coord_ignore(self, patch, inventory):
        """Or omitted entirely."""
        out = patch.enrich(
            inventory, attrs=False, coords=("nope",), on_missing="ignore"
        )
        assert "nope" not in set(out.coords.coord_map)

    def test_blanket_without_geometry(self, patch, inventory):
        """A path with no geometry has no axes to project."""
        inv = _replace_path(inventory, geometry=())
        out = patch.enrich(inv, attrs=False)
        assert "x" not in set(out.coords.coord_map)
        assert "zone" in set(out.coords.coord_map)

    def test_point_markers_cover_nothing(self, patch, inventory):
        """An annotation marking a spot documents it without covering it."""
        annotations = (
            OpticalPathAnnotation(
                start_distance=150.0, end_distance=150.0, group="zone", value="clamp"
            ),
        )
        inv = _replace_path(inventory, annotations=annotations)
        out = patch.enrich(inv, attrs=False, coords=("zone",))
        values = out.get_coord("zone").values
        # Nothing is covered, so every channel takes the empty marker.
        assert (values == "").all()


class TestNoOpticalPath:
    """An acquisition may have no path at the patch's time."""

    @pytest.fixture(scope="class")
    def pathless(self, inventory):
        """The inventory with its optical path removed."""
        array = inventory.networks[0].fiber_arrays[0]
        return inventory.replace(array, array.new(optical_paths=()))

    def test_blanket_coords_are_empty(self, patch, pathless):
        """There is nothing to project, which is not an error."""
        out = patch.enrich(pathless, attrs=False)
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)

    def test_named_coords_raise(self, patch, pathless):
        """A named request cannot be honored and says so."""
        with pytest.raises(PatchError, match="No optical path"):
            patch.enrich(pathless, attrs=False, coords=("zone",))

    def test_attrs_still_copy(self, patch, pathless):
        """The acquisition's own facts do not depend on a path."""
        out = patch.enrich(pathless, coords=False)
        assert out.attrs.gauge_length == 10.0


class TestNoInterrogator:
    """The interrogator is optional, like every other resource."""

    def test_blanket_skips_absent_interrogator(self, patch, inventory):
        """Absent facts are absent, not empty strings."""
        inv = _replace_acquisition(inventory, interrogator=None)
        out = patch.enrich(inv, coords=False)
        assert "interrogator.serial_number" not in dict(out.attrs)
        assert out.attrs.gauge_length == 10.0


class TestExampleIsUsable:
    """The shared example pair is a working inventory."""

    def test_example_resolves(self):
        """It is checked, and its patch resolves against it."""
        patch, inventory = inventory_patch_pair()
        assert isinstance(inventory, Inventory)
        context = inventory.resolve(patch.attrs.acquisition_key, time=None)
        assert isinstance(context.acquisition, Acquisition)
        assert isinstance(context.optical_path, OpticalPath)
        assert isinstance(context.fiber_array, FiberArray)
        assert isinstance(context.network, Network)


class TestEdgeCases:
    """Paths through enrich which only unusual inventories reach."""

    def test_straddling_path_epoch_raises(self, patch, inventory):
        """A fiber which breaks mid-recording splits the patch's context."""
        array = inventory.networks[0].fiber_arrays[0]
        path = array.optical_paths[0]
        time = patch.get_coord("time")
        middle = time.values[len(time) // 2]
        first = path.new(end_time=middle)
        second = path.new(start_time=middle, name="repaired")
        split = inventory.replace(array, array.new(optical_paths=(first, second)))
        with pytest.raises(PatchError, match="spans a change of optical path"):
            patch.enrich(split, coords=False)

    def test_incomparable_attr_is_a_conflict(self, patch, inventory):
        """An attr which cannot be compared has not been shown to agree."""
        odd = patch.update_attrs(gauge_length=np.array([1.0, 2.0]))
        with pytest.raises(PatchError, match="inventory says"):
            odd.enrich(inventory, coords=False, conflict="raise")

    def test_multidimensional_channel_coord_raises(self, patch, inventory):
        """One channel coordinate maps to one dimension of the patch."""
        affine = _replace_acquisition(
            inventory, distance_map=DistanceMap(channel=(0.0,), distance=(100.0,))
        )
        shape = patch.shape
        two_d = patch.update_coords(channel=(patch.dims, np.ones(shape, dtype=float)))
        with pytest.raises(PatchError, match="exactly one dimension"):
            two_d.enrich(affine, attrs=False, coords=("zone",))

    def test_empty_track_is_missing(self, patch, inventory):
        """A track with no items defines nothing to project."""
        inv = _replace_path(inventory, coupling=())
        with pytest.raises(PatchError, match=r"defines no 'coupling\.medium'"):
            patch.enrich(inv, attrs=False, coords=("coupling.medium",))

    def test_axis_missing_from_geometry(self, patch, inventory):
        """A two-dimensional geometry has no third axis to return."""
        flat = Geometry(
            name="flat", distance=(100.0, 400.0), coordinates=((0.0, 0.0), (1.0, 1.0))
        )
        inv = _replace_path(inventory, geometry=(flat,))
        with pytest.raises(PatchError, match="defines no 'z'"):
            patch.enrich(inv, attrs=False, coords=("z",))


class TestAttachInventory:
    """Attaching carries the inventory and changes nothing else."""

    def test_attaching_does_not_enrich(self, patch, inventory):
        """Carrying an inventory is not the same as using it."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert spool._inventory is inventory
        assert "gauge_length" not in dict(spool[0].attrs)

    def test_original_is_unchanged(self, patch, inventory):
        """Attaching returns a new spool, as every spool method does."""
        plain = dc.spool(patch)
        plain.attach_inventory(inventory)
        assert plain._inventory is None

    def test_derived_spool_keeps_inventory(self, patch, inventory):
        """A selection is still the spool it came from."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert spool.select(tag="random")._inventory is inventory

    def test_equality_includes_inventory(self, patch, inventory):
        """A carried inventory is part of what the spool is."""
        plain = dc.spool(patch)
        attached = plain.attach_inventory(inventory)
        assert attached != plain
        assert attached == plain.attach_inventory(inventory)

    def test_requires_an_inventory(self, patch):
        """Anything which is neither an inventory nor a path names none."""
        with pytest.raises(ParameterError, match="needs an Inventory"):
            dc.spool(patch).attach_inventory(42)


class TestRemoveInventory:
    """Removing an inventory takes its enrichment with it."""

    def test_removes_inventory_and_enrichment(self, patch, inventory):
        """Nothing is left to enrich from, so nothing is enriched."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich().remove_inventory()
        assert spool._inventory is None
        assert spool._enrich_kwargs is None
        assert "gauge_length" not in dict(spool[0].attrs)

    def test_original_is_unchanged(self, patch, inventory):
        """Removal is a new spool; the one it came from still enriches."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        spool.remove_inventory()
        assert spool[0].attrs.gauge_length == 10.0

    def test_no_inventory_is_a_no_op(self, patch):
        """A spool with nothing to remove is returned as it stands."""
        plain = dc.spool(patch)
        assert plain.remove_inventory() == plain

    def test_keeps_the_patches(self, patch, inventory):
        """Only the metadata goes; the spool's contents do not."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert spool.remove_inventory()[0].equals(patch)


# These tests put an inventory on disk, which needs a YAML writer.  The
# rest of this module builds them in memory, and dascore runs without
# PyYAML installed -- as the free-threaded and WebAssembly jobs do.
needs_yaml = pytest.mark.skipif(
    find_spec("yaml") is None, reason="PyYAML is not installed"
)


def _enforces_permissions() -> bool:
    """Return True if a file with no mode bits is actually unreadable."""
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "probe"
        path.write_text("")
        path.chmod(0o000)
        return not os.access(path, os.R_OK)


# Asked once, at collection: root ignores modes and Windows has none, so
# there the file a permission test means to make unreadable simply is not.
ENFORCES_PERMISSIONS = _enforces_permissions()


def write_inventory(root, files):
    """Write a mapping of relative path to text as an authoring directory."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture(scope="class")
def data_directory(tmp_path_factory, patch, inventory):
    """A directory of data which carries the inventory describing it."""
    path = tmp_path_factory.mktemp("blessed")
    dc.write(patch, path / "patch.h5", "dasdae")
    inventory.to_yaml(path / f"{BLESSED_NAME}.yaml")
    return path


@needs_yaml
class TestBlessedInventory:
    """A data directory hands its own inventory to the spool over it."""

    def test_a_directory_spool_comes_attached(self, data_directory):
        """Which is the whole point: the metadata is found where it lies."""
        spool = dc.spool(data_directory).update()
        assert spool.enrich()[0].attrs.gauge_length == 10.0

    def test_the_directory_form(self, tmp_path, patch):
        """The authoring directory is the other spelling of the name."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        write_inventory(
            tmp_path / BLESSED_NAME,
            {
                "acquisitions/DAS.R2D1..RAW.yaml": (
                    "object_type: Acquisition\ndata_category: DAS\ngauge_length: 3.0\n"
                ),
            },
        )
        spool = dc.spool(tmp_path).update()
        assert spool.enrich(coords=False)[0].attrs.gauge_length == 3.0

    def test_the_scanner_ignores_it(self, tmp_path, patch):
        """A hidden name is one the file scanner already skips."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        blessed = tmp_path / BLESSED_NAME
        blessed.mkdir()
        # A readable DAS file, which is the only thing that tells being
        # skipped from being unreadable: a stray .yaml is no format the
        # scanner would have indexed anyway.
        dc.write(patch, blessed / "patch.h5", "dasdae")
        (blessed / "inventory.yaml").write_text("schema_version: 1\n")
        assert len(dc.spool(tmp_path).update()) == 1
        visible = tmp_path / "visible"
        visible.mkdir()
        dc.write(patch, visible / "patch.h5", "dasdae")
        assert len(dc.spool(tmp_path).update()) == 2

    def test_a_visible_inventory_is_not_attached(self, tmp_path, patch, inventory):
        """`inventory.yaml` names the envelope, not a spool's inventory."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        inventory.to_yaml(tmp_path / "inventory.yaml")
        assert dc.spool(tmp_path).update()._inventory is None

    def test_in_memory_spools_carry_none(self, patch):
        """There is no directory to have carried one."""
        assert dc.spool(patch)._inventory is None

    def test_a_file_spool_carries_none(self, tmp_path, patch, inventory):
        """A file is not a directory, whatever lies beside it."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        inventory.to_yaml(tmp_path / f"{BLESSED_NAME}.yaml")
        assert dc.spool(tmp_path / "patch.h5")._inventory is None


@needs_yaml
class TestLazyInventory:
    """Attaching states where an inventory is; reading it waits."""

    def test_nothing_is_read_by_opening(self, data_directory):
        """Discovery is a stat, so an archive opens as fast as ever."""
        spool = dc.spool(data_directory).update()
        assert spool._inventory._inventory is None

    def test_nothing_is_read_by_using_the_data(self, data_directory):
        """Data access is never hostage to a metadata file."""
        spool = dc.spool(data_directory).update()
        len(spool)
        spool.get_contents()
        spool.select(time=...).sort("time").chunk(time=None)
        assert "gauge_length" not in dict(spool[0].attrs)
        assert spool._inventory._inventory is None

    def test_nothing_is_read_by_unselecting_on_the_index(self, data_directory):
        """Which names it can widen a query with is known without reading."""
        spool = dc.spool(data_directory).update()
        assert len(spool.unselect(tag="random")) == 0
        assert spool._inventory._inventory is None

    def test_the_attr_names_are_the_models_own(self, inventory):
        """What `_inventory_query` decides without reading rests on this.

        It gates on a constant rather than on the attached inventory, so
        an inventory whose attrs depended on its contents would make the
        gate skip reads it needs.
        """
        assert set(inventory.get_names().attrs) == set(_SYSTEM_FACT_NAMES)
        assert set(dc.inventory().get_names().attrs) == set(_SYSTEM_FACT_NAMES)

    def test_a_fiber_coordinate_reads_it(self, data_directory):
        """A name the index does not have is a question for the inventory."""
        spool = dc.spool(data_directory).update()
        assert len(spool.select(zone="north")) == 1
        assert spool._inventory._inventory is not None

    def test_a_system_fact_reads_it(self, data_directory):
        """The name is the models', but only the inventory holds the value."""
        spool = dc.spool(data_directory).update()
        assert len(spool.select(gauge_length=10.0)) == 1
        assert spool._inventory._inventory is not None

    def test_the_first_question_reads_it(self, data_directory):
        """And answers it from what was read."""
        spool = dc.spool(data_directory).update()
        assert spool.enrich()[0].attrs.gauge_length == 10.0
        assert spool._inventory._inventory is not None

    def test_derived_spools_read_once(self, data_directory):
        """The holder is shared, so a sliced spool cannot read twice."""
        spool = dc.spool(data_directory).update()
        derived = spool.select(time=...).sort("time")
        assert derived._inventory is spool._inventory
        derived.enrich()[0]
        assert spool._inventory._inventory is not None

    def test_a_malformed_inventory_still_loads_data(self, tmp_path, patch):
        """Only the calls which need an inventory fail."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("networks: [{code: [1, 2]}]\n")
        spool = dc.spool(tmp_path).update()
        assert len(spool) == 1
        assert spool[0].equals(patch)
        with pytest.raises(InvalidInventoryError, match=re.escape(BLESSED_NAME)):
            spool.enrich()[0]

    def test_the_failure_says_where_it_came_from(self, tmp_path, patch):
        """A spool nobody attached to must say how it came to be attached."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("[]\n")
        spool = dc.spool(tmp_path).update()
        with pytest.raises(InvalidInventoryError, match="when the spool was opened"):
            spool.expand_by("zone")

    def test_an_ambiguous_name_waits_for_a_question(self, tmp_path, patch):
        """Two spellings of the name are still no reason to refuse data."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("{}\n")
        (tmp_path / BLESSED_NAME).mkdir()
        spool = dc.spool(tmp_path).update()
        assert len(spool) == 1
        with pytest.raises(InvalidInventoryError, match="more than one inventory"):
            spool.enrich()[0]

    def test_an_inventory_which_leaves_says_so(self, tmp_path, patch):
        """Removed between opening and asking, which is not "none here"."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("{}\n")
        spool = dc.spool(tmp_path).update()
        (tmp_path / f"{BLESSED_NAME}.yaml").unlink()
        with pytest.raises(InvalidInventoryError, match="nothing is there now"):
            spool.enrich()[0]

    def test_an_edit_is_not_seen_until_it_is_asked_for(
        self, tmp_path, patch, inventory
    ):
        """An inventory is an input, not a cache: it is read once."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        inventory.to_yaml(tmp_path / f"{BLESSED_NAME}.yaml")
        spool = dc.spool(tmp_path).update()
        assert spool.enrich()[0].attrs.gauge_length == 10.0
        _replace_acquisition(inventory, gauge_length=2.0).to_yaml(
            tmp_path / f"{BLESSED_NAME}.yaml"
        )
        assert spool.enrich()[0].attrs.gauge_length == 10.0
        assert spool.attach_inventory().enrich()[0].attrs.gauge_length == 2.0

    def test_a_syntax_error_says_where_it_came_from(self, tmp_path, patch):
        """A file which does not parse is an inventory which does not load."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("this: [is not: yaml\n")
        spool = dc.spool(tmp_path).update()
        with pytest.raises(InvalidInventoryError, match="when the spool was opened"):
            spool.enrich()[0]

    def test_fields_which_are_not_named(self, tmp_path, patch):
        """`1: 2` is legal YAML and no inventory; the document is blamed."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("1: 2\n")
        spool = dc.spool(tmp_path).update()
        with pytest.raises(InvalidInventoryError, match="not named"):
            spool.enrich()[0]

    @pytest.mark.skipif(
        not ENFORCES_PERMISSIONS, reason="an unreadable file is readable here"
    )
    def test_an_unreadable_file_says_where_it_came_from(self, tmp_path, patch):
        """A permission error is no more the caller's to decipher."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        path = tmp_path / f"{BLESSED_NAME}.yaml"
        path.write_text("schema_version: 1\n")
        spool = dc.spool(tmp_path).update()
        path.chmod(0o000)
        with pytest.raises(InvalidInventoryError, match="when the spool was opened"):
            spool.enrich()[0]

    def test_a_failed_read_is_not_a_read(self, tmp_path, patch, inventory):
        """An unreadable inventory is a thing to fix, so the fix is seen."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        (tmp_path / f"{BLESSED_NAME}.yaml").write_text("[]\n")
        spool = dc.spool(tmp_path).update()
        with pytest.raises(InvalidInventoryError):
            spool.enrich()[0]
        inventory.to_yaml(tmp_path / f"{BLESSED_NAME}.yaml")
        assert spool.enrich()[0].attrs.gauge_length == 10.0

    def test_a_lazily_attached_spool_pickles(self, data_directory):
        """Which is how a spool reaches a process pool at all."""
        spool = dc.spool(data_directory).update()
        restored = pickle.loads(pickle.dumps(spool))
        assert restored.enrich()[0].attrs.gauge_length == 10.0

    def test_removing_it_sticks(self, data_directory):
        """Nothing refills the slot after the spool is opened."""
        spool = dc.spool(data_directory).update().remove_inventory()
        assert spool._inventory is None
        assert spool.select(time=...)._inventory is None

    def test_update_keeps_it(self, data_directory):
        """Re-indexing the files says nothing about the inventory."""
        spool = dc.spool(data_directory)
        assert spool.update()._inventory is spool._inventory

    def test_update_does_not_bring_it_back(self, data_directory):
        """Removal outlives a re-index, since nothing refills the slot."""
        spool = dc.spool(data_directory).update().remove_inventory()
        assert spool.update()._inventory is None


@needs_yaml
class TestAttachInventoryPath:
    """An inventory can be attached by name as well as by value."""

    def test_a_file_path(self, tmp_path, patch, inventory):
        """The serialized artifact shipped beside an archive."""
        path = tmp_path / "somewhere.yaml"
        inventory.to_yaml(path)
        spool = dc.spool(patch).attach_inventory(path)
        assert spool.enrich()[0].attrs.gauge_length == 10.0

    def test_a_directory_path(self, tmp_path, patch):
        """The authoring directory itself."""
        root = write_inventory(
            tmp_path / "authored",
            {
                "acquisitions/DAS.R2D1..RAW.yaml": (
                    "object_type: Acquisition\ndata_category: DAS\ngauge_length: 4.0\n"
                ),
            },
        )
        spool = dc.spool(patch).attach_inventory(root)
        assert spool.enrich(coords=False)[0].attrs.gauge_length == 4.0

    def test_a_path_is_read_lazily(self, tmp_path, patch, inventory):
        """As the blessed name is, and for the same reason."""
        path = tmp_path / "somewhere.yaml"
        inventory.to_yaml(path)
        spool = dc.spool(patch).attach_inventory(path)
        assert spool._inventory._inventory is None

    def test_a_path_which_is_not_there(self, patch):
        """Eager, though the read is not: the caller named this one."""
        with pytest.raises(InvalidInventoryError, match="No inventory at"):
            dc.spool(patch).attach_inventory("nowhere.yaml")

    def test_a_named_failure_says_which_file(self, tmp_path, patch):
        """Named by hand, and still said to have been attached here."""
        path = tmp_path / "broken.yaml"
        path.write_text("[]\n")
        spool = dc.spool(patch).attach_inventory(path)
        with pytest.raises(InvalidInventoryError, match="attached to this spool from"):
            spool.enrich()[0]

    def test_a_named_file_of_any_suffix(self, tmp_path, patch, inventory):
        """A path is a path; only the blessed name insists on a spelling."""
        path = tmp_path / "inventory.txt"
        path.write_text(inventory.to_yaml())
        spool = dc.spool(patch).attach_inventory(path)
        assert spool.enrich()[0].attrs.gauge_length == 10.0

    def test_a_named_file_which_does_not_parse(self, tmp_path, patch):
        """Whatever its suffix, and whichever parser reached for it."""
        path = tmp_path / "inventory.txt"
        path.write_text("this: [is not: yaml\n")
        spool = dc.spool(patch).attach_inventory(path)
        with pytest.raises(InvalidInventoryError, match="attached to this spool from"):
            spool.enrich()[0]

    def test_a_path_is_anchored_when_it_is_attached(self, tmp_path, patch, inventory):
        """The read comes later, possibly from another directory entirely."""
        inventory.to_yaml(tmp_path / "somewhere.yaml")
        here = os.getcwd()
        os.chdir(tmp_path)
        try:
            spool = dc.spool(patch).attach_inventory("somewhere.yaml")
        finally:
            os.chdir(here)
        assert spool.enrich()[0].attrs.gauge_length == 10.0

    def test_conform_uses_an_attached_path(self, tmp_path, patch, inventory):
        """A lazily attached path is read by the eager verb too."""
        path = tmp_path / "somewhere.yaml"
        inventory.to_yaml(path)
        assert len(dc.spool(patch).attach_inventory(path).conform_to_inventory()) == 1

    @pytest.mark.parametrize("verb", ["enrich", "conform_to_inventory"])
    def test_the_verbs_take_no_inventory(self, patch, inventory, verb):
        """Attaching is the only way in, so neither verb accepts one."""
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(TypeError, match="positional argument"):
            getattr(spool, verb)(inventory)

    def test_no_argument_needs_a_directory(self, patch):
        """There is nowhere an in-memory spool could have carried one."""
        with pytest.raises(InvalidInventoryError, match="not opened on a directory"):
            dc.spool(patch).attach_inventory()

    def test_no_argument_needs_a_blessed_name(self, tmp_path, patch):
        """The directory is there; what it was asked for is not."""
        dc.write(patch, tmp_path / "patch.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        with pytest.raises(InvalidInventoryError, match="holds nothing named"):
            spool.attach_inventory()

    def test_re_attaching_clears_enrichment(self, data_directory):
        """As attaching an inventory by value does."""
        spool = dc.spool(data_directory).update().enrich(coords=False)
        assert spool.attach_inventory()._enrich_kwargs is None


@needs_yaml
class TestLazyInventoryEquality:
    """Comparing two attachments never reads either of them."""

    def test_two_references_to_one_place(self, data_directory):
        """The same place is the same attachment, unread."""
        first, second = dc.spool(data_directory), dc.spool(data_directory)
        assert first.update() == second.update()
        assert first._inventory._inventory is None

    def test_two_references_to_different_places(self, tmp_path, patch, inventory):
        """Two places are two attachments, whatever they hold."""
        spools = []
        for name in ("first", "second"):
            root = tmp_path / name
            root.mkdir()
            dc.write(patch, root / "patch.h5", "dasdae")
            inventory.to_yaml(root / f"{BLESSED_NAME}.yaml")
            spools.append(dc.spool(root).update())
        assert spools[0] != spools[1]
        assert all(x._inventory._inventory is None for x in spools)

    def test_a_place_and_a_value(self, data_directory, inventory):
        """A place is no value until someone asks, and this does not ask."""
        spool = dc.spool(data_directory).update()
        assert spool != spool.attach_inventory(inventory)
        assert spool._inventory._inventory is None

    def test_a_directory_and_the_inventory_it_carries(self, data_directory):
        """One path, two things: the data directory and an inventory."""
        spool = dc.spool(data_directory).update()
        named = spool.attach_inventory(data_directory / f"{BLESSED_NAME}.yaml")
        assert spool._inventory != named._inventory
        assert spool._inventory != InventoryRef(data_directory)

    def test_something_which_is_no_kind_of_inventory(self, data_directory):
        """Which the comparison declines rather than answers."""
        spool = dc.spool(data_directory).update()
        assert spool._inventory.__eq__(42) is NotImplemented
        assert spool._inventory != 42
        assert spool._inventory._inventory is None

    def test_a_union_carries_one_over(self, data_directory):
        """Both operands name the same place, so the union has an answer."""
        spool = dc.spool(data_directory).update()
        combined = spool + spool.select(time=...)
        assert combined.enrich()[0].attrs.gauge_length == 10.0

    def test_a_union_of_two_different_ones_raises(self, data_directory, inventory):
        """Two answers to one question have no combined meaning."""
        spool = dc.spool(data_directory).update()
        with pytest.raises(InvalidSpoolError, match="different inventories"):
            spool + spool.attach_inventory(inventory)

    def test_a_union_never_reads_an_inventory(self, tmp_path, patch):
        """Data access is never hostage to a metadata file, `+` included."""
        spools = []
        for name in ("first", "second"):
            root = tmp_path / name
            root.mkdir()
            dc.write(patch, root / "patch.h5", "dasdae")
            (root / f"{BLESSED_NAME}.yaml").write_text("this: [is not: yaml\n")
            spools.append(dc.spool(root).update())
        with pytest.raises(InvalidSpoolError, match="different inventories"):
            spools[0] + spools[1]
        assert len(spools[0] + spools[0].select(time=...)) == 1

    def test_equality_never_raises(self, tmp_path, patch, inventory):
        """`==` is asked in places which cannot take an exception."""
        broken, whole = tmp_path / "broken", tmp_path / "whole"
        for root in (broken, whole):
            root.mkdir()
            dc.write(patch, root / "patch.h5", "dasdae")
        (broken / f"{BLESSED_NAME}.yaml").write_text("this: [is not: yaml\n")
        inventory.to_yaml(whole / f"{BLESSED_NAME}.yaml")
        assert dc.spool(broken).update() not in [dc.spool(whole).update()]


class TestMultiValuedTrackFields:
    """A coordinate holds one value per channel, so containers cannot go in."""

    def test_mapping_field_raises_patch_error(self, patch, inventory):
        """`extra_fields` is a real field of every track, and holds a mapping."""
        old = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        coupling = old.coupling[0]
        inv = inventory.replace(
            old, old.new(coupling=(coupling.new(extra_fields={"a": 1}),))
        )
        with pytest.raises(PatchError, match="multi-valued"):
            patch.enrich(inv, attrs=False, coords=("coupling.extra_fields",))


class TestOnUnresolved:
    """What a spool does with a patch its inventory does not describe."""

    @pytest.fixture
    def mixed(self, patch):
        """A spool of one patch the example inventory knows and one it does not."""
        return dc.spool([patch, dc.get_example_patch("random_das")])

    def test_warns_and_leaves_the_patch(self, mixed, inventory):
        """Enriching decides metadata, so the patch itself still comes out."""
        with pytest.warns(UserWarning, match="does not describe every patch"):
            out = list(mixed.attach_inventory(inventory).enrich())
        assert len(out) == len(mixed)
        assert out[0].attrs.gauge_length == 10.0
        assert "gauge_length" not in dict(out[1].attrs)

    def test_warning_does_not_scale_with_the_spool(self, patch, inventory):
        """The default exists for partly-covered archives, so it must not
        emit one warning per distinct key on exactly those.
        """
        strangers = [
            patch.update_attrs(acquisition_key=f"XX.A{i}..RAW") for i in range(25)
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            list(dc.spool(strangers).attach_inventory(inventory).enrich())
        assert len(caught) == 1

    def test_each_spool_says_it_once(self, patch, inventory):
        """Deduping on the message text would silence every spool after the
        first in a session, hiding the second one's problem entirely.
        """
        first = dc.spool([patch.update_attrs(acquisition_key="XX.A1..RAW")])
        second = dc.spool([patch.update_attrs(acquisition_key="XX.B2..RAW")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            list(first.attach_inventory(inventory).enrich())
            list(second.attach_inventory(inventory).enrich())
        assert len(caught) == 2

    def test_ignore_is_silent(self, mixed, inventory):
        """The same, without saying so."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = list(mixed.attach_inventory(inventory).enrich(on_unresolved="ignore"))
        assert len(out) == len(mixed)

    def test_raise_fails(self, mixed, inventory):
        """A spool which must be fully described can say so."""
        with pytest.raises(UnresolvedPatchError, match="no acquisition_key"):
            list(mixed.attach_inventory(inventory).enrich(on_unresolved="raise"))

    def test_unknown_id_is_unresolved(self, patch, inventory):
        """Naming an entry the inventory lacks is not being described either."""
        stranger = patch.update_attrs(acquisition_key="XX.R2D1..RAW")
        with pytest.warns(UserWarning, match="does not describe every patch"):
            out = dc.spool(stranger).attach_inventory(inventory).enrich()[0]
        assert "gauge_length" not in dict(out.attrs)

    def test_straddling_patch_still_raises(self, patch, inventory):
        """Described twice is not the same as not described; it needs splitting."""
        array = inventory.networks[0].fiber_arrays[0]
        acquisition = array.acquisitions[0]
        times = patch.get_coord("time").values
        middle = times[len(times) // 2]
        split = inventory.replace(
            array,
            array.new(
                acquisitions=(
                    acquisition.new(end_time=middle),
                    acquisition.new(start_time=middle, gauge_length=20.0),
                )
            ),
        )
        spool = dc.spool(patch).attach_inventory(split).enrich(on_unresolved="ignore")
        with pytest.raises(PatchError, match="spans a change of acquisition"):
            spool[0]

    def test_malformed_explicit_id_is_not_swallowed(self, patch, inventory):
        """A caller getting the argument wrong is not the inventory's silence."""
        bare = dc.spool(patch.update_attrs(acquisition_key=""))
        spool = bare.attach_inventory(inventory).enrich(
            acquisition_key="broken", on_unresolved="ignore"
        )
        with pytest.raises(ValueError, match="Invalid acquisition_key"):
            spool[0]

    def test_bad_policy_raises(self, patch, inventory):
        """An unknown policy names no behavior."""
        with pytest.raises(ParameterError, match="on_unresolved must be one of"):
            dc.spool(patch).attach_inventory(inventory).enrich(on_unresolved="nope")

    def test_policy_is_part_of_the_spool(self, patch, inventory):
        """Two policies mean two different behaviors on extraction."""
        plain = dc.spool(patch)
        assert plain.attach_inventory(inventory).enrich() != plain.attach_inventory(
            inventory
        ).enrich(on_unresolved="raise")
        attached = plain.attach_inventory(inventory)
        assert attached.enrich() == attached.enrich(on_unresolved="warn")

    def test_policy_survives_a_union(self, patch, inventory):
        """The policy is part of the enrichment, so it carries with it."""
        enriched = (
            dc.spool(patch).attach_inventory(inventory).enrich(on_unresolved="raise")
        )
        assert (enriched + dc.spool(patch))._on_unresolved == "raise"


class TestInventoryQueryError:
    """An attached inventory changes what an unknown query name means."""

    def test_inventory_field_is_answered_by_the_inventory(self, patch, inventory):
        """The index's 'no such attribute' would deny a field which exists."""
        spool = dc.spool(patch).attach_inventory(inventory)
        # No channel is coupled in cement, so the field answers with none.
        assert len(spool.select(coupling="cement")) == 0

    def test_plain_spool_keeps_its_message(self, patch):
        """With no inventory there is nothing to add."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute") as ex:
            dc.spool(patch).select(nope=1)
        assert "inventory" not in str(ex.value)

    def test_valid_query_still_works(self, patch, inventory):
        """Naming an indexed attr is unaffected by the inventory."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert len(spool.select(tag="random")) == 1


class TestSpoolEnrich:
    """The spool enriches each patch as it is extracted."""

    def test_getitem_is_enriched(self, patch, inventory):
        """A patch pulled by index arrives with its metadata."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert spool[0].attrs.gauge_length == 10.0

    def test_iteration_is_enriched(self, patch, inventory):
        """So does one pulled by iteration."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert all(x.attrs.gauge_length == 10.0 for x in spool)

    def test_enrich_keeps_the_attached_inventory(self, patch, inventory):
        """The spool carries what it enriches from."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert spool._inventory is inventory

    def test_kwargs_pass_through(self, patch, inventory):
        """Enrich's arguments are the spool's arguments."""
        spool = (
            dc.spool(patch)
            .attach_inventory(inventory)
            .enrich(attrs=("gauge_length",), coords=False)
        )
        out = spool[0]
        assert out.attrs.gauge_length == 10.0
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)

    def test_derived_spool_stays_enriched(self, patch, inventory):
        """A selection is still the spool it came from."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert spool.select(tag="random")[0].attrs.gauge_length == 10.0

    def test_original_is_unchanged(self, patch, inventory):
        """Setting up enrichment leaves the spool it came from alone."""
        plain = dc.spool(patch)
        plain.attach_inventory(inventory).enrich()
        assert "gauge_length" not in dict(plain[0].attrs)

    def test_equality_includes_arguments(self, patch, inventory):
        """Different arguments mean different patches come out."""
        attached = dc.spool(patch).attach_inventory(inventory)
        assert attached.enrich() != attached
        assert attached.enrich() == attached.enrich()
        assert attached.enrich() != attached.enrich(coords=False)

    def test_requires_an_inventory(self, patch):
        """With none attached and none given there is nothing to copy."""
        with pytest.raises(ParameterError, match="needs an inventory"):
            dc.spool(patch).enrich()

    def test_unknown_kwarg_raises_now(self, patch, inventory):
        """A misspelled argument fails here, not on a patch pulled later."""
        with pytest.raises(ParameterError, match="unknown argument"):
            dc.spool(patch).attach_inventory(inventory).enrich(attr=True)

    def test_re_enriching_replaces_arguments(self, patch, inventory):
        """The last call says what enrichment means."""
        attached = dc.spool(patch).attach_inventory(inventory)
        spool = attached.enrich(coords=False).enrich()
        assert "zone" in set(spool[0].coords.coord_map)

    def test_documents_the_arguments_it_forwards(self):
        """
        Spool.enrich documents the arguments it holds, from one source.

        The two enrich methods take the same arguments and describing them
        twice is how they drift, so both compose the same fragments.
        """
        spool_doc = dc.core.spool.Spool.enrich.__doc__
        patch_doc = dc.Patch.enrich.__doc__
        forwarded = set(inspect.signature(dc.Patch.enrich).parameters) - {
            "patch",
            "inventory",
        }
        for fragment in (
            enrich_attrs_description,
            enrich_coords_description,
            enrich_on_missing_description,
            enrich_conflict_description,
        ):
            # Indentation differs between a function and a method body, so
            # the shared text is compared with it removed.
            body = " ".join(fragment.split())
            assert body in " ".join(spool_doc.split())
            assert body in " ".join(patch_doc.split())
        for doc, name in itertools.product((spool_doc, patch_doc), forwarded):
            # Whatever it is indented by: Python 3.13 dedents a docstring
            # as it compiles it, so the depth is not the same everywhere.
            assert re.search(rf"^\s*{re.escape(name)}$", doc, re.MULTILINE), name


class TestReviewFindings:
    """Defects found reviewing the first cut of enrich."""

    def test_geometry_coords_carry_crs_units(self, patch, inventory):
        """The CRS states its units, so the coordinates carry them."""
        out = patch.enrich(inventory, attrs=False, coords=("x", "z"))
        assert out.get_coord("x").units == dc.get_quantity("degree")
        assert out.get_coord("z").units == dc.get_quantity("meter")

    def test_optical_distance_is_in_meters(self, patch, inventory):
        """Path lengths are meters, so path distance is too."""
        renamed = patch.rename_coords(distance="instrument_distance")
        out = renamed.enrich(inventory, attrs=False, coords=("distance",))
        assert out.get_coord("distance").units == dc.get_quantity("m")

    def test_default_enrich_is_idempotent(self, patch, inventory):
        """Re-enriching is a refresh, coordinates included."""
        once = patch.enrich(inventory)
        twice = once.enrich(inventory)
        assert set(twice.coords.coord_map) == set(once.coords.coord_map)
        assert twice.equals(once)

    def test_disagreeing_coord_still_raises(self, patch, inventory):
        """A coordinate the inventory contradicts is not silently replaced."""
        wrong = patch.enrich(inventory, attrs=False, coords=("zone",))
        values = np.array(["elsewhere"] * len(wrong.get_coord("zone")))
        wrong = wrong.update_coords(zone=("distance", values))
        with pytest.raises(PatchError, match="does not agree"):
            wrong.enrich(inventory, attrs=False, coords=("zone",))

    def test_no_map_raises_when_coords_are_wanted(self, patch, inventory):
        """Without a map there is nothing to project the tracks onto."""
        no_map = _replace_acquisition(inventory, distance_map=None)
        with pytest.raises(PatchError, match="no distance_map"):
            patch.enrich(no_map, attrs=False, coords=("zone",))

    def test_empty_coord_request_needs_no_map(self, patch, inventory):
        """Asking for no coordinates asks nothing of the channel map."""
        no_map = _replace_acquisition(inventory, distance_map=None)
        out = patch.enrich(no_map, coords=())
        assert out.attrs.gauge_length == 10.0

    def test_blanket_needs_no_map_when_path_is_bare(self, patch, inventory):
        """A path with nothing to project asks nothing of the map either."""
        no_map = _replace_acquisition(inventory, distance_map=None)
        bare = _replace_path(no_map, geometry=(), annotations=())
        out = patch.enrich(bare)
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)

    def test_unset_track_field_is_missing(self, patch, inventory):
        """A field no interval states defines nothing, so on_missing rules."""
        with pytest.raises(PatchError, match=r"defines no 'coupling\.depth'"):
            patch.enrich(inventory, attrs=False, coords=("coupling.depth",))

    def test_unknown_track_field_is_missing(self, patch, inventory):
        """A misspelled field is missing rather than an all-null coordinate."""
        out = patch.enrich(
            inventory, attrs=False, coords=("coupling.nope",), on_missing="ignore"
        )
        assert "coupling.nope" not in set(out.coords.coord_map)

    def test_track_field_units(self, patch, inventory):
        """Track fields the inventory documents in meters say so."""
        old = inventory.networks[0].fiber_arrays[0].optical_paths[0].coupling[0]
        inv = _replace_path(inventory, coupling=(old.new(depth=2.0),))
        out = patch.enrich(inv, attrs=False, coords=("coupling.depth",))
        assert out.get_coord("coupling.depth").units == dc.get_quantity("m")

    def test_union_carries_the_inventory(self, patch, inventory):
        """Combining spools keeps metadata the operands already had."""
        enriched = dc.spool(patch).attach_inventory(inventory).enrich()
        combined = enriched + dc.spool(patch)
        assert combined[0].attrs.gauge_length == 10.0
        assert (dc.spool(patch) + enriched)[0].attrs.gauge_length == 10.0

    def test_union_of_different_enrichment_raises(self, patch, inventory):
        """Two ways of enriching have no combined meaning."""
        first = dc.spool(patch).attach_inventory(inventory).enrich()
        second = dc.spool(patch).attach_inventory(inventory).enrich(coords=False)
        with pytest.raises(InvalidSpoolError, match="different enrich arguments"):
            first + second

    def test_union_of_different_inventories_raises(self, patch, inventory):
        """Neither do two different inventories."""
        other = _replace_acquisition(inventory, gauge_length=20.0)
        first = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolError, match="different inventories"):
            first + dc.spool(patch).attach_inventory(other)


class TestSecondReviewFindings:
    """Defects found by the adversarial reviews of the first fixes."""

    def test_endpoint_belongs_to_its_own_run(self, patch, inventory):
        """A track's coverage end is local: a later interval cannot move it."""
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        far = path.coupling[0].new(start_distance=300.0, end_distance=400.0)
        with_far = _replace_path(inventory, coupling=(path.coupling[0], far))
        near = patch.enrich(inventory, attrs=False, coords=("coupling.medium",))
        both = with_far.networks and patch.enrich(
            with_far, attrs=False, coords=("coupling.medium",)
        )
        # channel 150 sits at path distance 250, the end of the first run
        assert near.get_coord("coupling.medium").values[150] == "soil"
        assert both.get_coord("coupling.medium").values[150] == "soil"

    def test_geometry_endpoint_is_local(self, patch, inventory):
        """The same rule holds for the geometry track."""
        first = Geometry(distance=(100.0, 200.0), coordinates=((0.0, 0.0), (1.0, 1.0)))
        second = Geometry(distance=(300.0, 400.0), coordinates=((3.0, 3.0), (4.0, 4.0)))
        inv = _replace_path(inventory, geometry=(first, second))
        out = patch.enrich(inv, attrs=False, coords=("x",))
        # channel 100 is path distance 200, the last point of the first segment
        assert out.get_coord("x").values[100] == 1.0

    def test_multivalued_track_field_raises(self, patch, inventory):
        """A per-wavelength loss is not one value per channel."""
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        component = path.optical_components[0]
        measurements = (
            OpticalMeasurement(wavelength=1550.0),
            OpticalMeasurement(wavelength=1625.0),
        )
        multi = component.new(loss_db=(0.5, 0.3), loss_measurement=measurements)
        inv = _replace_path(inventory, optical_components=(multi,))
        with pytest.raises(PatchError, match="multi-valued"):
            patch.enrich(inv, attrs=False, coords=("optical_components.loss_db",))

    def test_nan_attr_is_filled_not_conflicted(self, patch, inventory):
        """NaN is how a reader spells unknown, so the inventory fills it."""
        unknown = patch.update_attrs(gauge_length=np.nan)
        out = unknown.enrich(inventory, coords=False, conflict="raise")
        assert out.attrs.gauge_length == 10.0

    def test_nan_marker_round_trips(self, patch, inventory):
        """Re-enriching a nan-filled attr is a refresh, not a conflict."""
        once = patch.enrich(
            inventory, attrs=("pulse_rate",), coords=False, on_missing="null"
        )
        twice = once.enrich(
            inventory,
            attrs=("pulse_rate",),
            coords=False,
            on_missing="null",
            conflict="raise",
        )
        assert np.isnan(twice.attrs.pulse_rate)

    def test_missing_marker_matches_the_field(self, patch, inventory):
        """A string field's missing marker is not a float."""
        out = patch.enrich(
            inventory, attrs=("firmware_version",), coords=False, on_missing="null"
        )
        assert out.attrs.firmware_version is None

    def test_geometryless_path_honors_on_missing(self, patch, inventory):
        """Every axis of a path with no geometry is missing, not all-NaN."""
        inv = _replace_path(inventory, geometry=())
        with pytest.raises(PatchError, match="defines no 'x'"):
            patch.enrich(inv, attrs=False, coords=("x",))

    def test_single_point_instrument_map_is_an_offset(self, patch, inventory):
        """Interrogator meters map onto path meters one for one."""
        acq = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        one_point = acq.new(
            spatial_interval=2.0,
            distance_map=DistanceMap(instrument_distance=(0.0,), distance=(100.0,)),
        )
        inv = inventory.replace(acq, one_point)
        renamed = patch.rename_coords(distance="instrument_distance")
        out = renamed.enrich(inv, attrs=False, coords=("distance",))
        assert out.get_coord("distance").values[10] == 110.0

    def test_reserved_annotation_group_raises(self, inventory):
        """A group named after a coordinate would shadow it at enrichment."""
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        annotation = OpticalPathAnnotation(
            start_distance=100.0, end_distance=200.0, group="time", value=True
        )
        with pytest.raises(InvalidInventoryError, match="reserved name"):
            path.new(annotations=(annotation,)).check()

    def test_boolean_group_is_a_union(self, patch, inventory):
        """Membership groups overlap, so any covering true interval wins."""
        annotations = (
            OpticalPathAnnotation(
                start_distance=100.0, end_distance=400.0, group="wet", value=True
            ),
            OpticalPathAnnotation(
                start_distance=200.0, end_distance=300.0, group="wet", value=False
            ),
        )
        inv = _replace_path(inventory, annotations=annotations)
        out = patch.enrich(inv, attrs=False, coords=("wet",))
        assert out.get_coord("wet").values.all()

    def test_missing_marker_for_a_non_scalar_field(self, patch, inventory):
        """A field which is neither text nor number has no numeric marker."""
        out = patch.enrich(
            inventory, attrs=("data_units",), coords=False, on_missing="null"
        )
        assert out.attrs.data_units is None

    def test_coord_unit_change_is_a_disagreement(self, patch, inventory):
        """The same numbers in other units are other coordinates."""
        enriched = patch.enrich(inventory, attrs=False, coords=("x",))
        stripped = enriched.update_coords(x=("distance", enriched.get_array("x")))
        with pytest.raises(PatchError, match="does not agree"):
            stripped.enrich(inventory, attrs=False, coords=("x",))

    def test_coord_dtype_change_is_a_disagreement(self, patch, inventory):
        """A coordinate of another dtype is not the one enrich would add."""
        floats = np.zeros(len(patch.get_coord("distance")))
        collides = patch.update_coords(noisy=("distance", floats))
        with pytest.raises(PatchError, match="does not agree"):
            collides.enrich(inventory, attrs=False, coords=("noisy",))

    def test_disagreeing_axes_raise(self, patch, inventory):
        """A patch which contradicts the map about its own channels raises."""
        stretched = _replace_acquisition(
            inventory,
            distance_map=DistanceMap(
                channel=(0.0, 299.0),
                instrument_distance=(0.0, 598.0),
                distance=(100.0, 399.0),
            ),
        )
        channels = np.arange(len(patch.get_coord("distance")))
        both = patch.update_coords(channel=("distance", channels))
        with pytest.raises(PatchError, match="m apart on the path"):
            both.enrich(stretched, attrs=False, coords=("zone",))

    def test_agreeing_axes_are_fine(self, patch, inventory):
        """Two coordinates which say the same thing are not a contradiction."""
        both_axes = _replace_acquisition(
            inventory,
            distance_map=DistanceMap(
                channel=(0.0, 299.0),
                instrument_distance=(0.0, 299.0),
                distance=(100.0, 399.0),
            ),
        )
        channels = np.arange(len(patch.get_coord("distance")))
        both = patch.update_coords(channel=("distance", channels))
        out = both.enrich(both_axes, attrs=False, coords=("zone",))
        assert out.get_coord("zone").values[0] == "north"

    def test_axes_on_different_dimensions_raise(self, patch, inventory):
        """A coordinate named channel which is not the channel axis is caught."""
        both_axes = _replace_acquisition(
            inventory,
            distance_map=DistanceMap(
                channel=(0.0, 299.0),
                instrument_distance=(0.0, 299.0),
                distance=(100.0, 399.0),
            ),
        )
        mislabeled = patch.update_coords(
            channel=("time", np.arange(len(patch.get_coord("time"))))
        )
        with pytest.raises(PatchError, match="belong to different dimensions"):
            mislabeled.enrich(both_axes, attrs=False, coords=("zone",))

    def test_unreadable_axis_leaves_the_others(self, patch, inventory):
        """A channel axis with no spacing does not veto the other axis."""
        no_slope = _replace_acquisition(
            inventory,
            spatial_interval=None,
            distance_map=DistanceMap(
                channel=(0.0,), instrument_distance=(0.0,), distance=(100.0,)
            ),
        )
        channels = np.arange(len(patch.get_coord("distance")))
        both = patch.update_coords(channel=("distance", channels))
        out = both.enrich(no_slope, attrs=False, coords=("zone",))
        assert out.get_coord("zone").values[0] == "north"

    def test_no_readable_axis_raises(self, patch, inventory):
        """When no axis can be read, the reasons are reported together."""
        no_slope = _replace_acquisition(
            inventory,
            spatial_interval=None,
            distance_map=DistanceMap(channel=(0.0,), distance=(100.0,)),
        )
        channels = np.arange(len(patch.get_coord("distance")))
        with_channel = patch.update_coords(channel=("distance", channels))
        with pytest.raises(PatchError, match="None of the patch's coordinates"):
            with_channel.enrich(no_slope, attrs=False, coords=("zone",))

    def test_attached_spool_survives_pickle(self, patch, inventory):
        """A spool and its pickle are the same spool, inventory and all."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert pickle.loads(pickle.dumps(spool)) == spool

    def test_update_coords_stays_safe_to_bypass(self):
        """Enrich calls update_coords.raw_function to skip its history entry.

        That is only legal while the wrapper does nothing else: any
        requirement or data_type added to update_coords would be silently
        skipped for enriched patches.
        """
        from dascore.proc.coords import update_coords  # noqa: PLC0415

        cells = dict(
            zip(
                update_coords.__code__.co_freevars,
                [x.cell_contents for x in update_coords.__closure__],
                strict=True,
            )
        )
        for name in ("required_dims", "required_coords", "required_attrs", "data_type"):
            assert cells[name] is None, (
                f"update_coords now sets {name!r}, which Patch.enrich would "
                "silently skip by calling raw_function."
            )

    def test_enrich_writes_one_history_entry(self, patch, inventory):
        """The operation is enrich; how it updates coords is its own business."""
        history = patch.enrich(inventory).attrs.history
        assert len(history) == 1
        assert history[0].startswith("enrich(")


class TestSplitReviewFindings:
    """Defects found reviewing the attach/enrich/remove split."""

    def test_attaching_clears_enrichment(self, patch, inventory):
        """Swapping the source under a configured enrichment would rewrite
        every patch's metadata without being asked.
        """
        spool = (
            dc.spool(patch)
            .attach_inventory(inventory)
            .enrich()
            .attach_inventory(inventory)
        )
        assert spool._enrich_kwargs is None
        assert "gauge_length" not in dict(spool[0].attrs)

    def test_attaching_the_same_inventory_keeps_a_union_working(self, patch, inventory):
        """Supplying more information must not break what already worked."""
        enriched = dc.spool(patch).attach_inventory(inventory).enrich()
        attached = dc.spool(patch).attach_inventory(inventory)
        assert (enriched + attached)[0].attrs.gauge_length == 10.0

    def test_stating_a_default_does_not_change_the_spool(self, patch, inventory):
        """Spools which enrich identically are the same spool."""
        plain = dc.spool(patch)
        attached = plain.attach_inventory(inventory)
        assert attached.enrich() == attached.enrich(
            attrs=True, coords=True, conflict="keep_first"
        )

    def test_name_collections_compare_by_content(self, patch, inventory):
        """A list of names asks for what a tuple of them asks for."""
        plain = dc.spool(patch)
        tupled = plain.attach_inventory(inventory).enrich(attrs=("gauge_length",))
        assert tupled == plain.attach_inventory(inventory).enrich(
            attrs=["gauge_length"]
        )

    def test_update_keeps_the_inventory(self, patch, inventory, tmp_path):
        """Re-reading the file is no reason to stop enriching."""
        path = tmp_path / "patch.h5"
        dc.write(patch, path, "dasdae")
        spool = dc.spool(path).attach_inventory(inventory).enrich()
        updated = spool.update()
        assert updated._inventory is inventory
        assert updated[0].attrs.gauge_length == 10.0


class TestPartialStringCoverage:
    """A track covering part of a patch still yields a usable patch."""

    def test_uncovered_string_channels_are_a_str_coord(self, patch, inventory):
        """A coordinate has one dtype; None beside strings makes an object
        array which cannot be written, chunked, or sorted.
        """
        out = patch.enrich(inventory, attrs=False, coords=("coupling.medium",))
        values = out.get_coord("coupling.medium").values
        assert values.dtype.kind in "US"
        # The example coupling covers 100-250 m of a patch spanning 100-399.
        assert (values == "").any() and (values != "").any()

    def test_partly_covered_patch_can_be_written(self, patch, inventory, tmp_path):
        """The failure this guards against was a write, not a read."""
        out = patch.enrich(inventory, attrs=False, coords=("coupling.medium",))
        path = dc.write(out, tmp_path / "partial.h5", "dasdae")
        back = dc.read(path)[0]
        assert np.array_equal(
            back.get_coord("coupling.medium").values,
            out.get_coord("coupling.medium").values,
        )


class TestEmptyIsUnambiguous:
    """Absence has one spelling, so nothing legitimate can wear it."""

    def test_empty_annotation_value_rejected(self):
        """A group saying nothing would read as an uncovered channel."""
        with pytest.raises(ValidationError, match="may not be the empty string"):
            OpticalPathAnnotation(
                start_distance=0.0, end_distance=10.0, group="zone", value=""
            )

    def test_empty_key_is_not_resolvable(self, inventory):
        """The empty key is legal on a patch and names no entry."""
        with pytest.raises(InvalidInventoryError, match="empty acquisition_key"):
            inventory.resolve("")


class TestThirdReviewFindings:
    """Defects found reviewing the post-merge fixes."""

    def test_re_enriching_a_selection_is_still_a_refresh(self, patch, inventory):
        """A string coordinate's width is fixed by the longest value present.

        A patch sliced down to the short values keeps the wider dtype, so
        comparing dtypes exactly made a fresh projection of the same values
        look like a disagreement.
        """
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        coupling = path.coupling[0]
        inv = _replace_path(
            inventory,
            coupling=(
                coupling.new(end_distance=200.0),
                coupling.new(
                    start_distance=200.0,
                    end_distance=300.0,
                    medium="clay_and_gravel_backfill",
                ),
            ),
        )
        full = patch.enrich(inv, attrs=False, coords=("coupling.medium",))
        narrow = full.select(distance=(0, 40), samples=True)
        out = narrow.enrich(inv, attrs=False, coords=("coupling.medium",))
        assert set(out.get_coord("coupling.medium").values) == {"soil"}

    def test_a_field_no_interval_records_is_missing(self, patch, inventory):
        """Every interval leaving a field at its empty default defines nothing.

        Handing back a blank coordinate would say the inventory had an
        answer, and would slip past on_missing entirely.
        """
        with pytest.raises(PatchError, match=r"defines no 'coupling\.attachment'"):
            patch.enrich(inventory, attrs=False, coords=("coupling.attachment",))

    def test_overlong_key_is_malformed_not_unknown(self, patch, inventory):
        """PatchAttrs bounds the field, so the validator must bound it too.

        Otherwise a key too long to store reads as merely unknown to the
        inventory, which on_unresolved is allowed to wave through.
        """
        long_key = "DAS." + "A" * 70 + "..RAW"
        with pytest.raises(InvalidInventoryError, match="characters"):
            inventory.resolve(long_key)
        bare = dc.spool(patch.update_attrs(acquisition_key=""))
        spool = bare.attach_inventory(inventory).enrich(
            acquisition_key=long_key, on_unresolved="ignore"
        )
        with pytest.raises(InvalidInventoryError, match="characters"):
            spool[0]


@pytest.fixture(scope="module")
def two_patch_spool(patch):
    """Two patches from one acquisition, neither stating gauge_length."""
    return dc.spool([patch, patch.update_attrs(tag="second")])


class TestInventorySelect:
    """Selecting a spool on the facts an inventory states."""

    def test_selects_on_an_unstated_attr(self, two_patch_spool, inventory):
        """A name no patch states is answered by the inventory."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert "gauge_length" not in spool.get_contents().columns
        assert len(spool.select(gauge_length=10.0)) == 2

    def test_non_matching_value_selects_nothing(self, two_patch_spool, inventory):
        """The inventory's answer has to match to keep a patch."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert len(spool.select(gauge_length=99.0)) == 0

    def test_needs_an_inventory(self, two_patch_spool):
        """Without one the name is simply not something the spool knows."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            two_patch_spool.select(gauge_length=10.0)

    def test_len_is_exact_and_nothing_is_read(self, two_patch_spool, inventory):
        """Selection is metadata work: the count is final and exact."""
        spool = two_patch_spool.attach_inventory(inventory)
        out = spool.select(gauge_length=10.0)
        assert len(out) == len(out.get_contents()) == 2

    def test_dotted_names_select(self, two_patch_spool, inventory):
        """An interrogator fact is reached by its qualified name."""
        spool = two_patch_spool.attach_inventory(inventory)
        query = {"interrogator.model": "FI-1"}
        assert len(spool.select(**query)) == 2
        assert len(spool.select(_attrs=query)) == 2
        assert len(spool.select(**{"interrogator.model": "nope"})) == 0

    def test_selector_shapes(self, two_patch_spool, inventory):
        """A range, a collection, and a glob mean what they do on the index."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert len(spool.select(gauge_length=(5.0, 15.0))) == 2
        assert len(spool.select(gauge_length=(15.0, 25.0))) == 0
        assert len(spool.select(gauge_length=[10.0, 20.0])) == 2
        assert len(spool.select(gauge_length=[20.0, 30.0])) == 0
        assert len(spool.select(**{"interrogator.model": "FI-*"})) == 2
        assert len(spool.select(**{"interrogator.model": "XX-*"})) == 0

    def test_names_combine_with_and(self, two_patch_spool, inventory):
        """Two inventory-backed selectors both have to hold."""
        spool = two_patch_spool.attach_inventory(inventory)
        both = spool.select(gauge_length=10.0, **{"interrogator.model": "FI-1"})
        assert len(both) == 2
        one = spool.select(gauge_length=10.0, **{"interrogator.model": "nope"})
        assert len(one) == 0

    def test_combines_with_index_selectors(self, two_patch_spool, inventory):
        """An index name and an inventory name compose in one call."""
        spool = two_patch_spool.attach_inventory(inventory)
        out = spool.select(tag="second", gauge_length=10.0)
        assert len(out) == 1
        assert out.get_contents()["tag"].tolist() == ["second"]

    def test_undescribed_patch_is_not_selected(self, patch, inventory):
        """
        A patch the inventory does not describe silently does not match.

        Select is a filter, and a patch with no entry has no gauge length
        any more than one which never states the attr does.
        """
        other = patch.update_attrs(tag="other", acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        out = spool.select(gauge_length=10.0)
        assert len(out) == 1
        assert out.get_contents()["tag"].tolist() == ["random"]

    def test_patch_without_identity_is_not_selected(self, patch, inventory):
        """One naming no entry resolves to nothing, and matches nothing."""
        bare = patch.update_attrs(tag="bare", acquisition_key="")
        spool = dc.spool([patch, bare]).attach_inventory(inventory)
        assert len(spool.select(gauge_length=10.0)) == 1

    def test_stated_value_wins(self, patch, inventory):
        """
        Precedence is per row: the patch states it, the inventory fills in.

        A file which recorded its own gauge length is the authority for
        that patch even when the inventory disagrees, and the inventory
        answers only for the patches which left it unstated.
        """
        stated = patch.update_attrs(tag="stated", gauge_length=20.0)
        spool = dc.spool([patch, stated]).attach_inventory(inventory)
        by_patch = spool.select(gauge_length=20.0)
        by_inventory = spool.select(gauge_length=10.0)
        assert by_patch.get_contents()["tag"].tolist() == ["stated"]
        assert by_inventory.get_contents()["tag"].tolist() == ["random"]

    def test_empty_string_is_unstated(self, patch, inventory):
        """
        A string column spells "not known" as the empty string.

        It is a value the index stores and can compare, which is exactly
        why the inventory has to be asked for it anyway.
        """
        described = _replace_acquisition(inventory, firmware_version="v1.2.3")
        spool = dc.spool(
            [
                patch.update_attrs(tag="blank", firmware_version=""),
                patch.update_attrs(tag="stated", firmware_version="v9"),
            ]
        ).attach_inventory(described)
        assert "firmware_version" in spool.get_contents().columns
        assert spool.select(firmware_version="v1.2.3").get_contents()[
            "tag"
        ].tolist() == ["blank"]
        assert spool.select(firmware_version="v9").get_contents()["tag"].tolist() == [
            "stated"
        ]

    def test_straddling_patch_is_not_selected(self, patch, inventory):
        """
        A patch described twice has no single answer, so it matches neither.

        Subdividing it is `conform_to_inventory`'s job; until it runs, a
        straddling patch is one the selection cannot speak for.
        """
        old = inventory.networks[0].fiber_arrays[0]
        acq = old.acquisitions[0]
        times = patch.get_coord("time").values
        middle = times[len(times) // 2]
        split = inventory.replace(
            old,
            old.new(
                acquisitions=(
                    acq.new(end_time=middle, gauge_length=10.0),
                    acq.new(start_time=middle, gauge_length=20.0),
                )
            ),
        )
        spool = dc.spool([patch]).attach_inventory(split)
        assert len(spool.select(gauge_length=10.0)) == 0
        assert len(spool.select(gauge_length=20.0)) == 0

    def test_empty_spool(self, inventory):
        """Nothing to select from stays nothing."""
        spool = dc.spool([]).attach_inventory(inventory)
        assert len(spool.select(gauge_length=10.0)) == 0

    def test_selection_composes(self, two_patch_spool, inventory):
        """A selection is a spool, so it selects again."""
        spool = two_patch_spool.attach_inventory(inventory)
        out = spool.select(gauge_length=10.0).select(tag="second")
        assert len(out) == 1

    def test_enrichment_survives(self, two_patch_spool, inventory):
        """The derived spool is the same spool, inventory and all."""
        spool = two_patch_spool.attach_inventory(inventory).enrich()
        out = spool.select(gauge_length=10.0)
        assert out[0].attrs.gauge_length == 10.0

    def test_original_is_unchanged(self, two_patch_spool, inventory):
        """Selecting leaves the spool it came from alone."""
        spool = two_patch_spool.attach_inventory(inventory)
        spool.select(gauge_length=99.0)
        assert len(spool) == 2

    def test_channel_level_name_trims_channels(self, two_patch_spool, inventory):
        """
        A track name selects channels rather than whole patches.

        Telling one from an attr takes the names accessor, and this is
        what it was added for.
        """
        spool = two_patch_spool.attach_inventory(inventory)
        out = spool.select(coupling="trench")
        assert len(out) == 2
        assert set(out.get_contents()["distance_max"]) == {150.0}

    def test_a_channel_level_name_nothing_matches_keeps_nothing(
        self, two_patch_spool, inventory
    ):
        """A value no channel holds selects no channel, so no patch."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert len(spool.select(zone="east")) == 0

    def test_unknown_name_still_raises(self, two_patch_spool, inventory):
        """A name neither side knows is a misspelling, and says so."""
        spool = two_patch_spool.attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            spool.select(not_a_name=1)

    def test_samples_stays_coordinate_only(self, two_patch_spool, inventory):
        """samples=True is a coordinate form; an attr is an error there."""
        spool = two_patch_spool.attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError):
            spool.select(gauge_length=10.0, samples=True)

    def test_indexed_coordinate_is_not_intercepted(self, two_patch_spool, inventory):
        """
        `distance` names both an inventory coordinate and a real one.

        The patch's own axis is what a range on it means, exactly as
        without an inventory.
        """
        spool = two_patch_spool.attach_inventory(inventory)
        coord = spool[0].get_coord("distance")
        out = spool.select(distance=(coord.min(), coord.max()))
        assert len(out) == 2

    def test_unit_bearing_selector_against_a_unitless_index(
        self, two_patch_spool, inventory
    ):
        """
        With nothing on either side to convert to, this is the same error.

        The inventory's own units are fixed rather than recorded, so a
        spool no patch of which states the attr has nothing to read a
        quantity against — exactly as the index has not.
        """
        spool = two_patch_spool.attach_inventory(inventory)
        with pytest.raises(UnitError, match="unitless"):
            spool.select(gauge_length=10.0 * dc.get_quantity("m"))

    def test_unit_bearing_selector_uses_the_index_units(self, patch, inventory):
        """
        Where the index does record units, a quantity converts to them.

        Attaching an inventory must not turn a selector which works into
        an error; the rows the index alone would have answered are still
        its to answer.
        """
        meters = dc.get_quantity("m")
        spool = dc.spool(
            [
                patch.update_attrs(tag="stated", gauge_length=10.0 * meters),
                patch.update_attrs(tag="blank"),
            ]
        ).attach_inventory(inventory)
        for selector in (10.0 * meters, (5.0 * meters, 15.0 * meters), [10.0 * meters]):
            out = sorted(spool.select(gauge_length=selector).get_contents()["tag"])
            assert out == ["blank", "stated"], selector


class TestInventoryUnselect:
    """Unselect reaches whatever select reaches."""

    def test_removes_what_select_keeps(self, two_patch_spool, inventory):
        """The complement holds for an inventory-backed name too."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert len(spool.unselect(gauge_length=10.0)) == 0
        assert len(spool.unselect(gauge_length=99.0)) == 2

    def test_undescribed_patch_is_kept(self, patch, inventory):
        """
        Silently not matching means silently not being removed.

        Unselect is the complement of a filter, so a patch the inventory
        cannot speak for stays for the same reason it is never selected.
        """
        other = patch.update_attrs(tag="other", acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        out = spool.unselect(gauge_length=10.0)
        assert out.get_contents()["tag"].tolist() == ["other"]

    def test_stated_value_wins(self, patch, inventory):
        """Per-row precedence is the same on the way out."""
        stated = patch.update_attrs(tag="stated", gauge_length=20.0)
        spool = dc.spool([patch, stated]).attach_inventory(inventory)
        assert spool.unselect(gauge_length=20.0).get_contents()["tag"].tolist() == [
            "random"
        ]

    def test_channel_level_name_removes_channels(self, two_patch_spool, inventory):
        """
        A track name removes channels rather than whole patches.

        Which is the one thing `unselect` refuses a real coordinate for:
        a coordinate range would leave a hole in every patch, while a
        channel query chooses which channels a patch holds.
        """
        spool = two_patch_spool.attach_inventory(inventory)
        out = spool.unselect(coupling="trench")
        assert len(out) == 2
        assert set(out.get_contents()["distance_min"]) == {151.0}


class TestInventorySelectCoverage:
    """Selector shapes and inventory shapes the main tests do not reach."""

    def test_regex_selector(self, two_patch_spool, inventory):
        """A compiled pattern is a selector shape the index takes too."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert len(spool.select(**{"interrogator.model": re.compile("FI-")})) == 2
        assert len(spool.select(**{"interrogator.model": re.compile("^ZZ")})) == 0

    def test_half_open_ranges(self, two_patch_spool, inventory):
        """An open end bounds nothing, exactly as on an indexed attr."""
        spool = two_patch_spool.attach_inventory(inventory)
        assert len(spool.select(gauge_length=(5.0, None))) == 2
        assert len(spool.select(gauge_length=(None, 5.0))) == 0
        assert len(spool.select(gauge_length=(None, 15.0))) == 2

    def test_unrelated_branches_are_skipped(self, patch, inventory):
        """
        Only the key's own branch decides where its epochs are.

        An inventory describing several networks and arrays must not have
        another one's epoch boundaries chop up this one's patches.
        """
        network = inventory.networks[0]
        array = network.fiber_arrays[0]
        elsewhere = network.new(
            code="ZZ",
            fiber_arrays=(array.new(code="L999", start_time="2001-01-01"),),
        )
        sibling = array.new(code="L998", start_time="2001-01-02")
        crowded = inventory.new(
            networks=(
                elsewhere,
                network.new(fiber_arrays=(sibling, *network.fiber_arrays)),
            )
        )
        spool = dc.spool([patch]).attach_inventory(crowded)
        assert len(spool.select(gauge_length=10.0)) == 1


def _two_row_spool(inventory, **acquisition):
    """One row the index states and one the inventory answers, same value."""
    patch = inventory_patch_pair()[0]
    ((name, value),) = acquisition.items()
    described = _replace_acquisition(inventory, **acquisition)
    return dc.spool(
        [
            patch.update_attrs(tag="unstated", **{name: type(value)()}),
            patch.update_attrs(tag="stated", **{name: value}),
        ]
    ).attach_inventory(described)


class TestSelectorMeansOneThing:
    """
    A selector must mean the same thing on both sides of the split.

    Each of these puts the identical value in two rows -- one stated in
    the header, one answered by the inventory -- so any difference in the
    verdict is the two predicates disagreeing rather than the data.
    """

    def test_a_regex_searches(self, inventory):
        """The index applies its regex residual with search, not match."""
        spool = _two_row_spool(inventory, firmware_version="v1.2.3")
        out = spool.select(firmware_version=re.compile("1.2"))
        assert sorted(out.get_contents()["tag"]) == ["stated", "unstated"]

    def test_a_glob_is_sqlites(self, inventory):
        """
        A negated class is spelled the way the index reads it.

        `[!x]` is fnmatch's spelling and `[^x]` SQLite's; reaching for
        fnmatch made each spelling select the half the other did not.
        """
        spool = _two_row_spool(inventory, firmware_version="v1.2.3")
        assert len(spool.select(firmware_version="v[^x]*")) == 2
        assert len(spool.select(firmware_version="v[!x]*")) == 0
        assert len(spool.select(firmware_version="v[^1]*")) == 0

    def test_a_range_of_the_wrong_kind_matches_nothing(self, inventory):
        """
        The index reaches for the typed column a selector names.

        A numeric range against a string column matches nothing there, so
        it matches nothing here rather than comparing across types.
        """
        spool = _two_row_spool(inventory, firmware_version="v1.2.3")
        assert len(spool.select(firmware_version=(1.0, 2.0))) == 0

    def test_a_number_does_not_match_a_bool(self, patch, inventory):
        """`1` and `True` are stored as different kinds, so they differ."""
        described = _replace_acquisition(inventory, closed_fiber_loop=True)
        spool = dc.spool(
            [
                patch.update_attrs(tag="unstated"),
                patch.update_attrs(tag="stated", closed_fiber_loop=True),
            ]
        ).attach_inventory(described)
        assert len(spool.select(closed_fiber_loop=1)) == 0
        assert len(spool.select(closed_fiber_loop=True)) == 2


class TestNothingToResolveWith:
    """Rows which offer no identity, or no instant to resolve it at."""

    def test_spool_without_acquisition_keys(self, inventory):
        """A patch naming no entry is one the inventory cannot describe."""
        spool = dc.spool([dc.get_example_patch()]).attach_inventory(inventory)
        assert "acquisition_key" not in spool.get_contents().columns
        assert len(spool.select(gauge_length=10.0)) == 0

    def test_lag_times_are_not_instants(self, patch, inventory):
        """
        A relative time axis says nothing about which epoch applies.

        `Patch.enrich` refuses such a patch outright; reading its lags as
        instants since 1970 would pick an epoch from an offset.
        """
        array = inventory.networks[0].fiber_arrays[0]
        acquisition = array.acquisitions[0]
        split = np.datetime64("1970-01-01T00:00:10", "ns")
        described = inventory.replace(
            array,
            array.new(
                acquisitions=(
                    acquisition.new(end_time=split, gauge_length=10.0),
                    acquisition.new(start_time=split, gauge_length=20.0),
                )
            ),
        )
        lags = patch.get_coord("time").values - patch.get_coord("time").min()
        spool = dc.spool(
            [
                patch.update_coords(time=lags).update_attrs(tag="early"),
                patch.update_coords(time=lags + np.timedelta64(20, "s")).update_attrs(
                    tag="late"
                ),
            ]
        ).attach_inventory(described)
        assert len(spool.select(gauge_length=10.0)) == 0
        assert len(spool.select(gauge_length=20.0)) == 0


class TestSelectOnAView:
    """A spool whose membership is already fixed still selects correctly."""

    def test_windowed_spool_respects_stated_values(self, patch, inventory):
        """
        A window fixes which rows are present, not what they say.

        The index's verdict for a windowed spool has to be the predicate's,
        not the whole membership, or a row contradicting the selector
        would ride along on the strength of being present.
        """
        spool = dc.spool(
            [
                patch.update_attrs(tag="states", gauge_length=99.0),
                patch.update_attrs(tag="blank"),
            ]
        ).attach_inventory(inventory)
        window = spool[0:2]
        assert len(window) == 2
        assert window.select(gauge_length=10.0).get_contents()["tag"].tolist() == [
            "blank"
        ]


class TestCodexReviewFindings:
    """Defects found reviewing inventory-backed selection."""

    def test_a_patch_with_no_instant_is_not_resolved(self, patch, inventory):
        """
        A row whose times are NaT says nothing about which epoch applies.

        Resolving at NaT holds every epoch effective, which is the whole
        inventory answering rather than the one entry describing the row.
        """
        stamps = np.full(len(patch.get_coord("time")), np.datetime64("NaT", "ns"))
        spool = dc.spool(
            [
                patch.update_attrs(tag="good"),
                patch.update_coords(time=stamps).update_attrs(tag="undated"),
            ]
        ).attach_inventory(inventory)
        out = spool.select(gauge_length=10.0)
        assert out.get_contents()["tag"].tolist() == ["good"]

    def test_a_reversed_glob_range_means_one_thing(self, inventory):
        """
        SQLite tests a range's low endpoint before testing the range.

        So `[z-a]` matches `z` there, and the in-memory twin has to agree
        rather than reading the reversed range as matching nothing.
        """
        spool = _two_row_spool(inventory, firmware_version="z")
        assert len(spool.select(firmware_version="[z-a]")) == 2
        assert len(spool.select(firmware_version="[^z-a]")) == 0

    def test_a_range_with_no_bounds_raises(self, inventory):
        """
        An unusable range is the error it is on the index side.

        Applying no bounds instead would keep every row the inventory
        describes, and the same selector would raise or not depending on
        whether some unrelated patch happened to state the name.
        """
        spool = _two_row_spool(inventory, firmware_version="v1")
        for empty in ((None, None), (..., ...)):
            with pytest.raises(InvalidSpoolQueryError, match="no usable bounds"):
                spool.select(**{"firmware_version": empty})

    def test_a_backwards_range_raises(self, patch, inventory):
        """And so is one whose bounds are the wrong way round."""
        spool = dc.spool([patch]).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="lo > hi"):
            spool.select(gauge_length=(20.0, 5.0))

    def test_a_malformed_selector_raises_where_the_index_answers(
        self, patch, inventory
    ):
        """
        A bad selector is bad whether or not an inventory is attached.

        Every patch here states the name, so the inventory is never
        consulted and the index's own complaint is the only one there is;
        swallowing it as "the index selects nothing" would turn the error
        into an empty spool.

        The count is what forces it: an index-only select composes the
        predicate lazily and complains when the rows are realized, while
        an inventory-backed one realizes them at the call.
        """
        stated = patch.update_attrs(gauge_length=10.0)
        for spool in (
            dc.spool([stated]),
            dc.spool([stated]).attach_inventory(inventory),
        ):
            with pytest.raises(InvalidSpoolQueryError, match="lo > hi"):
                len(spool.select(gauge_length=(20.0, 5.0)))

    def test_units_are_spelled_the_way_a_patch_spells_them(self, patch, inventory):
        """
        The inventory models units as a quantity; a patch carries a string.

        The same fact stored two ways would never match one selector, so
        the inventory's answer is rendered the way a header states it.
        """
        described = _replace_acquisition(inventory, data_units="m/s")
        spool = dc.spool(
            [
                patch.update_attrs(tag="stated", data_units="m/s"),
                patch.update_attrs(tag="blank", data_units=""),
            ]
        ).attach_inventory(described)
        out = sorted(spool.select(data_units="m / s").get_contents()["tag"])
        assert out == ["blank", "stated"]

    def test_a_wildcard_crosses_a_newline(self, inventory):
        """SQLite's wildcards match any byte, newlines included."""
        spool = _two_row_spool(inventory, firmware_version="a\nb")
        assert len(spool.select(firmware_version="a*")) == 2


class TestRelationAndIdListDisagree:
    """The relation is realized by a route of its own."""

    def test_rows_it_omits_resolve_to_nothing(self, patch, inventory, monkeypatch):
        """
        A row the relation leaves out is one nothing was resolved for.

        The two are aligned by id rather than by position, so a relation
        presenting fewer rows than the id list cannot silently shift
        every context onto the wrong patch.
        """
        spool = dc.spool(
            [patch.update_attrs(tag="first"), patch.update_attrs(tag="second")]
        ).attach_inventory(inventory)
        full = spool._df
        monkeypatch.setattr(
            type(spool), "_df", property(lambda self, df=full.iloc[1:]: df)
        )
        out = spool.select(gauge_length=10.0)
        assert out.get_contents()["tag"].tolist() == ["second"]


class TestConnectorReviewFindings:
    """Defects the PR's automated reviewers found."""

    def test_a_group_may_share_an_attrs_name(self, patch, inventory):
        """
        An annotation group is free to be named after an acquisition field.

        Bare names resolve to attrs first, so selecting on the field has
        to keep working; only a caller who asked for `_coords` means the
        group, which trims channels rather than choosing patches.
        """
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        clash = inventory.replace(
            path,
            path.new(
                annotations=(
                    *path.annotations,
                    OpticalPathAnnotation(
                        start_distance=0.0,
                        end_distance=100.0,
                        group="gauge_length",
                        value="odd",
                    ),
                )
            ),
        )
        names = clash.get_names()
        assert "gauge_length" in names.attrs and "gauge_length" in names.coords
        spool = dc.spool([patch]).attach_inventory(clash)
        assert len(spool.select(gauge_length=10.0)) == 1
        assert len(spool.select(_attrs={"gauge_length": 10.0})) == 1
        # The group covers the first metre of the path, which is the
        # patch's first channel and nothing else.
        for form in ({"gauge_length": "odd"}, "gauge_length", ["gauge_length"]):
            # The mapping form and both tag forms all name the coordinate.
            kwargs = {} if isinstance(form, Mapping) else {"gauge_length": "odd"}
            out = spool.select(_coords=form, **kwargs)
            assert out.get_contents()["distance_max"].tolist() == [0.0]

    def test_a_field_scalar_on_another_path_is_kept(self):
        """
        One path stating several values does not silence the others.

        A field is only unusable where nothing states it as one value; a
        path which records a scalar can still give it to a channel.
        """

        def path(loss, measurement):
            return OpticalPath(
                location_code="",
                optical_components=(
                    FiberSegment(
                        name="c",
                        optical_length=100.0,
                        loss_db=loss,
                        loss_measurement=measurement,
                    ),
                ),
            )

        def build(*paths):
            array = FiberArray(code="R2D1", optical_paths=paths)
            return Inventory(
                networks=(Network(code="DAS", fiber_arrays=(array,)),),
                resources={"m1": OpticalMeasurement(resource_id="m1")},
            )

        many = path((0.2, 0.25), ("m1", "m1"))
        one = path(0.3, "m1")
        assert "optical_components.loss_db" not in build(many).get_names().coords
        assert "optical_components.loss_db" in build(many, one).get_names().coords
        assert "optical_components.loss_db" in build(one).get_names().coords


def _split_epochs(inventory, when, *, acquisitions=False, second=None):
    """
    Return the inventory with its one path (or acquisition) split at `when`.

    The two halves meet exactly at `when`, which epochs being half-open
    makes the first instant of the second half rather than the last of
    the first.
    """
    array = inventory.networks[0].fiber_arrays[0]
    old = array.acquisitions[0] if acquisitions else array.optical_paths[0]
    changed = {} if second is None else second
    halves = (old.new(end_time=when), old.new(start_time=when, **changed))
    field = "acquisitions" if acquisitions else "optical_paths"
    return inventory.replace(array, array.new(**{field: halves})).check()


def _pieces(spool):
    """The (min, max) time envelope of each patch a spool presents."""
    contents = spool.get_contents()
    return list(zip(contents["time_min"], contents["time_max"], strict=True))


@pytest.fixture(scope="module")
def off_grid_boundary(patch):
    """An epoch boundary a third of a sample past one of the patch's samples."""
    coord = patch.get_coord("time")
    return coord.min() + (coord.max() - coord.min()) / 2 + coord.step / 3


@pytest.fixture(scope="module")
def path_epochs(inventory, off_grid_boundary):
    """The example inventory, its optical path split into two epochs."""
    return _split_epochs(inventory, off_grid_boundary, second={"name": "moved"})


class TestConformBoundaryPolicy:
    """
    Where a subdivided patch is cut, and which piece each sample joins.

    Including the rows which offer no usable boundary at all — no
    instants to place one against, or no identity to look one up with.
    """

    def test_split_is_lossless(self, patch, path_epochs):
        """
        The pieces hold every sample the patch held, in order, once.

        Subdivision reconciles metadata; it must not restructure the
        data, so a sample dropped or duplicated at the seam would be the
        one failure the whole operation cannot afford.
        """
        spool = dc.spool(patch).attach_inventory(path_epochs).conform_to_inventory()
        assert len(spool) == 2
        first, second = spool[0], spool[1]
        times = np.concatenate(
            [first.get_coord("time").values, second.get_coord("time").values]
        )
        assert np.array_equal(times, patch.get_coord("time").values)
        data = np.concatenate([first.data, second.data], axis=1)
        assert np.array_equal(data, patch.data)

    def test_off_grid_boundary_keeps_its_straddling_sample(
        self, patch, path_epochs, off_grid_boundary
    ):
        """
        The sample between `boundary - step` and `boundary` is not lost.

        An epoch boundary owes the sample grid nothing, so the naive
        split of `[t_min, boundary - step]` and `[boundary, t_max]`
        excludes any sample falling in between them from both pieces.
        Cutting on the grid instead is what makes the split exact.
        """
        coord = patch.get_coord("time")
        naive = off_grid_boundary - coord.step
        (_, first_end), (second_start, _) = _pieces(
            dc.spool(patch).attach_inventory(path_epochs).conform_to_inventory()
        )
        # the sample the naive convention would have dropped
        assert naive < first_end < off_grid_boundary
        assert second_start == first_end + coord.step

    def test_boundary_sample_opens_the_second_piece(self, patch, inventory):
        """
        A boundary landing exactly on a sample gives it to the new epoch.

        Epochs are half-open (`is_effective_at` admits `start` and
        refuses `end`), so the split has to agree with them about which
        epoch owns the instant they meet at.
        """
        coord = patch.get_coord("time")
        on_grid = coord.min() + coord.step * 10
        split = _split_epochs(inventory, on_grid, second={"name": "moved"})
        (_, first_end), (second_start, _) = _pieces(
            dc.spool(patch).attach_inventory(split).conform_to_inventory()
        )
        assert second_start == on_grid
        assert first_end == on_grid - coord.step

    def test_unstated_time_step_raises_naming_the_file(self, patch, inventory):
        """
        A patch which must be cut but states no step says so, loudly.

        The cut is found on the patch's own sample grid, which the step
        is the only description of. Backing off to the raw boundary
        would silently drop the sample beside it, and for an operation
        asked to reconcile metadata that is the worse answer.
        """
        coord = patch.get_coord("time")
        times = np.sort(np.concatenate([coord.values[:5], coord.values[10:15]]))
        uneven = patch.select(time=(0, 10), samples=True).update_coords(time=times)
        assert pd.isnull(dc.spool(uneven).get_contents()["time_step"]).all()
        split = _split_epochs(inventory, times[3], second={"name": "moved"})
        spool = dc.spool(uneven)
        named = re.escape(spool.get_contents()["source_path"].iloc[0])
        with pytest.raises(PatchError, match=f"no time step.*{named}"):
            spool.attach_inventory(split).conform_to_inventory()

    def test_a_row_with_no_instants_is_undescribed(self, patch, path_epochs):
        """
        A patch whose instants are unknown resolves to every epoch at once.

        `is_effective_at(NaT)` holds every epoch effective, so such a row
        would be answered by the whole inventory rather than by one
        entry. That is ambiguity rather than resolution, so
        `on_unresolved` governs it — the same call `Patch.enrich` makes.
        """
        undated = patch.update_coords(
            time=np.full(patch.shape[1], np.datetime64("NaT"))
        ).update_attrs(tag="undated")
        # alongside a dated patch, so the row really does carry NaT
        # instants rather than a time column of another kind entirely
        spool = dc.spool([patch, undated]).attach_inventory(path_epochs)
        assert pd.isnull(spool.get_contents()["time_min"]).any()
        out = spool.conform_to_inventory(on_unresolved="ignore")
        assert set(out.get_contents()["tag"]) == {"random"}
        with pytest.raises(UnresolvedPatchError, match="does not describe"):
            spool.conform_to_inventory()

    def test_an_empty_key_is_undescribed(self, patch, path_epochs):
        """
        The empty string is how a patch spells "no identity" at all.

        It names no entry, which is not the same as naming a missing
        one, and neither is something to resolve.
        """
        bare = patch.update_attrs(acquisition_key="", tag="bare")
        spool = dc.spool([patch, bare]).attach_inventory(path_epochs)
        out = spool.conform_to_inventory(on_unresolved="ignore")
        assert set(out.get_contents()["tag"]) == {"random"}

    def test_a_relative_time_axis_is_undescribed(self, patch, path_epochs):
        """
        Lag times are not instants, so they pick no epoch either.

        Reading them as instants since 1970 would choose an epoch from
        an offset; `Patch.enrich` refuses such a patch outright, and
        conform has a policy for it instead of a guess.
        """
        coord = patch.get_coord("time")
        lags = patch.update_coords(time=coord.values - coord.min())
        spool = dc.spool(lags).attach_inventory(path_epochs)
        assert len(spool.conform_to_inventory(on_unresolved="ignore")) == 0
        with pytest.raises(UnresolvedPatchError, match="does not describe"):
            spool.conform_to_inventory()

    def test_a_merged_patch_carries_one_key(self, patch, path_epochs):
        """
        Conform needs one acquisition_key per row, and merging leaves it one.

        `acquisition_key` is a default group attribute, so patches from
        two acquisitions partition apart and no output is ever built
        from both. Grouping the partition away instead makes the key a
        conflicted column, which merging refuses outright.
        """
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool([patch, other])
        assert "acquisition_key" in dc.get_config().groupby_attrs
        merged = spool.chunk(time=None)
        assert sorted(merged.get_contents()["acquisition_key"]) == [
            "DAS.R2D1..OTHER",
            "DAS.R2D1..RAW",
        ]
        conformed = merged.attach_inventory(path_epochs).conform_to_inventory(
            on_unresolved="ignore"
        )
        assert set(conformed.get_contents()["acquisition_key"]) == {"DAS.R2D1..RAW"}
        with pytest.raises(CoordMergeError, match="acquisition_key"):
            spool.chunk(time=None, group="tag")

    def test_a_merge_which_drops_the_key_is_undescribed(self, patch, path_epochs):
        """
        A row whose merge discarded the key resolves to nothing, loudly.

        `conflict="drop"` is the one way an output can carry no identity
        at all, and conform's own default reports it rather than
        quietly returning an empty spool.
        """
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        merged = dc.spool([patch, other]).chunk(time=None, group="tag", conflict="drop")
        assert "acquisition_key" not in merged.get_contents().columns
        with pytest.raises(UnresolvedPatchError, match="does not describe"):
            merged.attach_inventory(path_epochs).conform_to_inventory()


class TestConformMembership:
    """Which patches a conformed spool holds, and how it says so."""

    def test_a_described_spool_is_unchanged(self, patch, inventory):
        """An inventory which already describes everything removes nothing."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert len(spool.conform_to_inventory()) == len(spool) == 1

    def test_undescribed_patches_drop(self, patch, inventory):
        """A patch naming an entry the inventory lacks is not conformable."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER", tag="other")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        out = spool.conform_to_inventory(on_unresolved="ignore")
        assert out.get_contents()["tag"].tolist() == ["random"]

    def test_warn_drops_and_names_them(self, patch, inventory):
        """The middle policy removes the rows but says which they were."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER", tag="other")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        with pytest.warns(UserWarning, match="does not describe") as record:
            out = spool.conform_to_inventory(on_unresolved="warn")
        assert len(out) == 1
        # the point of warning at all is saying which patch was dropped
        dropped = spool.get_contents()["source_path"].iloc[1]
        assert dropped in str(record[0].message)

    def test_raise_is_the_default(self, patch, inventory):
        """Conforming to an inventory which covers less says so by default."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        with pytest.raises(UnresolvedPatchError, match="does not describe"):
            spool.conform_to_inventory()

    def test_len_matches_the_contents(self, patch, inventory, path_epochs):
        """
        Conforming is metadata work, so the count it reports is final.

        Both the subdivided and the merely-filtered route have to agree
        with what iteration will actually yield.
        """
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        subdivided = dc.spool([patch, other]).attach_inventory(path_epochs)
        # the filtered route keeps a row rather than emptying the spool,
        # so an agreement of zeroes cannot stand in for the real thing
        filtered = dc.spool([patch, other]).attach_inventory(inventory)
        for out, expected in ((subdivided, 2), (filtered, 1)):
            out = out.conform_to_inventory(on_unresolved="ignore")
            assert len(out) == len(out.get_contents()) == len(list(out)) == expected

    def test_needs_an_inventory(self, patch):
        """With none attached and none passed there is nothing to conform to."""
        with pytest.raises(ParameterError, match="needs an inventory"):
            dc.spool(patch).conform_to_inventory()

    def test_an_empty_spool_conforms_to_nothing(self, inventory):
        """No rows to resolve is not a reason to fail."""
        assert len(dc.spool([]).attach_inventory(inventory).conform_to_inventory()) == 0

    def test_policy_is_checked(self, patch, inventory):
        """A misspelled policy is caught before any row is resolved."""
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(ParameterError, match="on_unresolved must be"):
            spool.conform_to_inventory(on_unresolved="drop")

    def test_conform_keeps_the_attached_inventory(self, patch, inventory):
        """Conforming leaves the spool carrying what it conformed to."""
        out = dc.spool(patch).attach_inventory(inventory).conform_to_inventory()
        assert out._inventory is inventory
        assert out.enrich()[0].attrs.gauge_length == 10.0

    def test_patches_without_keys_are_undescribed(self, inventory):
        """A spool whose rows name no entry has nothing to resolve with."""
        spool = dc.spool([dc.get_example_patch()]).attach_inventory(inventory)
        assert "acquisition_key" not in spool.get_contents().columns
        assert len(spool.conform_to_inventory(on_unresolved="ignore")) == 0


class TestConformSubdivision:
    """Patches the inventory describes twice, split into pieces it describes once."""

    def test_the_spool_grows(self, patch, path_epochs):
        """One patch spanning two path epochs becomes two patches."""
        spool = dc.spool(patch).attach_inventory(path_epochs)
        assert len(spool) == 1
        assert len(spool.conform_to_inventory()) == 2

    def test_each_piece_enriches_from_its_own_epoch(self, patch, inventory):
        """
        Subdividing is what makes enrichment possible at all here.

        The undivided patch is described twice, which `Patch.enrich`
        refuses; each piece is described once and takes the geometry of
        the epoch it falls in.
        """
        coord = patch.get_coord("time")
        when = coord.min() + (coord.max() - coord.min()) / 2
        moved = inventory.networks[0].fiber_arrays[0].optical_paths[0].annotations[0]
        split = _split_epochs(
            inventory,
            when,
            second={
                "annotations": (
                    moved.new(value="moved", start_distance=100.0, end_distance=400.0),
                )
            },
        )
        spool = dc.spool(patch).attach_inventory(split)
        with pytest.raises(PatchError, match="spans a change"):
            patch.enrich(split)
        first, second = spool.conform_to_inventory().enrich(coords=True)
        assert set(np.unique(first.get_coord("zone").values)) == {"north", "south"}
        assert set(np.unique(second.get_coord("zone").values)) == {"moved"}

    def test_several_epochs_split_a_patch_several_ways(self, patch, inventory):
        """A row is cut once per boundary it crosses, not once at all."""
        coord = patch.get_coord("time")
        array = inventory.networks[0].fiber_arrays[0]
        old = array.optical_paths[0]
        edges = [coord.min() + coord.step * x for x in (500, 1000)]
        paths = (
            old.new(end_time=edges[0]),
            old.new(start_time=edges[0], end_time=edges[1], name="middle"),
            old.new(start_time=edges[1], name="last"),
        )
        split = inventory.replace(array, array.new(optical_paths=paths)).check()
        out = dc.spool(patch).attach_inventory(split).conform_to_inventory()
        assert len(out) == 3
        starts = [x[0] for x in _pieces(out)]
        assert starts[1] == edges[0] and starts[2] == edges[1]

    def test_an_epoch_shorter_than_one_sample_makes_one_cut(self, patch, inventory):
        """
        Two boundaries inside one sample interval name one split.

        The epoch between them covers no sample of this patch, so a
        piece for it would hold nothing; the samples still divide into
        the epoch before and the epoch after.
        """
        coord = patch.get_coord("time")
        array = inventory.networks[0].fiber_arrays[0]
        old = array.optical_paths[0]
        edges = [coord.min() + coord.step * 500 + x for x in (-coord.step / 3, 0)]
        paths = (
            old.new(end_time=edges[0]),
            old.new(start_time=edges[0], end_time=edges[1], name="blink"),
            old.new(start_time=edges[1], name="after"),
        )
        split = inventory.replace(array, array.new(optical_paths=paths)).check()
        out = dc.spool(patch).attach_inventory(split).conform_to_inventory()
        assert len(out) == 2
        assert _pieces(out)[1][0] == edges[1]
        assert out[0].shape[1] + out[1].shape[1] == patch.shape[1]

    def test_a_boundary_outside_the_patch_cuts_nothing(self, patch, inventory):
        """Only the epochs a patch actually reaches into can divide it."""
        coord = patch.get_coord("time")
        split = _split_epochs(
            inventory, coord.max() + coord.step * 10, second={"name": "later"}
        )
        assert len(dc.spool(patch).attach_inventory(split).conform_to_inventory()) == 1

    def test_a_boundary_which_changes_nothing_cuts_nothing(self, patch, inventory):
        """
        An epoch bound the answers do not change across is not a boundary.

        Every stated time on the key's branch is a candidate, so a fiber
        array re-registered mid-patch — renamed, say — puts a bound
        inside the row while leaving its acquisition and optical path
        saying exactly what they said before. Splitting there would grow
        the spool for nothing, and refusing the patch would be worse.
        """
        coord = patch.get_coord("time")
        when = coord.min() + coord.step * 500
        array = inventory.networks[0].fiber_arrays[0]
        network = inventory.networks[0]
        renamed = inventory.replace(
            network,
            network.new(
                fiber_arrays=(
                    array.new(end_time=when),
                    array.new(start_time=when, description="renamed"),
                )
            ),
        ).check()
        # the bound really does fall inside the row
        assert coord.min() < when <= coord.max()
        assert (
            len(dc.spool(patch).attach_inventory(renamed).conform_to_inventory()) == 1
        )
        # and enrich agrees, which is the point of comparing by value
        assert patch.enrich(renamed).attrs.gauge_length == 10.0

    def test_selection_reads_across_a_bound_which_changes_nothing(
        self, patch, inventory
    ):
        """
        Selecting resolves a row crossing a bound its answers survive.

        `Patch.enrich` has always allowed this — it refuses only a real
        change of acquisition or path — so a spool refusing it would
        have made the two disagree about the same patch.
        """
        coord = patch.get_coord("time")
        when = coord.min() + coord.step * 500
        network = inventory.networks[0]
        array = network.fiber_arrays[0]
        renamed = inventory.replace(
            network,
            network.new(
                fiber_arrays=(
                    array.new(end_time=when),
                    array.new(start_time=when, description="renamed"),
                )
            ),
        ).check()
        assert coord.min() < when <= coord.max()
        spool = dc.spool(patch).attach_inventory(renamed)
        assert len(spool.select(gauge_length=10.0)) == 1
        assert patch.enrich(renamed).attrs.gauge_length == 10.0

    def test_an_acquisition_change_raises(self, patch, inventory, off_grid_boundary):
        """
        No subdivision makes a patch recorded two ways into one patch.

        The inventory describes it twice, so `on_unresolved` — which is
        about patches it does not describe — has no say either.
        """
        split = _split_epochs(
            inventory,
            off_grid_boundary,
            acquisitions=True,
            second={"gauge_length": 20.0},
        )
        spool = dc.spool(patch).attach_inventory(split)
        named = re.escape(spool.get_contents()["source_path"].iloc[0])
        for policy in ("raise", "warn", "ignore"):
            with pytest.raises(
                PatchError, match=f"change of acquisition.*{named} \\(at .*"
            ):
                spool.conform_to_inventory(on_unresolved=policy)


class TestConformComposition:
    """How conforming sits with the rest of the spool API."""

    def test_conform_is_idempotent(self, patch, path_epochs):
        """The pieces of a conformed spool each sit inside one epoch."""
        once = dc.spool(patch).attach_inventory(path_epochs).conform_to_inventory()
        twice = once.conform_to_inventory()
        assert len(twice) == len(once) == 2
        assert _pieces(twice) == _pieces(once)

    def test_conform_nests_over_a_chunked_spool(self, patch, inventory):
        """
        Conforming a chunked spool splits the chunks, not their sources.

        The pieces are cut out of the assembled outputs, so a chunk
        which straddles a boundary becomes two and the rest stand.
        """
        coord = patch.get_coord("time")
        when = coord.min() + np.timedelta64(6, "s")
        split = _split_epochs(inventory, when, second={"name": "moved"})
        chunked = dc.spool(patch).chunk(time=4)
        assert len(chunked) == 2
        out = chunked.attach_inventory(split).conform_to_inventory()
        assert len(out) == 3
        assert [x[0] for x in _pieces(out)][2] == when
        assert out[2].get_coord("time").min() == when

    def test_selecting_after_conforming(self, patch, path_epochs, off_grid_boundary):
        """The pieces are ordinary rows, so ordinary selection reaches them."""
        out = dc.spool(patch).attach_inventory(path_epochs).conform_to_inventory()
        late = out.select(time=(off_grid_boundary, None))
        assert len(late) == 1
        assert late[0].get_coord("time").min() >= off_grid_boundary

    def test_the_inventory_carries_over(self, patch, path_epochs):
        """A conformed spool still knows what it was conformed to."""
        out = dc.spool(patch).attach_inventory(path_epochs).conform_to_inventory()
        assert out._inventory is path_epochs
        assert len(out.select(gauge_length=10.0)) == 2

    def test_conform_over_a_directory_spool(self, tmp_path_factory, patch, inventory):
        """
        The pieces of a file-backed patch are read back exactly.

        In-memory patches ignore the trim hints a plan passes down, so
        only a real file exercises the read path the pieces take.
        """
        coord = patch.get_coord("time")
        path = tmp_path_factory.mktemp("conform_directory")
        dc.write(patch, path / "patch.h5", "dasdae")
        split = _split_epochs(
            inventory, coord.min() + coord.step * 500, second={"name": "moved"}
        )
        out = dc.spool(path).update().attach_inventory(split).conform_to_inventory()
        assert len(out) == 2
        first, second = out[0], out[1]
        assert first.shape[1] == 500
        assert second.shape[1] == patch.shape[1] - 500
        data = np.concatenate([first.data, second.data], axis=1)
        assert np.array_equal(data, patch.data)


class TestConformPartialCoverage:
    """Rows the inventory describes for part of their span."""

    def test_a_row_described_only_at_its_start_is_undescribed(
        self, patch, inventory, off_grid_boundary
    ):
        """
        A patch is judged over its whole span, not where it begins.

        The acquisition lapses partway through with nothing after it, so
        the tail of the patch has no entry. Keeping the described head
        would answer a question the caller did not ask — they asked
        about the patch — so the row is undescribed and `on_unresolved`
        governs it.
        """
        array = inventory.networks[0].fiber_arrays[0]
        lapsed = inventory.replace(
            array,
            array.new(
                acquisitions=(array.acquisitions[0].new(end_time=off_grid_boundary),)
            ),
        ).check()
        spool = dc.spool(patch).attach_inventory(lapsed)
        assert len(spool.conform_to_inventory(on_unresolved="ignore")) == 0
        with pytest.raises(UnresolvedPatchError, match="does not describe"):
            spool.conform_to_inventory()

    def test_a_path_which_lapses_divides_the_patch(
        self, patch, inventory, off_grid_boundary
    ):
        """
        An optical path ending mid-patch is a change of path like any other.

        An acquisition with no optical path is a described acquisition —
        it simply projects nothing along the fiber — so the piece after
        the lapse is described once, as the piece before it is. This is
        why a lapsed *path* subdivides where a lapsed *acquisition*
        leaves the row undescribed: the acquisition is what makes a
        patch describable at all.
        """
        array = inventory.networks[0].fiber_arrays[0]
        lapsed = inventory.replace(
            array,
            array.new(
                optical_paths=(array.optical_paths[0].new(end_time=off_grid_boundary),)
            ),
        ).check()
        out = dc.spool(patch).attach_inventory(lapsed).conform_to_inventory()
        assert len(out) == 2
        first, second = out.enrich(coords=True)
        assert "zone" in first.coords.coord_map
        assert "zone" not in second.coords.coord_map

    def test_many_undescribed_patches_are_summarized(self, patch, inventory):
        """Naming every file of a mismatched archive helps nobody."""
        spool = dc.spool(
            [
                patch.update_attrs(acquisition_key="DAS.R2D1..OTHER", tag=f"t{index}")
                for index in range(7)
            ]
        ).attach_inventory(inventory)
        with pytest.raises(UnresolvedPatchError, match=r"\(and 2 more\)"):
            spool.conform_to_inventory()

    def test_a_row_ending_before_it_starts_resolves_to_nothing(self, patch, inventory):
        """
        An envelope running backwards spans no epoch, so it answers none.

        Envelopes are value-ordered by construction, so this is a
        malformed row rather than a reachable state; resolution says so
        rather than failing on the empty span it computes.
        """
        coord = patch.get_coord("time")
        (epochs,) = resolve_row_epochs(
            inventory, [patch.attrs.acquisition_key], [coord.max()], [coord.min()]
        )
        assert not epochs.described and not epochs.cuts


class TestConformAndEnrichment:
    """How conforming interacts with enrichment already set up."""

    def test_conforming_keeps_enrichment(self, patch, path_epochs):
        """The spool's own inventory is not a new one, so nothing is swapped."""
        spool = dc.spool(patch).attach_inventory(path_epochs).enrich(coords=False)
        out = spool.conform_to_inventory()
        assert out._enrich_kwargs is not None
        assert out[0].attrs.gauge_length == 10.0

    def test_reattaching_clears_enrichment(self, patch, path_epochs):
        """
        Attaching always clears enrichment, and conform sees that.

        Swapping the source underneath a configured enrichment would
        rewrite every patch's metadata, so the new one has to be asked
        for. `attach_inventory` is the only way a spool gets an
        inventory, so it is the only place that rule is needed.
        """
        spool = dc.spool(patch).attach_inventory(path_epochs).enrich(coords=False)
        out = spool.attach_inventory(path_epochs).conform_to_inventory()
        assert out._enrich_kwargs is None
        assert "gauge_length" not in dict(out[0].attrs)


@pytest.fixture(scope="module")
def two_zones(inventory):
    """The example inventory, with a group covering two separate stretches."""
    path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
    return inventory.replace(
        path,
        path.new(
            annotations=(
                *path.annotations,
                OpticalPathAnnotation(
                    start_distance=110.0, end_distance=150.0, group="hole", value="a"
                ),
                OpticalPathAnnotation(
                    start_distance=300.0, end_distance=340.0, group="hole", value="a"
                ),
            )
        ),
    )


def _channels(spool):
    """Every channel a spool holds, gathered from its patches."""
    return np.concatenate([x.get_array("distance") for x in spool])


class TestChannelSelect:
    """Selecting on the coordinates an inventory defines along the fiber."""

    def test_trims_to_the_matching_channels(self, patch, inventory):
        """
        A track name keeps the channels it covers and no others.

        The example path is coupled from 100 to 250 m along the fiber and
        the patch's channels start at 100, so the trench runs from the
        patch's channel 0 to its channel 150.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.select(coupling="trench")
        assert len(out) == 1
        assert out[0].get_coord("distance").min() == 0
        assert out[0].get_coord("distance").max() == 150

    def test_the_data_is_the_channels_it_names(self, patch, inventory):
        """
        The rows kept are the rows the coordinate names.

        Trimming the envelope without trimming the same rows out of the
        array is the one failure which would not show up in the contents.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.select(coupling="trench")[0]
        whole = patch.get_array("distance")
        rows = np.searchsorted(whole, out.get_array("distance"))
        assert np.array_equal(out.data, patch.data[rows])

    def test_a_disjoint_match_subdivides(self, patch, two_zones):
        """
        A group covering two stretches gives two patches, not one span.

        This is why selection builds a plan: `len` grows, and the hole
        between the two runs belongs to neither.
        """
        spool = dc.spool(patch).attach_inventory(two_zones)
        out = spool.select(hole="a")
        assert len(out) == 2
        contents = out.get_contents()
        assert contents["distance_min"].tolist() == [10.0, 200.0]
        assert contents["distance_max"].tolist() == [50.0, 240.0]

    def test_a_disjoint_match_keeps_each_piece_whole(self, patch, two_zones):
        """Each piece holds exactly the rows its own envelope names."""
        spool = dc.spool(patch).attach_inventory(two_zones)
        whole = patch.get_array("distance")
        pieces = list(spool.select(hole="a"))
        assert len(pieces) == 2
        for piece in pieces:
            rows = np.searchsorted(whole, piece.get_array("distance"))
            assert np.array_equal(piece.data, patch.data[rows])

    def test_selection_agrees_with_enrichment(self, patch, two_zones):
        """
        Every channel kept really does hold the value asked for.

        The two run on one projection, so this is the property which
        makes that worth doing rather than a coincidence to maintain.
        """
        spool = dc.spool(patch).attach_inventory(two_zones)
        pieces = list(spool.select(hole="a").enrich())
        assert len(pieces) == 2
        for piece in pieces:
            assert len(piece.get_array("hole"))
            assert set(piece.get_array("hole")) == {"a"}

    def test_a_value_nothing_holds_keeps_nothing(self, patch, inventory):
        """
        A patch with no matching channel is no more selected than one
        which lacks the attr entirely.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        assert len(spool.select(coupling="cement")) == 0

    def test_an_undescribed_patch_is_silently_not_selected(self, patch, inventory):
        """The one agreed exception to loud-by-default."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(spool.select(coupling="trench")) == 1

    def test_a_straddling_patch_is_not_selected(self, patch, inventory):
        """
        Described twice is not one answer, so it is not selected.

        `conform_to_inventory` is what turns such a row into rows which
        each resolve; select is a filter and says nothing about it.
        """
        coord = patch.get_coord("time")
        middle = coord.min() + (coord.max() - coord.min()) / 2
        assert coord.min() < middle <= coord.max()
        split = _split_epochs(inventory, middle, second={"name": "moved"})
        spool = dc.spool(patch).attach_inventory(split)
        assert len(spool.select(coupling="trench")) == 0

    def test_conforming_first_makes_it_selectable(self, patch, inventory):
        """The pieces each resolve, so each is judged on its own."""
        coord = patch.get_coord("time")
        middle = coord.min() + (coord.max() - coord.min()) / 2
        split = _split_epochs(inventory, middle, second={"name": "moved"})
        spool = dc.spool(patch).attach_inventory(split)
        out = spool.conform_to_inventory().select(coupling="trench")
        assert len(out) == 2
        assert set(out.get_contents()["distance_max"]) == {150.0}

    def test_composes_with_a_time_split(self, patch, inventory):
        """Both subdivisions hold, and the pieces still hold their own data."""
        coord = patch.get_coord("time")
        middle = coord.min() + (coord.max() - coord.min()) / 2
        split = _split_epochs(inventory, middle, second={"name": "moved"})
        spool = dc.spool(patch).attach_inventory(split)
        out = spool.conform_to_inventory().select(coupling="trench")
        assert len(out) == 2
        times, dists = patch.get_array("time"), patch.get_array("distance")
        for piece in out:
            rows = np.searchsorted(dists, piece.get_array("distance"))
            cols = np.searchsorted(times, piece.get_array("time"))
            assert np.array_equal(piece.data, patch.data[np.ix_(rows, cols)])

    def test_two_selections_are_both_applied(self, patch, two_zones):
        """
        Selecting twice narrows; it does not re-plan from the source.

        Chunking the same dimension twice replaces the first plan, which
        is right for chunk and would silently undo a selection here.
        """
        spool = dc.spool(patch).attach_inventory(two_zones)
        out = spool.select(hole="a").select(zone="north")
        assert len(out) == 1
        assert out.get_contents()["distance_max"].tolist() == [50.0]

    def test_kwargs_are_and(self, patch, two_zones):
        """Two names in one call are judged together, as the index does."""
        spool = dc.spool(patch).attach_inventory(two_zones)
        both = spool.select(hole="a", zone="north")
        chained = spool.select(hole="a").select(zone="north")
        assert both.get_contents()["distance_max"].tolist() == (
            chained.get_contents()["distance_max"].tolist()
        )

    def test_a_whole_match_changes_nothing(self, patch, inventory):
        """Keeping every channel needs no plan, so the spool is unchanged."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.select(geometry="trench")
        # The same resolver, not merely an equal spool: building a plan
        # whose single piece covered its row would load the same patch
        # and pass every other assertion here.
        assert out._catalog.resolver is spool._catalog.resolver
        assert len(out) == 1
        assert out[0].shape == patch.shape

    def test_selector_shapes_match_the_attr_side(self, patch, inventory):
        """A glob, a sequence, and a scalar all mean what they mean there."""
        spool = dc.spool(patch).attach_inventory(inventory)
        scalar = spool.select(coupling="trench").get_contents()["distance_max"]
        for selector in ("tren*", "trenc?", ["trench", "conduit"]):
            got = spool.select(coupling=selector).get_contents()["distance_max"]
            assert got.tolist() == scalar.tolist()

    def test_a_numeric_range_selects_a_stretch(self, patch, inventory):
        """A qualified numeric field takes a range, as a coordinate does."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.select(**{"optical_components.optical_length": (400.0, 600.0)})
        assert len(out) == 1

    def test_none_matches_the_undefined_channels(self, patch, inventory):
        """
        `None` is how a query spells the marker absence is stored as.

        The path is coupled only to 250 m, so the channels past it hold
        the empty string a string coordinate has instead of a null.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.select(coupling=None)
        assert len(out) == 1
        assert out.get_contents()["distance_min"].tolist() == [151.0]

    def test_a_membership_group_takes_a_boolean(self, patch, inventory):
        """A membership group is False where nothing includes it."""
        spool = dc.spool(patch).attach_inventory(inventory)
        noisy = spool.select(noisy=True).get_contents()
        assert noisy["distance_min"].tolist() == [50.0]
        assert len(spool.select(noisy=False)) == 2

    def test_a_name_this_path_defines_nowhere_matches_nothing(self, patch, inventory):
        """
        Listing a name is not promising a value for it.

        One path recording a track makes the name selectable for the
        whole inventory, so a patch whose own path records none must
        answer with no channel rather than with an error.
        """
        array = inventory.networks[0].fiber_arrays[0]
        acquisition, path = array.acquisitions[0], array.optical_paths[0]
        both = inventory.replace(
            array,
            array.new(
                acquisitions=(acquisition, acquisition.new(location_code="01")),
                optical_paths=(path, path.new(location_code="01", coupling=())),
            ),
        )
        assert "coupling" in both.get_names().coords
        other = patch.update_attrs(acquisition_key="DAS.R2D1.01.RAW", tag="second")
        spool = dc.spool([patch, other]).attach_inventory(both)
        # The first patch's path is coupled; the second's records none.
        out = spool.select(coupling="trench")
        assert out.get_contents()["tag"].tolist() == ["random"]

    def test_the_patch_axis_keeps_its_own_meaning(self, patch, inventory):
        """
        `distance` is the patch's axis, inventory attached or not.

        The inventory could also place it on the fiber — its channels
        start at 100 m along the path — so a name the index already uses
        for a coordinate has to keep it, or attaching would move a name
        out of the namespace it has always been in.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        for form in ({}, {"_coords": {"distance": (0, 100)}}):
            kwargs = {"distance": (0, 100)} if not form else {}
            out = spool.select(**form, **kwargs)
            assert out.get_contents()["distance_max"].tolist() == [100.0]

    @pytest.mark.parametrize("flag", ["samples", "relative"])
    def test_axis_keywords_with_a_channel_name_raise(self, patch, inventory, flag):
        """
        Neither keyword has anything to say about a fiber coordinate.

        Both describe the patch's own axis, while these say what is
        attached to each channel of it. Ignoring one quietly would answer
        a different question than the caller asked.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match=f"{flag}=True cannot"):
            spool.select(coupling="trench", **{flag: True})

    def test_a_channel_name_through_attrs_raises(self, patch, inventory):
        """It describes channels, so the attrs namespace is the wrong one."""
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="along the fiber"):
            spool.select(_attrs={"coupling": "trench"})

    def test_an_acquisition_with_no_map_refuses(self, patch, inventory):
        """Its channels cannot be placed, and guessing would trim wrongly."""
        without = _replace_acquisition(inventory, distance_map=None)
        spool = dc.spool(patch).attach_inventory(without)
        with pytest.raises(PatchError, match="no distance_map"):
            spool.select(coupling="trench")

    def test_a_patch_without_the_axis_refuses(self, inventory, patch):
        """A patch carrying no dimension the map places channels by."""
        lag = patch.rename_coords(distance="offset")
        spool = dc.spool(lag).attach_inventory(inventory)
        with pytest.raises(PatchError, match="places channels by"):
            spool.select(coupling="trench")


class TestChannelUnselect:
    """Removing the channels a selection would have kept."""

    def test_complements_the_selection(self, patch, two_zones):
        """Together the two hold every channel, and share none."""
        spool = dc.spool(patch).attach_inventory(two_zones)
        kept = _channels(spool.select(hole="a"))
        dropped = _channels(spool.unselect(hole="a"))
        # Stated rather than merely complementary: kept=nothing and
        # dropped=everything satisfies a partition too, and is exactly
        # what the complement plumbing could produce by mistake.
        assert len(kept) == 82  # 10-50 and 200-240 inclusive
        assert not set(kept) & set(dropped)
        assert sorted([*kept, *dropped]) == sorted(patch.get_array("distance"))

    def test_removes_the_undefined_channels(self, patch, inventory):
        """The spec's own example: drop the channels with no coupling."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.unselect(coupling=None)
        assert out.get_contents()["distance_max"].tolist() == [150.0]

    def test_an_undescribed_patch_is_kept_whole(self, patch, inventory):
        """The selection never held it, so its complement keeps all of it."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        out = spool.unselect(coupling="trench")
        assert len(out) == 2
        assert sorted(out.get_contents()["distance_min"]) == [0.0, 151.0]

    def test_attrs_and_channels_stay_one_complement(self, patch, inventory):
        """
        A patch the attrs never matched keeps every channel.

        Complementing the two halves apart would drop it, since the
        selection it is the complement of never held it.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.unselect(gauge_length=99.0, coupling="trench")
        assert len(out) == 1
        assert out[0].shape == patch.shape

    def test_attrs_and_channels_trim_the_matched_patch(self, patch, inventory):
        """And one the attrs did match loses the channels which matched."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.unselect(gauge_length=10.0, coupling="trench")
        assert out.get_contents()["distance_min"].tolist() == [151.0]

    def test_a_patch_coordinate_still_raises(self, patch, inventory):
        """A real coordinate range is a trim, and unselect is not one."""
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="unselect cannot take"):
            spool.unselect(_coords={"distance": (0, 100)})


class TestSplitBy:
    """Expanding a spool into one patch per value along the fiber."""

    def test_one_patch_per_value(self, patch, inventory):
        """The example path annotates two zones, so two patches come out."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("zone")
        assert len(out) == 2
        assert out.get_contents()["zone"].tolist() == ["north", "south"]

    def test_the_pieces_hold_their_own_channels(self, patch, inventory):
        """Each output holds the rows its value covers, and no others."""
        spool = dc.spool(patch).attach_inventory(inventory)
        whole = patch.get_array("distance")
        pieces = list(spool.expand_by("zone"))
        assert len(pieces) == 2
        for piece in pieces:
            rows = np.searchsorted(whole, piece.get_array("distance"))
            assert np.array_equal(piece.data, patch.data[rows])

    def test_the_value_is_stamped_on_the_patch(self, patch, inventory):
        """So overlapping siblings stay apart once they are patches."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert [x.attrs.zone for x in spool.expand_by("zone")] == ["north", "south"]

    def test_stamp_false_leaves_the_attrs_alone(self, patch, inventory):
        """Which is what a nested split wants of the second one."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("zone", stamp=False)
        assert len(out) == 2  # the split still happened; only the stamp is off
        assert "zone" not in out.get_contents().columns
        assert not any(dict(x.attrs).get("zone") for x in out)

    def test_the_stamp_shadows_the_inventory_name(self, patch, inventory):
        """
        A stamped value is an ordinary attr, and attrs win over the fiber.

        Both paths would keep one patch here, so the patch's shape is
        what tells them apart: the attr path keeps the north piece whole,
        while resolving `zone` along the fiber again would trim it.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("zone")
        north = out.select(zone="north")
        assert len(north) == 1
        assert north[0].shape == out[0].shape

    def test_a_membership_group_splits_in_two(self, patch, inventory):
        """Both sides come out: the channels included and those not."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("noisy")
        assert sorted(out.get_contents()["noisy"].tolist()) == [False, False, True]

    def test_one_split_partitions_the_channels(self, patch, inventory):
        """
        A channel holds one value of a group, so one split cannot share it.

        Overlapping intervals of a group resolve to a single value per
        channel — the projection `Patch.enrich` uses — so the outputs of
        one call divide the fiber rather than covering it twice. Two
        *different* groups may still cut it differently, which is what
        makes a nested split worth doing.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        for group in ("zone", "noisy"):
            channels = _channels(spool.expand_by(group))
            assert sorted(channels) == sorted(patch.get_array("distance"))
            assert len(channels) == len(set(channels.tolist()))
        # The two groups do not agree about where the fiber divides.
        zoned = {tuple(x.get_array("distance")) for x in spool.expand_by("zone")}
        noisy = {tuple(x.get_array("distance")) for x in spool.expand_by("noisy")}
        assert zoned != noisy

    def test_a_disjoint_value_becomes_several_patches(self, patch, two_zones):
        """One value covering two stretches keeps them apart."""
        spool = dc.spool(patch).attach_inventory(two_zones)
        out = spool.expand_by("hole")
        assert len(out) == 2
        assert out.get_contents()["hole"].tolist() == ["a", "a"]

    def test_include_keeps_only_what_it_names(self, patch, inventory):
        """The globs read the value written as a string."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("zone", include="nor*")
        assert out.get_contents()["zone"].tolist() == ["north"]

    def test_exclude_wins_over_include(self, patch, inventory):
        """So naming a family and carving one out of it reads either way."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("zone", include=("nor*", "sou*"), exclude="north")
        assert out.get_contents()["zone"].tolist() == ["south"]

    def test_a_name_the_inventory_lacks_raises(self, patch, inventory):
        """
        A misspelling has no values to split into, so it says so.

        Returning an empty spool would be indistinguishable from a group
        which happens to cover nothing, and selection refuses a name it
        does not know for the same reason.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="not a coordinate"):
            spool.expand_by("not_a_group")

    def test_needs_an_inventory(self, patch):
        """The values it expands into are ones an inventory states."""
        with pytest.raises(ParameterError, match="needs an inventory"):
            dc.spool(patch).expand_by("zone")

    def test_a_nested_split_keeps_the_first_stamp(self, patch, inventory):
        """Which is what `stamp=False` is for."""
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.expand_by("zone").expand_by("noisy", stamp=False)
        assert set(out.get_contents()["zone"]) == {"north", "south"}


class TestChannelSelectEdges:
    """Rows a fiber query cannot answer for, and the ways it says so."""

    def test_a_lag_time_patch_has_no_epoch_to_resolve_at(self, patch, inventory):
        """
        Without instants there is no context, so nothing is selected.

        The same thing `Patch.enrich` refuses to guess at: a correlation's
        lag times are not the moments the fiber was in some state.
        """
        lags = patch.get_coord("time").values - patch.get_coord("time").min()
        spool = dc.spool(patch.update_coords(time=lags)).attach_inventory(inventory)
        assert len(spool.select(coupling="trench")) == 0

    def test_an_empty_spool_stays_empty(self, patch, inventory):
        """With no rows there is nothing to resolve, and no work to do."""
        spool = dc.spool(patch).attach_inventory(inventory).select(tag="nope")
        assert len(spool) == 0
        assert len(spool.select(coupling="trench")) == 0

    def test_an_unevenly_sampled_patch_refuses(self, patch, inventory):
        """
        Which channels match is decided on the sample grid.

        A patch with a hole in its distance coordinate records no
        spacing, so the grid cannot be rebuilt; trimming the wrong
        channels quietly is worse than saying so.
        """
        holed = patch.unselect(distance=(50, 200))
        assert holed.get_coord("distance").step is None
        spool = dc.spool(holed).attach_inventory(inventory)
        with pytest.raises(PatchError, match="no channel spacing"):
            spool.select(coupling="trench")

    def test_patches_on_different_channel_dimensions_refuse(self, patch, inventory):
        """
        One spool, one dimension to trim: two is no answer at all.

        The patch-level twin refuses the same shape, where one patch
        carries two coordinates the map could be read on.
        """
        distance = patch.get_coord("distance")
        both_axes = _replace_acquisition(
            inventory,
            distance_map=DistanceMap(
                channel=(float(distance.min()), float(distance.max())),
                instrument_distance=(float(distance.min()), float(distance.max())),
                distance=(100.0, 100.0 + float(distance.max() - distance.min())),
            ),
        )
        channels = dc.Patch(
            data=patch.data,
            coords={
                "channel": patch.get_array("distance"),
                "time": patch.get_array("time"),
            },
            dims=("channel", "time"),
            attrs=patch.attrs,
        )
        spool = dc.spool([patch, channels]).attach_inventory(both_axes)
        with pytest.raises(InvalidSpoolQueryError, match="different"):
            spool.select(coupling="trench")

    def test_none_on_a_membership_group(self, patch, inventory):
        """A membership group says something about every channel: False."""
        spool = dc.spool(patch).attach_inventory(inventory)
        # noisy runs from 150 to 300 m, which is patch distance 50 to 200,
        # so the channels it says nothing about are those on either side.
        undefined = spool.select(noisy=None).get_contents()
        assert undefined["distance_min"].tolist() == [0.0, 201.0]
        assert undefined["distance_max"].tolist() == [49.0, 299.0]
        assert undefined["distance_min"].tolist() == (
            spool.select(noisy=False).get_contents()["distance_min"].tolist()
        )

    def test_none_on_a_numeric_group(self, patch, inventory):
        """A numeric group spells absence NaN, which no range matches."""
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        numeric = inventory.replace(
            path,
            path.new(
                annotations=(
                    *path.annotations,
                    OpticalPathAnnotation(
                        start_distance=100.0,
                        end_distance=200.0,
                        group="frost_depth",
                        value=1.5,
                    ),
                )
            ),
        )
        spool = dc.spool(patch).attach_inventory(numeric)
        # The group covers the first hundred channels, so the rest are NaN.
        assert spool.select(frost_depth=None).get_contents()[
            "distance_min"
        ].tolist() == [101.0]
        assert spool.select(frost_depth=(1.0, 2.0)).get_contents()[
            "distance_max"
        ].tolist() == [100.0]

    def test_splitting_an_undescribed_spool_yields_nothing(self, patch, inventory):
        """No fiber to split on means no output, not an error."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        spool = dc.spool(other).attach_inventory(inventory)
        assert len(spool.expand_by("zone")) == 0

    def test_splitting_skips_a_row_it_cannot_place(self, patch, inventory):
        """A described patch splits; one the inventory is silent about does not."""
        other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER", tag="second")
        spool = dc.spool([patch, other]).attach_inventory(inventory)
        out = spool.expand_by("zone")
        assert out.get_contents()["zone"].tolist() == ["north", "south"]

    def test_splitting_an_unevenly_sampled_patch_refuses(self, patch, inventory):
        """The grid is what values are read on here too."""
        holed = patch.unselect(distance=(50, 200))
        spool = dc.spool(holed).attach_inventory(inventory)
        with pytest.raises(PatchError, match="no channel spacing"):
            spool.expand_by("zone")

    def test_one_patch_carrying_two_channel_axes_refuses(self, patch, inventory):
        """
        Both are dimensions, so neither is obviously the channel one.

        The patch-level resolver settles this by projecting each and
        checking they agree; two *dimensions* are different axes rather
        than two spellings of one, so there is nothing to check.
        """
        distance = patch.get_coord("distance")
        both_axes = _replace_acquisition(
            inventory,
            distance_map=DistanceMap(
                channel=(float(distance.min()), float(distance.max())),
                instrument_distance=(float(distance.min()), float(distance.max())),
                distance=(100.0, 100.0 + float(distance.max() - distance.min())),
            ),
        )
        stacked = patch.data[:3][..., np.newaxis]
        cube = dc.Patch(
            data=np.broadcast_to(stacked, (3, patch.shape[1], 2)).copy(),
            coords={
                "channel": np.arange(3.0),
                "time": patch.get_array("time"),
                "distance": np.arange(2.0),
            },
            dims=("channel", "time", "distance"),
            attrs=patch.attrs,
        )
        spool = dc.spool(cube).attach_inventory(both_axes)
        with pytest.raises(PatchError, match="ambiguous"):
            spool.select(coupling="trench")

    def test_splitting_a_lag_time_patch_yields_nothing(self, patch, inventory):
        """Without instants there is no context to read a group from."""
        lags = patch.get_coord("time").values - patch.get_coord("time").min()
        spool = dc.spool(patch.update_coords(time=lags)).attach_inventory(inventory)
        assert len(spool.expand_by("zone")) == 0


@pytest.fixture(scope="module")
def uneven_spool(patch, inventory):
    """
    Three patches whose fiber answers differ, in one spool.

    The first resolves to a path putting one group in two stretches, the
    second to a path putting it in one, and the third to nothing at all.
    """
    array = inventory.networks[0].fiber_arrays[0]
    acquisition, path = array.acquisitions[0], array.optical_paths[0]
    holes = (
        OpticalPathAnnotation(
            start_distance=110.0, end_distance=150.0, group="hole", value="a"
        ),
        OpticalPathAnnotation(
            start_distance=300.0, end_distance=340.0, group="hole", value="a"
        ),
    )
    both = inventory.replace(
        array,
        array.new(
            acquisitions=(acquisition, acquisition.new(location_code="01")),
            optical_paths=(
                path.new(annotations=(*path.annotations, *holes)),
                path.new(location_code="01", annotations=(*path.annotations, holes[0])),
            ),
        ),
    )
    patches = [
        patch.update_attrs(tag="split"),
        patch.update_attrs(acquisition_key="DAS.R2D1.01.RAW", tag="solid"),
        patch.update_attrs(acquisition_key="DAS.R2D1..NOPE", tag="unknown"),
    ]
    return dc.spool(patches).attach_inventory(both), patches


class TestChannelSelectAlignment:
    """
    Rows subdividing by different amounts must not swap answers.

    One patch becoming two while its neighbour becomes one and a third
    leaves entirely is where a plan built by position rather than by
    identity would quietly hand a patch someone else's envelope.
    """

    def test_each_patch_keeps_its_own_pieces(self, uneven_spool):
        """The split patch gets two, the solid one gets one, the third none."""
        spool, _ = uneven_spool
        contents = spool.select(hole="a").get_contents()
        by_tag = contents.groupby("tag")["distance_min"].apply(sorted)
        assert by_tag["split"] == [10.0, 200.0]
        assert by_tag["solid"] == [10.0]
        assert "unknown" not in by_tag

    def test_each_piece_holds_its_own_patch_data(self, uneven_spool):
        """And the array behind each envelope is that patch's, not a neighbour's."""
        spool, patches = uneven_spool
        sources = {x.attrs.tag: x for x in patches}
        whole = patches[0].get_array("distance")
        for piece in spool.select(hole="a"):
            source = sources[piece.attrs.tag]
            rows = np.searchsorted(whole, piece.get_array("distance"))
            assert np.array_equal(piece.data, source.data[rows])

    def test_the_complement_is_exact_for_every_patch(self, uneven_spool):
        """Including the one the inventory says nothing about, kept whole."""
        spool, patches = uneven_spool
        whole = sorted(patches[0].get_array("distance"))
        kept, dropped = {}, {}
        for store, out in (
            (kept, spool.select(hole="a")),
            (dropped, spool.unselect(hole="a")),
        ):
            for piece in out:
                store.setdefault(piece.attrs.tag, []).extend(
                    piece.get_array("distance")
                )
        for tag in ("split", "solid"):
            assert not set(kept[tag]) & set(dropped[tag])
            assert sorted([*kept[tag], *dropped[tag]]) == whole
        assert sorted(dropped["unknown"]) == whole

    def test_splitting_keeps_each_patch_with_its_value(self, uneven_spool):
        """expand_by builds the same plan, so it aligns the same way."""
        spool, patches = uneven_spool
        sources = {x.attrs.tag: x for x in patches}
        whole = patches[0].get_array("distance")
        out = spool.expand_by("hole")
        assert out.get_contents()["hole"].tolist() == ["a", "a", "a"]
        for piece in out:
            source = sources[piece.attrs.tag]
            rows = np.searchsorted(whole, piece.get_array("distance"))
            assert np.array_equal(piece.data, source.data[rows])


def _float_grid_pair(distance, span):
    """A patch on a given float grid, and an inventory annotating its middle."""
    time = get_example_patch().get_array("time")
    data = np.random.default_rng(0).random((len(distance), len(time)))
    patch = dc.Patch(
        data=data,
        coords={"distance": distance, "time": time},
        dims=("distance", "time"),
        attrs={"acquisition_key": "DAS.R2D1..RAW", "category": "DAS"},
    )
    acquisition = Acquisition(
        code="RAW",
        location_code="",
        data_type="velocity",
        data_category="DAS",
        gauge_length=10.0,
        sample_rate=1.0 / dc.to_float(patch.get_coord("time").step),
        distance_map=DistanceMap(
            instrument_distance=(float(distance.min()), float(distance.max())),
            distance=(0.0, span),
        ),
    )
    path = OpticalPath(
        name="main",
        location_code="",
        optical_components=(FiberSegment(name="c", optical_length=span + 10),),
        annotations=(
            OpticalPathAnnotation(
                start_distance=span * 0.23,
                end_distance=span * 0.61,
                group="zone",
                value="mid",
            ),
        ),
    )
    return patch, Inventory(
        networks=(
            Network(
                code="DAS",
                fiber_arrays=(
                    FiberArray(
                        code="R2D1",
                        acquisitions=(acquisition,),
                        optical_paths=(path,),
                    ),
                ),
            ),
        )
    ).check()


class TestChannelReviewFindings:
    """Defects the review pipeline found, each with its own scenario."""

    def test_a_reverse_sorted_patch_is_still_placed(self, patch, inventory):
        """
        A descending coordinate states a negative step, and envelopes
        do not: counting samples with the signed one gives none at all,
        so every fiber query would silently drop the patch.
        """
        reversed_patch = patch.sort_coords("distance", reverse=True)
        spool = dc.spool(reversed_patch).attach_inventory(inventory)
        assert spool.get_contents()["distance_step"].iloc[0] < 0
        out = spool.select(coupling="trench")
        assert len(out) == 1
        assert out.get_contents()["distance_max"].tolist() == [150.0]
        assert out[0].shape == (151, patch.shape[1])

    def test_an_acquisition_without_a_path_states_nothing(self, patch, inventory):
        """
        A pathless acquisition is valid, and simply projects nothing.

        The CRS labels stay listed whatever the paths say, so a geometry
        axis is still a name a query may use — and must answer with no
        channel rather than by dereferencing the absent path.
        """
        array = inventory.networks[0].fiber_arrays[0]
        pathless = inventory.replace(array, array.new(optical_paths=()))
        spool = dc.spool(patch).attach_inventory(pathless)
        assert "x" in pathless.get_names().coords
        assert len(spool.select(x=(-1e9, 1e9))) == 0
        assert len(spool.expand_by("x")) == 0

    def test_a_unit_bearing_selector_is_converted(self, patch, inventory):
        """
        The inventory documents the units of the fields it projects, so
        dropping them would refuse selectors the index accepts against a
        stated attr of the same kind.
        """
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        deep = inventory.replace(
            path,
            path.new(
                coupling=(
                    CouplingCondition(
                        start_distance=100.0,
                        end_distance=250.0,
                        coupling_type="trench",
                        medium="soil",
                        depth=2.0,
                    ),
                )
            ),
        )
        spool = dc.spool(patch).attach_inventory(deep)
        bare = spool.select(**{"coupling.depth": (1.0, 3.0)})
        assert bare.get_contents()["distance_max"].tolist() == [150.0]
        for equivalent in ((1 * m, 3 * m), (100 * cm, 300 * cm)):
            out = spool.select(**{"coupling.depth": equivalent})
            assert out.get_contents()["distance_max"].tolist() == [150.0]

    def test_a_bare_ellipsis_selects_everything(self, patch, inventory):
        """
        As it does everywhere else, so attaching an inventory does not
        turn a no-op selector into an error. `None` is the exception,
        and means the undefined marker rather than "everything".
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        assert len(spool.select(coupling=...)) == 1
        assert spool.select(coupling=...)[0].shape == patch.shape

    def test_unselect_takes_a_channel_name_through_coords(self, patch, inventory):
        """
        Both spellings `select` accepts, since `_coords` is the only way
        to name a group which shares an acquisition field's name.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        expected = [151.0]
        assert (
            spool.unselect(coupling="trench").get_contents()["distance_min"].tolist()
            == expected
        )
        assert (
            spool.unselect(_coords={"coupling": "trench"})
            .get_contents()["distance_min"]
            .tolist()
            == expected
        )
        assert (
            spool.unselect(_coords="coupling", coupling="trench")
            .get_contents()["distance_min"]
            .tolist()
            == expected
        )

    def test_rechunking_a_selection_keeps_it(self, patch, inventory):
        """
        Re-planning the same dimension collapses onto the sources, which
        is sound only while the plan's pieces cover them. A selection's
        do not, so collapsing would load back the channels it removed —
        and the contents would go on describing the ones it kept.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        selected = spool.select(coupling="trench")
        assert selected[0].shape == (151, patch.shape[1])
        rechunked = selected.chunk(distance=None)
        assert rechunked.get_contents()["distance_max"].tolist() == [150.0]
        assert rechunked[0].shape == (151, patch.shape[1])

    def test_only_a_plan_which_drops_samples_is_lossy(
        self, patch, inventory, path_epochs
    ):
        """
        The other side of it: conform's pieces do cover their row, so it
        stays collapsible and only a selection does not.

        Checked on the plans themselves rather than by re-chunking each,
        because chunking a derived catalog is broken on dev already
        (#871) and would fail here for a reason of its own.
        """
        conformed = dc.spool(patch).attach_inventory(path_epochs).conform_to_inventory()
        assert len(conformed) == 2
        assert not conformed._catalog.resolver.lossy
        selected = dc.spool(patch).attach_inventory(inventory).select(coupling="trench")
        assert selected._catalog.resolver.lossy

    def test_a_stamp_cannot_overwrite_the_plan(self, patch, inventory):
        """
        A group may be named anything the inventory does not reserve,
        and the stamp is assigned onto the plan's outputs — so a group
        called `output_id` would replace what binds each output to the
        data it came from, and the catalog would be built from nonsense.
        """
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        clash = inventory.replace(
            path,
            path.new(
                annotations=(
                    OpticalPathAnnotation(
                        start_distance=100.0,
                        end_distance=200.0,
                        group="output_id",
                        value="a",
                    ),
                )
            ),
        )
        spool = dc.spool(patch).attach_inventory(clash)
        with pytest.raises(InvalidSpoolQueryError, match="overwrite"):
            spool.expand_by("output_id")
        # Splitting on it is still legal; only recording the value is not.
        assert len(spool.expand_by("output_id", stamp=False)) == 1

    def test_a_no_op_channel_selector_does_not_veto_a_flag(self, patch, inventory):
        """
        `...` names a fiber coordinate without asking anything of it, so
        it must not refuse a `samples` the rest of the query needs.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        out = spool.select(distance=(0, 10), coupling=..., samples=True)
        assert len(out) == 1
        assert out[0].shape == (10, patch.shape[1])
        # A selector which does ask something still refuses it.
        with pytest.raises(InvalidSpoolQueryError, match="samples=True cannot"):
            spool.select(coupling="trench", samples=True)

    def test_a_scalar_quantity_selector_is_converted(self, patch, inventory):
        """
        A geometry axis is stored in the CRS's own units while a quantity
        selector is typed in its base unit, so comparing the magnitudes
        unconverted matched nothing — where the range form matched.
        """
        spool = dc.spool(patch).attach_inventory(inventory)
        enriched = patch.enrich(inventory, coords=("y",), attrs=False)
        stated = float(enriched.get_array("y")[0])
        degrees = get_quantity("degree")
        assert len(spool.select(y=stated)) == 1
        assert len(spool.select(y=stated * degrees)) == 1

    @pytest.mark.parametrize(
        ("start", "step", "size"),
        [
            (1.0, 0.1, 300),
            (0.1, 0.25, 500),
            (-1234.5678, 0.037, 400),
            (0.0, 1 / 3, 350),
        ],
    )
    def test_the_index_selects_what_enrichment_projects(self, start, step, size):
        """
        A rebuilt grid must name the same channels the patch's own does.

        The grid is reconstructed from the index envelope rather than
        read off the patch, and `Patch.select` does not snap an interior
        bound to the nearest sample — so a float grid which drifted from
        the patch's own values could trim one channel too many. Compared
        by count and position rather than by value, since a trimmed
        CoordRange regenerates its values from the piece's own start and
        so differs in the last ulp for any trim, inventory or not.
        """
        distance = start + np.arange(size) * step
        span = float(distance.max() - distance.min())
        patch, inventory = _float_grid_pair(distance, span)
        spool = dc.spool(patch).attach_inventory(inventory)
        selected = spool.select(zone="mid")
        got = np.concatenate([x.get_array("distance") for x in selected])
        projected = patch.enrich(inventory, coords=("zone",), attrs=False)
        wanted = distance[projected.get_array("zone") == "mid"]
        assert len(wanted)  # the group really does cover part of this fiber
        assert len(got) == len(wanted)
        assert np.allclose(np.sort(got), np.sort(wanted), rtol=0, atol=1e-9)
