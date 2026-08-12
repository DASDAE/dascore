"""Tests for copying inventory metadata onto a patch."""

from __future__ import annotations

import inspect
import itertools
import pickle
import re
import warnings
from collections.abc import Mapping

import numpy as np
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.constants import (
    enrich_attrs_description,
    enrich_conflicts_description,
    enrich_coords_description,
    enrich_on_missing_description,
)
from dascore.core.inventory import (
    Acquisition,
    CoordinateReferenceSystem,
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
from dascore.examples import inventory_patch_pair
from dascore.exceptions import (
    InvalidInventoryError,
    InvalidSpoolError,
    InvalidSpoolQueryError,
    ParameterError,
    PatchError,
    UnitError,
    UnresolvedPatchError,
)


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
            stale.enrich(inventory, coords=False, conflicts="raise")

    def test_drop_removes_the_attr(self, patch, inventory):
        """A dropped conflict leaves neither value behind."""
        stale = patch.update_attrs(gauge_length=99.0)
        out = stale.enrich(inventory, coords=False, conflicts="drop")
        assert "gauge_length" not in dict(out.attrs)

    def test_equal_values_are_not_conflicts(self, patch, inventory):
        """Agreement is not disagreement, whatever the policy."""
        agreeing = patch.update_attrs(gauge_length=10.0)
        out = agreeing.enrich(inventory, coords=False, conflicts="raise")
        assert out.attrs.gauge_length == 10.0

    def test_filling_is_not_a_conflict(self, patch, inventory):
        """An attr the patch does not have cannot disagree with one it lacks."""
        out = patch.enrich(inventory, coords=False, conflicts="raise")
        assert out.attrs.gauge_length == 10.0

    def test_re_enrich_is_a_refresh(self, patch, inventory):
        """Enriching twice is not an error and changes nothing."""
        once = patch.enrich(inventory, coords=False)
        twice = once.enrich(inventory, coords=False)
        assert dict(twice.attrs.drop("history")) == dict(once.attrs.drop("history"))

    def test_bad_conflicts_raises(self, patch, inventory):
        """The flag shares chunking's vocabulary and its validation."""
        with pytest.raises(ParameterError, match="conflict"):
            patch.enrich(inventory, conflicts="whatever")

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
            odd.enrich(inventory, coords=False, conflicts="raise")

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
        """Anything else names no metadata to attach."""
        with pytest.raises(ParameterError, match="needs an Inventory"):
            dc.spool(patch).attach_inventory("inventory.yaml")


class TestRemoveInventory:
    """Removing an inventory takes its enrichment with it."""

    def test_removes_inventory_and_enrichment(self, patch, inventory):
        """Nothing is left to enrich from, so nothing is enriched."""
        spool = dc.spool(patch).enrich(inventory).remove_inventory()
        assert spool._inventory is None
        assert spool._enrich_kwargs is None
        assert "gauge_length" not in dict(spool[0].attrs)

    def test_original_is_unchanged(self, patch, inventory):
        """Removal is a new spool; the one it came from still enriches."""
        spool = dc.spool(patch).enrich(inventory)
        spool.remove_inventory()
        assert spool[0].attrs.gauge_length == 10.0

    def test_no_inventory_is_a_no_op(self, patch):
        """A spool with nothing to remove is returned as it stands."""
        plain = dc.spool(patch)
        assert plain.remove_inventory() == plain

    def test_keeps_the_patches(self, patch, inventory):
        """Only the metadata goes; the spool's contents do not."""
        spool = dc.spool(patch).enrich(inventory)
        assert spool.remove_inventory()[0].equals(patch)


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
            out = list(mixed.enrich(inventory))
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
            list(dc.spool(strangers).enrich(inventory))
        assert len(caught) == 1

    def test_each_spool_says_it_once(self, patch, inventory):
        """Deduping on the message text would silence every spool after the
        first in a session, hiding the second one's problem entirely.
        """
        first = dc.spool([patch.update_attrs(acquisition_key="XX.A1..RAW")])
        second = dc.spool([patch.update_attrs(acquisition_key="XX.B2..RAW")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            list(first.enrich(inventory))
            list(second.enrich(inventory))
        assert len(caught) == 2

    def test_ignore_is_silent(self, mixed, inventory):
        """The same, without saying so."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = list(mixed.enrich(inventory, on_unresolved="ignore"))
        assert len(out) == len(mixed)

    def test_raise_fails(self, mixed, inventory):
        """A spool which must be fully described can say so."""
        with pytest.raises(UnresolvedPatchError, match="no acquisition_key"):
            list(mixed.enrich(inventory, on_unresolved="raise"))

    def test_unknown_id_is_unresolved(self, patch, inventory):
        """Naming an entry the inventory lacks is not being described either."""
        stranger = patch.update_attrs(acquisition_key="XX.R2D1..RAW")
        with pytest.warns(UserWarning, match="does not describe every patch"):
            out = dc.spool(stranger).enrich(inventory)[0]
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
        spool = dc.spool(patch).enrich(split, on_unresolved="ignore")
        with pytest.raises(PatchError, match="spans a change of acquisition"):
            spool[0]

    def test_malformed_explicit_id_is_not_swallowed(self, patch, inventory):
        """A caller getting the argument wrong is not the inventory's silence."""
        bare = dc.spool(patch.update_attrs(acquisition_key=""))
        spool = bare.enrich(inventory, acquisition_key="broken", on_unresolved="ignore")
        with pytest.raises(ValueError, match="Invalid acquisition_key"):
            spool[0]

    def test_bad_policy_raises(self, patch, inventory):
        """An unknown policy names no behavior."""
        with pytest.raises(ParameterError, match="on_unresolved must be one of"):
            dc.spool(patch).enrich(inventory, on_unresolved="nope")

    def test_policy_is_part_of_the_spool(self, patch, inventory):
        """Two policies mean two different behaviors on extraction."""
        plain = dc.spool(patch)
        assert plain.enrich(inventory) != plain.enrich(inventory, on_unresolved="raise")
        assert plain.enrich(inventory) == plain.enrich(inventory, on_unresolved="warn")

    def test_policy_survives_a_union(self, patch, inventory):
        """The policy is part of the enrichment, so it carries with it."""
        enriched = dc.spool(patch).enrich(inventory, on_unresolved="raise")
        assert (enriched + dc.spool(patch))._on_unresolved == "raise"


class TestInventoryQueryError:
    """An attached inventory changes what an unknown query name means."""

    def test_inventory_field_names_the_inventory(self, patch, inventory):
        """The index's 'no such attribute' would deny a field which exists."""
        spool = dc.spool(patch).attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="along the fiber"):
            spool.select(coupling="cement")

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
        spool = dc.spool(patch).enrich(inventory)
        assert spool[0].attrs.gauge_length == 10.0

    def test_iteration_is_enriched(self, patch, inventory):
        """So does one pulled by iteration."""
        spool = dc.spool(patch).enrich(inventory)
        assert all(x.attrs.gauge_length == 10.0 for x in spool)

    def test_uses_attached_inventory(self, patch, inventory):
        """With one already carried, enrich needs no argument."""
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        assert spool[0].attrs.gauge_length == 10.0

    def test_enrich_attaches_its_inventory(self, patch, inventory):
        """The spool carries what it enriches from."""
        spool = dc.spool(patch).enrich(inventory)
        assert spool._inventory is inventory

    def test_kwargs_pass_through(self, patch, inventory):
        """Enrich's arguments are the spool's arguments."""
        spool = dc.spool(patch).enrich(inventory, attrs=("gauge_length",), coords=False)
        out = spool[0]
        assert out.attrs.gauge_length == 10.0
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)

    def test_derived_spool_stays_enriched(self, patch, inventory):
        """A selection is still the spool it came from."""
        spool = dc.spool(patch).enrich(inventory)
        assert spool.select(tag="random")[0].attrs.gauge_length == 10.0

    def test_original_is_unchanged(self, patch, inventory):
        """Setting up enrichment leaves the spool it came from alone."""
        plain = dc.spool(patch)
        plain.enrich(inventory)
        assert "gauge_length" not in dict(plain[0].attrs)

    def test_equality_includes_arguments(self, patch, inventory):
        """Different arguments mean different patches come out."""
        plain = dc.spool(patch)
        assert plain.enrich(inventory) != plain.attach_inventory(inventory)
        assert plain.enrich(inventory) == plain.enrich(inventory)
        assert plain.enrich(inventory) != plain.enrich(inventory, coords=False)

    def test_requires_an_inventory(self, patch):
        """With none attached and none given there is nothing to copy."""
        with pytest.raises(ParameterError, match="needs an inventory"):
            dc.spool(patch).enrich()

    def test_unknown_kwarg_raises_now(self, patch, inventory):
        """A misspelled argument fails here, not on a patch pulled later."""
        with pytest.raises(ParameterError, match="unknown argument"):
            dc.spool(patch).enrich(inventory, attr=True)

    def test_re_enriching_replaces_arguments(self, patch, inventory):
        """The last call says what enrichment means."""
        spool = dc.spool(patch).enrich(inventory, coords=False).enrich(inventory)
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
            enrich_conflicts_description,
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
        enriched = dc.spool(patch).enrich(inventory)
        combined = enriched + dc.spool(patch)
        assert combined[0].attrs.gauge_length == 10.0
        assert (dc.spool(patch) + enriched)[0].attrs.gauge_length == 10.0

    def test_union_of_different_enrichment_raises(self, patch, inventory):
        """Two ways of enriching have no combined meaning."""
        first = dc.spool(patch).enrich(inventory)
        second = dc.spool(patch).enrich(inventory, coords=False)
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
        out = unknown.enrich(inventory, coords=False, conflicts="raise")
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
            conflicts="raise",
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
        spool = dc.spool(patch).enrich(inventory)
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
        spool = dc.spool(patch).enrich(inventory).attach_inventory(inventory)
        assert spool._enrich_kwargs is None
        assert "gauge_length" not in dict(spool[0].attrs)

    def test_attaching_the_same_inventory_keeps_a_union_working(self, patch, inventory):
        """Supplying more information must not break what already worked."""
        enriched = dc.spool(patch).enrich(inventory)
        attached = dc.spool(patch).attach_inventory(inventory)
        assert (enriched + attached)[0].attrs.gauge_length == 10.0

    def test_stating_a_default_does_not_change_the_spool(self, patch, inventory):
        """Spools which enrich identically are the same spool."""
        plain = dc.spool(patch)
        assert plain.enrich(inventory) == plain.enrich(
            inventory, attrs=True, coords=True, conflicts="keep_first"
        )

    def test_name_collections_compare_by_content(self, patch, inventory):
        """A list of names asks for what a tuple of them asks for."""
        plain = dc.spool(patch)
        tupled = plain.enrich(inventory, attrs=("gauge_length",))
        assert tupled == plain.enrich(inventory, attrs=["gauge_length"])

    def test_update_keeps_the_inventory(self, patch, inventory, tmp_path):
        """Re-reading the file is no reason to stop enriching."""
        path = tmp_path / "patch.h5"
        dc.write(patch, path, "dasdae")
        spool = dc.spool(path).enrich(inventory)
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
        spool = dc.spool(patch.update_attrs(acquisition_key="")).enrich(
            inventory, acquisition_key=long_key, on_unresolved="ignore"
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
        spool = two_patch_spool.enrich(inventory)
        out = spool.select(gauge_length=10.0)
        assert out[0].attrs.gauge_length == 10.0

    def test_original_is_unchanged(self, two_patch_spool, inventory):
        """Selecting leaves the spool it came from alone."""
        spool = two_patch_spool.attach_inventory(inventory)
        spool.select(gauge_length=99.0)
        assert len(spool) == 2

    def test_channel_level_name_says_so(self, two_patch_spool, inventory):
        """
        A track name is a real inventory name which select cannot use yet.

        Saying which of the two it is takes the names accessor, and this
        is what it was added for.
        """
        spool = two_patch_spool.attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="along the fiber"):
            spool.select(coupling="trench")
        with pytest.raises(InvalidSpoolQueryError, match="along the fiber"):
            spool.select(zone="east")

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

    def test_channel_level_name_says_so(self, two_patch_spool, inventory):
        """A track name is as unsupported here as it is in select."""
        spool = two_patch_spool.attach_inventory(inventory)
        with pytest.raises(InvalidSpoolQueryError, match="along the fiber"):
            spool.unselect(coupling="trench")


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
        group, which is the channel-level half and not supported yet.
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
        for form in ({"gauge_length": 10.0}, "gauge_length", ["gauge_length"]):
            # The mapping form and both tag forms all name the coordinate.
            kwargs = {} if isinstance(form, Mapping) else {"gauge_length": 10.0}
            with pytest.raises(InvalidSpoolQueryError, match="along the fiber"):
                spool.select(_coords=form, **kwargs)

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
