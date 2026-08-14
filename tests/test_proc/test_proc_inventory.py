"""Tests for copying inventory metadata onto a patch."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.inventory import (
    Acquisition,
    CoordinateReferenceSystem,
    DistanceMap,
    FiberArray,
    Geometry,
    Inventory,
    Network,
    OpticalPath,
    OpticalPathAnnotation,
)
from dascore.examples import inventory_patch_pair
from dascore.exceptions import (
    InvalidInventoryError,
    ParameterError,
    PatchError,
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


class TestProjectionDetails:
    """Units, maps, and refresh details of enrich's projection."""

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


class TestEnrichContracts:
    """Contracts enrich keeps at track ends, markers, and channel maps."""

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
