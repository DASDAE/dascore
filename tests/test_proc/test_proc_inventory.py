"""Tests for copying inventory metadata onto a patch."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.inventory import (
    Acquisition,
    CoordinateReferenceSystem,
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
        bare = patch.update_attrs(data_source_id="")
        out = bare.enrich(inventory, data_source_id="DAS.R2D1..RAW", coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_no_id_raises(self, patch, inventory):
        """A patch naming no entry cannot be resolved."""
        bare = patch.update_attrs(data_source_id="")
        with pytest.raises(PatchError, match="no data_source_id"):
            bare.enrich(inventory)

    def test_disagreeing_id_raises(self, patch, inventory):
        """Two different ids are two different data sources."""
        with pytest.raises(PatchError, match="disagree"):
            patch.enrich(inventory, data_source_id="DAS.R2D1..OTHER")

    def test_agreeing_id_is_allowed(self, patch, inventory):
        """Repeating the patch's own id is not a disagreement."""
        out = patch.enrich(inventory, data_source_id="DAS.R2D1..RAW", coords=False)
        assert out.attrs.gauge_length == 10.0

    def test_unknown_id_raises(self, patch, inventory):
        """An id the inventory does not contain resolves to nothing."""
        with pytest.raises(InvalidInventoryError, match="resolves to 0"):
            patch.update_attrs(data_source_id="XX.R2D1..RAW").enrich(inventory)

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
        assert attrs.spatial_interval == 1.0
        assert attrs.get("interrogator.serial_number") == "sn-1"
        assert attrs.get("interrogator.model") == "FI-1"

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

    def test_missing_named_attr_nan(self, patch, inventory):
        """It can instead be filled with the missing marker."""
        out = patch.enrich(
            inventory, attrs=("pulse_rate",), coords=False, on_missing="nan"
        )
        assert np.isnan(out.attrs.pulse_rate)

    def test_missing_named_attr_skip(self, patch, inventory):
        """Or omitted entirely."""
        out = patch.enrich(
            inventory, attrs=("pulse_rate",), coords=False, on_missing="skip"
        )
        assert "pulse_rate" not in dict(out.attrs)

    def test_adds_nothing_unasked(self, patch, inventory):
        """Enrich never leaves the patch holding inventory state."""
        out = patch.enrich(inventory, attrs=False, coords=False)
        assert dict(out.attrs.drop("history")) == dict(patch.attrs.drop("history"))

    def test_identity_and_native_untouched(self, patch, inventory):
        """data_source_id and tag belong to the patch, not the inventory."""
        out = patch.enrich(inventory, coords=False)
        assert out.attrs.data_source_id == patch.attrs.data_source_id
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

    def test_affine_needs_channel_coord(self, patch, inventory):
        """The affine form maps channel numbers, which this patch lacks."""
        affine = _replace_acquisition(
            inventory, distance_map=None, start_distance=100.0
        )
        with pytest.raises(PatchError, match="'channel' coordinate"):
            patch.enrich(affine, attrs=False, coords=("x",))

    def test_affine_with_channel_coord(self, patch, inventory):
        """Given channel numbers, the affine form resolves them."""
        affine = _replace_acquisition(
            inventory, distance_map=None, start_distance=100.0
        )
        channels = np.arange(len(patch.get_coord("distance")))
        with_channel = patch.update_coords(channel=("distance", channels))
        out = with_channel.enrich(affine, attrs=False, coords=("zone",))
        assert out.get_coord("zone").values[0] == "north"

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
        assert values[0] == "soil" and values[-1] is None

    def test_component_field(self, patch, inventory):
        """Component intervals come from the cumulative layout."""
        out = patch.enrich(inventory, attrs=False, coords=("optical_components.name",))
        assert out.get_coord("optical_components.name").values[0] == "cable"

    def test_coordinate_alias(self, patch, inventory):
        """A label the CRS defines resolves to its canonical axis."""
        out = patch.enrich(inventory, attrs=False, coords=("latitude", "longitude"))
        assert out.get_coord("longitude").values[0] == -117.0

    def test_unknown_alias_raises(self, patch, inventory):
        """A label this CRS does not define has no axis to resolve to."""
        with pytest.raises(InvalidInventoryError, match="not defined by this CRS"):
            patch.enrich(inventory, attrs=False, coords=("northing",))

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

    def test_missing_coord_nan(self, patch, inventory):
        """It can instead be filled with the missing marker."""
        out = patch.enrich(inventory, attrs=False, coords=("nope",), on_missing="nan")
        assert np.isnan(out.get_coord("nope").values).all()

    def test_missing_coord_skip(self, patch, inventory):
        """Or omitted entirely."""
        out = patch.enrich(inventory, attrs=False, coords=("nope",), on_missing="skip")
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
        # An all-uncovered object coord degenerates to floats under numpy.
        assert all(x is None or np.isnan(x) for x in values)


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
        context = inventory.resolve(patch.attrs.data_source_id, time=None)
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
            inventory, distance_map=None, start_distance=100.0
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
    """The spool carries the inventory so extraction enriches."""

    def test_getitem_is_enriched(self, patch, inventory):
        """A patch pulled by index arrives with its metadata."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert spool[0].attrs.gauge_length == 10.0

    def test_iteration_is_enriched(self, patch, inventory):
        """So does one pulled by iteration."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert all(x.attrs.gauge_length == 10.0 for x in spool)

    def test_kwargs_pass_through(self, patch, inventory):
        """Enrich's arguments are the spool's arguments."""
        spool = dc.spool(patch).attach_inventory(
            inventory, attrs=("gauge_length",), coords=False
        )
        out = spool[0]
        assert out.attrs.gauge_length == 10.0
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)

    def test_enrich_false_holds_only(self, patch, inventory):
        """Attaching without enriching just carries the inventory."""
        spool = dc.spool(patch).attach_inventory(inventory, enrich=False)
        assert "gauge_length" not in dict(spool[0].attrs)

    def test_derived_spool_keeps_inventory(self, patch, inventory):
        """A selection is still the spool it came from."""
        spool = dc.spool(patch).attach_inventory(inventory)
        assert spool.select(tag="random")[0].attrs.gauge_length == 10.0

    def test_equality_includes_inventory(self, patch, inventory):
        """An attached inventory changes what the spool yields."""
        plain = dc.spool(patch)
        attached = plain.attach_inventory(inventory)
        assert attached != plain
        assert attached == plain.attach_inventory(inventory)
        assert attached != plain.attach_inventory(inventory, coords=False)

    def test_requires_an_inventory(self, patch):
        """Anything else names no metadata to attach."""
        with pytest.raises(ParameterError, match="needs an Inventory"):
            dc.spool(patch).attach_inventory("inventory.yaml")
