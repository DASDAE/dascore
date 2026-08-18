"""Tests for the DASDAE inventory model."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import dascore as dc
from dascore.constants import INVENTORY_ATTRS
from dascore.core import Inventory
from dascore.core import inventory as inv
from dascore.exceptions import InvalidInventoryError
from dascore.models import InventoryModel, values_equal
from dascore.utils.mapping import FrozenDict
from dascore.utils.namespace import InventoryNameSpace


def build_inventory() -> inv.Inventory:
    """Build a small, valid inventory used across tests."""
    acquisition = inv.Acquisition(
        code="RAW",
        location_code="00",
        start_time="2026-06-01",
        data_category="DAS",
        data_type="strain_rate",
        sample_rate=500.0,
        gauge_length=10.0,
        spatial_interval=1.0,
        distance_map=inv.DistanceMap(channel=(0.0,), distance=(0.0,)),
    )
    geometry = inv.Geometry(
        name="survey",
        distance=(0.0, 100.0, 200.0),
        coordinates={"x": (0.0, 1.0, 2.0), "y": (0.0, 0.0, 0.0), "z": (0.0, 0.0, 1.0)},
    )
    path = inv.OpticalPath(
        name="main",
        location_code="00",
        start_time="2026-06-01",
        optical_components=(inv.FiberSegment(name="fiber", optical_length=250.0),),
        geometry=(geometry,),
        coupling=(
            inv.CouplingCondition(
                start_distance=0.0, end_distance=200.0, coupling_type="trench"
            ),
        ),
        labels=(
            inv.OpticalPathLabel(
                start_distance=0.0, end_distance=100.0, group="zone", value="east"
            ),
        ),
    )
    array = inv.FiberArray(
        code="L001",
        start_time="2026-06-01",
        acquisitions=(acquisition,),
        optical_paths=(path,),
    )
    network = inv.Network(code="DAS", fiber_arrays=(array,))
    return inv.Inventory(networks=(network,))


def build_full_inventory() -> inv.Inventory:
    """Build a valid inventory holding one of every inventory model."""
    cable = inv.Cable(resource_id="cable-1", name="trunk", fiber_count=12)
    enclosure = inv.Enclosure(resource_id="box-1", enclosure_type="splice_box")
    interrogator = inv.Interrogator(resource_id="int-1", model="FI-1")
    report = inv.ExternalResource(resource_id="otdr-file", uri="file://otdr.sor")
    measurement = inv.OpticalMeasurement(
        resource_id="otdr-1", method="otdr", wavelength=1550.0, data=report
    )
    path = inv.OpticalPath(
        name="main",
        location_code="00",
        start_time="2026-06-01",
        optical_components=(
            inv.FiberSegment(
                name="lead",
                optical_length=100.0,
                container=cable,
                fiber_number=1,
                loss_db=0.2,
                loss_measurement=measurement,
            ),
            inv.Connector(name="patch", container=enclosure),
            inv.Splice(name="splice", container=enclosure),
            inv.FiberSegment(name="run", optical_length=400.0, container=cable),
            inv.Terminator(name="end", container=enclosure),
        ),
        geometry=(
            inv.Geometry(
                name="trench",
                distance=(100.0, 400.0),
                coordinates={
                    "x": (-117.0, -117.0),
                    "y": (40.0, 40.1),
                    "z": (1500.0, 1500.0),
                },
            ),
        ),
        coupling=(
            inv.CouplingCondition(
                start_distance=100.0,
                end_distance=250.0,
                coupling_type="trench",
                medium="soil",
                depth=1.0,
            ),
        ),
        labels=(
            inv.OpticalPathLabel(
                start_distance=100.0, end_distance=200.0, group="zone", value="north"
            ),
            inv.OpticalPathLabel(
                start_distance=150.0, end_distance=300.0, group="noisy", value=True
            ),
            inv.OpticalPathLabel(
                start_distance=150.0, end_distance=300.0, group="quiet", value=False
            ),
            inv.OpticalPathLabel(
                start_distance=100.0, end_distance=200.0, group="count", value=0
            ),
            inv.OpticalPathLabel(
                start_distance=200.0, end_distance=300.0, group="offset", value=0.0
            ),
        ),
        measurements=(measurement,),
    )
    acquisition = inv.Acquisition(
        code="RAW",
        location_code="00",
        start_time="2026-06-01",
        data_category="DAS",
        data_type="strain_rate",
        data_units="strain/s",
        sample_rate=500.0,
        gauge_length=10.0,
        spatial_interval=1.0,
        interrogator=interrogator,
        interrogator_port="1",
        extra_fields={"native_key": "", "count": 0},
        distance_map=inv.DistanceMap(channel=(0.0, 300.0), distance=(100.0, 400.0)),
    )
    channel = inv.Channel(
        code="HSF",
        location_code="00",
        coordinates=(-117.0, 40.0, 1500.0),
        sample_rate=500.0,
        response=inv.Response(sensitivity=1.0, input_units="m/s", output_units="V"),
    )
    station = inv.Station(
        code="STA1",
        name="station one",
        identifiers=("doi:10.7914/SN/XX",),
        coordinates=(-117.0, 40.0, 1500.0),
        channels=(channel,),
    )
    array = inv.FiberArray(
        code="R2D1",
        name="array one",
        start_time="2026-06-01",
        acquisitions=(acquisition,),
        optical_paths=(path,),
    )
    network = inv.Network(
        code="DAS",
        name="network one",
        start_time="2026-06-01",
        fiber_arrays=(array,),
        stations=(station,),
    )
    return inv.Inventory(
        creation_info=inv.CreationInfo(
            agency_id="agency", author="author", version="1.0"
        ),
        networks=(network,),
    ).check()


class TestGeometryColumns:
    """A geometry states named numeric columns, of which some are axes."""

    @staticmethod
    def _inventory(*geometry, crs=None, labels=()):
        """Wrap geometry segments in the smallest inventory holding them."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=1000.0),),
            geometry=geometry,
            labels=labels,
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        return inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),),
            **({"coordinate_reference_system": crs} if crs else {}),
        )

    def test_a_column_which_is_not_a_position(self):
        """Chainage is a curve along the fiber and no part of a position."""
        chainage = inv.Geometry(
            name="chainage",
            distance=(0.0, 100.0),
            coordinates={"chainage": (1200.0, 1300.0)},
            units={"chainage": "m"},
        )
        assert self._inventory(chainage).check() is not None
        out = chainage.interpolate([50.0])
        assert out["chainage"][0] == 1250.0

    def test_a_segment_with_no_axes_is_legal(self):
        """A path may be described without ever being placed in space."""
        depth = inv.Geometry(
            distance=(0.0, 40.0), coordinates={"borehole_depth": (0.0, 40.0)}
        )
        inventory = self._inventory(depth)
        assert inventory.check() is inventory
        crs = inventory.coordinate_reference_system
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        assert np.all(np.isnan(path.coordinates_at([20.0], crs)))

    def test_axes_are_all_or_none(self):
        """Half a position is not a position."""
        partial = inv.Geometry(
            name="partial", distance=(0.0, 10.0), coordinates={"x": (0.0, 1.0)}
        )
        with pytest.raises(InvalidInventoryError, match="every axis or none"):
            self._inventory(partial).check()

    def test_an_axis_stated_twice(self):
        """`x` and the label the CRS gives it are one axis, not two."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("easting", "northing"), units=("meter", "meter")
        )
        doubled = inv.Geometry(
            name="doubled",
            distance=(0.0, 10.0),
            coordinates={
                "x": (0.0, 1.0),
                "easting": (0.0, 1.0),
                "northing": (0.0, 1.0),
            },
        )
        with pytest.raises(InvalidInventoryError, match="twice"):
            self._inventory(doubled, crs=crs).check()

    def test_overlap_is_refused_per_column(self):
        """Two segments may overlap unless they state the same column."""
        first = inv.Geometry(distance=(0.0, 60.0), coordinates={"depth": (0.0, 6.0)})
        second = inv.Geometry(distance=(50.0, 80.0), coordinates={"depth": (5.0, 8.0)})
        with pytest.raises(InvalidInventoryError, match="for column 'depth'"):
            self._inventory(first, second).check()

    def test_different_columns_may_overlap(self):
        """Each column is its own function track, so they are independent."""
        depth = inv.Geometry(
            name="hole 1", distance=(0.0, 60.0), coordinates={"depth": (0.0, 6.0)}
        )
        azimuth = inv.Geometry(
            name="hole 1", distance=(50.0, 80.0), coordinates={"azimuth": (5.0, 8.0)}
        )
        inventory = self._inventory(depth, azimuth)
        assert inventory.check() is inventory

    def test_overlapping_segments_share_a_name(self):
        """The bare `geometry` coordinate is that name, so there is one."""
        depth = inv.Geometry(
            name="depth survey",
            distance=(0.0, 60.0),
            coordinates={"depth": (0.0, 6.0)},
        )
        azimuth = inv.Geometry(
            name="azimuth survey",
            distance=(50.0, 80.0),
            coordinates={"azimuth": (5.0, 8.0)},
        )
        with pytest.raises(InvalidInventoryError, match="share its name"):
            self._inventory(depth, azimuth).check()

    def test_a_reserved_column_name(self):
        """A column becomes a coordinate, so it cannot shadow one."""
        clash = inv.Geometry(distance=(0.0, 10.0), coordinates={"time": (0.0, 1.0)})
        with pytest.raises(InvalidInventoryError, match="reserved name"):
            self._inventory(clash).check()

    def test_a_column_which_is_also_a_label_group(self):
        """One name is one coordinate, whichever track would define it."""
        column = inv.Geometry(distance=(0.0, 10.0), coordinates={"zone": (0.0, 1.0)})
        label = inv.OpticalPathLabel(
            start_distance=0.0, end_distance=10.0, group="zone", value=1.0
        )
        with pytest.raises(InvalidInventoryError, match="one name is one coordinate"):
            self._inventory(column, labels=(label,)).check()

    def test_units_on_an_axis_are_refused(self):
        """The CRS states the units of its own axes."""
        segment = inv.Geometry(
            name="axed",
            distance=(0.0, 10.0),
            coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
            units={"x": "furlong"},
        )
        with pytest.raises(InvalidInventoryError, match="states the units of its own"):
            self._inventory(segment).check()

    def test_units_for_a_column_which_is_not_there(self):
        """A unit sitting on nothing is a typo no reader would find."""
        with pytest.raises(ValidationError, match="has no column"):
            inv.Geometry(
                distance=(0.0, 10.0),
                coordinates={"depth": (0.0, 1.0)},
                units={"dpeth": "m"},
            )

    def test_one_unit_per_column(self):
        """Two segments cannot measure one column in two units."""
        meters = inv.Geometry(
            distance=(0.0, 10.0),
            coordinates={"depth": (0.0, 1.0)},
            units={"depth": "m"},
        )
        feet = inv.Geometry(
            distance=(20.0, 30.0),
            coordinates={"depth": (0.0, 1.0)},
            units={"depth": "ft"},
        )
        with pytest.raises(InvalidInventoryError, match="has one unit"):
            self._inventory(meters, feet).check()

    def test_a_crs_label_the_crs_does_not_declare(self):
        """`depth` is a column of its own where the CRS spends z on elevation."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("easting", "northing", "elevation"),
            units=("meter", "meter", "meter"),
        )
        segment = inv.Geometry(
            distance=(0.0, 40.0),
            coordinates={"depth": (0.0, 40.0)},
            units={"depth": "m"},
        )
        inventory = self._inventory(segment, crs=crs)
        assert inventory.check() is inventory
        assert "depth" in inventory.get_names().coords

    def test_a_column_never_bridges_two_segments(self):
        """Distance between two segments is uncovered, whatever they hold."""
        first = inv.Geometry(distance=(0.0, 10.0), coordinates={"depth": (0.0, 1.0)})
        second = inv.Geometry(distance=(20.0, 30.0), coordinates={"depth": (2.0, 3.0)})
        path = self._inventory(first, second).networks[0].fiber_arrays[0]
        values = path.optical_paths[0].column_at("depth", [5.0, 15.0, 25.0])
        assert not np.isnan(values[0]) and not np.isnan(values[2])
        assert np.isnan(values[1])

    def test_a_column_no_segment_states(self):
        """None, so the caller's on_missing policy rules rather than a nan."""
        segment = inv.Geometry(distance=(0.0, 10.0), coordinates={"depth": (0.0, 1.0)})
        path = self._inventory(segment).networks[0].fiber_arrays[0].optical_paths[0]
        assert path.column_at("azimuth", [5.0]) is None


class TestGeometryColumnReviewFindings:
    """What a review of the column form found, kept as regressions."""

    @staticmethod
    def _path(*geometry):
        return inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=geometry,
        )

    def _inventory(self, *geometry, crs=None):
        array = inv.FiberArray(code="L001", optical_paths=(self._path(*geometry),))
        return inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),),
            **({"coordinate_reference_system": crs} if crs else {}),
        )

    def test_two_spellings_of_one_axis_overlap(self):
        """Checking columns by name alone would let these two through."""
        canonical = inv.Geometry(
            name="a",
            distance=(0.0, 10.0),
            coordinates={"x": (0.0, 10.0), "y": (0.0, 10.0), "z": (0.0, 10.0)},
        )
        labelled = inv.Geometry(
            name="b",
            distance=(5.0, 15.0),
            coordinates={
                "longitude": (100.0, 110.0),
                "latitude": (100.0, 110.0),
                "elevation": (100.0, 110.0),
            },
        )
        with pytest.raises(InvalidInventoryError, match="for axis"):
            self._inventory(canonical, labelled).check()

    def test_an_axis_stated_twice_is_refused_when_placing(self):
        """Otherwise whichever spelling came last would win, silently."""
        doubled = inv.Geometry(
            name="doubled",
            distance=(0.0, 10.0),
            coordinates={
                "x": (0.0, 10.0),
                "longitude": (100.0, 110.0),
                "y": (0.0, 1.0),
                "z": (0.0, 1.0),
            },
        )
        crs = inv.CoordinateReferenceSystem()
        with pytest.raises(InvalidInventoryError, match="twice"):
            self._path(doubled).coordinates_at([5.0], crs)

    def test_a_column_segment_does_not_claim_a_position_run_end(self):
        """The position track's coverage is the segments which place it."""
        placed = inv.Geometry(
            name="placed",
            distance=(0.0, 10.0),
            coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
        )
        measured = inv.Geometry(
            name="measured",
            distance=(10.0, 20.0),
            coordinates={"borehole_depth": (0.0, 10.0)},
        )
        crs = inv.CoordinateReferenceSystem()
        out = self._path(placed, measured).coordinates_at([10.0], crs)
        assert np.allclose(out, [[1.0, 1.0, 1.0]])

    def test_a_dotted_column_name(self):
        """A dotted name is how a field of a typed track is asked for."""
        dotted = inv.Geometry(
            distance=(0.0, 10.0), coordinates={"coupling.depth": (0.0, 1.0)}
        )
        with pytest.raises(InvalidInventoryError, match="dotted name"):
            self._inventory(dotted).check()

    def test_select_revalidates_the_columns_it_writes(self):
        """model_copy would skip the validators and leave a mutable dict."""
        segment = inv.Geometry(
            name="run",
            distance=(0.0, 10.0),
            coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
        )
        out = self._path(segment).select(distance=(2.0, 8.0))
        coordinates = out.geometry[0].coordinates
        assert isinstance(coordinates, Mapping)
        with pytest.raises(TypeError):
            coordinates["x"] = (0.0,)

    def test_a_canonical_name_the_crs_has_no_axis_for(self):
        """`z` is a column of its own where the CRS declares two axes."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("easting", "northing"), units=("meter", "meter")
        )
        segment = inv.Geometry(
            distance=(0.0, 10.0), coordinates={"z": (0.0, 1.0)}, units={"z": "m"}
        )
        inventory = self._inventory(segment, crs=crs)
        assert inventory.check() is inventory
        assert "z" in inventory.get_names().coords


class TestGeometry:
    """Geometry segment rules."""

    def test_requires_two_points(self):
        """Requires two points."""
        with pytest.raises(ValidationError, match="at least 2 control points"):
            inv.Geometry(distance=(1.0,), coordinates={"x": (0.0,), "y": (0.0,)})

    def test_strictly_increasing(self):
        """Strictly increasing."""
        with pytest.raises(ValidationError, match="strictly increasing"):
            inv.Geometry(
                distance=(1.0, 1.0), coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)}
            )

    def test_paired_lengths(self):
        """Paired lengths."""
        with pytest.raises(ValidationError, match="one value per distance"):
            inv.Geometry(distance=(0.0, 1.0), coordinates={"x": (0.0,), "y": (0.0,)})

    def test_coil_repeated_coordinates(self):
        """A coil interpolates to a constant coordinate."""
        coil = inv.Geometry(
            distance=(1200.0, 1300.0),
            coordinates={"x": (500.0, 500.0), "y": (120.0, 120.0)},
        )
        out = coil.interpolate([1200.0, 1250.0, 1299.0])
        assert np.allclose(out["x"], [500.0] * 3)
        assert np.allclose(out["y"], [120.0] * 3)

    def test_uncovered_is_nan(self):
        """Uncovered is nan."""
        geo = inv.Geometry(
            distance=(10.0, 20.0), coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)}
        )
        out = geo.interpolate([5.0, 25.0])
        assert all(np.all(np.isnan(x)) for x in out.values())


class TestPathTracks:
    """Track-kind rules: tiling, function tracks, set track."""

    def test_partial_coverage_is_legal(self):
        """Partial coverage is legal."""
        path = build_inventory().networks[0].fiber_arrays[0].optical_paths[0]
        assert path.check() is path

    def test_coupling_overlap_raises(self):
        """Coupling overlap raises."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    start_distance=0.0, end_distance=60.0, coupling_type="trench"
                ),
                inv.CouplingCondition(
                    start_distance=50.0, end_distance=70.0, coupling_type="conduit"
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Overlapping coupling"):
            path.check()

    def test_geometry_overlap_raises(self):
        """Geometry overlap raises."""
        seg = dict(coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)})
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(distance=(0.0, 60.0), **seg),
                inv.Geometry(distance=(50.0, 80.0), **seg),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Overlapping geometry"):
            path.check()

    def test_boolean_labels_overlap_freely(self):
        """Membership labels overlap, within and across groups."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=0.0, end_distance=60.0, group="noisy"
                ),
                inv.OpticalPathLabel(
                    start_distance=50.0, end_distance=70.0, group="noisy"
                ),
                inv.OpticalPathLabel(
                    start_distance=40.0, end_distance=80.0, group="repaired"
                ),
            ),
        )
        assert path.check() is path

    def test_valued_label_groups_may_not_overlap(self):
        """A single-valued group cannot claim two values at one distance."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=0.0,
                    end_distance=60.0,
                    group="rock_type",
                    value="granite",
                ),
                inv.OpticalPathLabel(
                    start_distance=50.0,
                    end_distance=70.0,
                    group="rock_type",
                    value="shale",
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="only boolean groups"):
            path.check()

    def test_label_group_holds_one_kind_of_value(self):
        """Mixing value kinds in one group is a modeling error."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=0.0, end_distance=10.0, group="zone", value="east"
                ),
                inv.OpticalPathLabel(
                    start_distance=20.0, end_distance=30.0, group="zone", value=True
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="one kind of value"):
            path.check()

    def test_numeric_label_group(self):
        """Numeric groups are single valued but otherwise ordinary."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=0.0,
                    end_distance=40.0,
                    group="frost_depth",
                    value=1.2,
                ),
                inv.OpticalPathLabel(
                    start_distance=40.0,
                    end_distance=90.0,
                    group="frost_depth",
                    value=0.8,
                ),
            ),
        )
        assert path.check() is path

    def test_out_of_bounds_raises(self):
        """Out of bounds raises."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    start_distance=90.0, end_distance=150.0, coupling_type="trench"
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="extends past"):
            path.check()

    def test_outer_endpoint_included(self):
        """The outermost covered endpoint of the geometry track resolves."""
        inventory = build_inventory()
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        crs = inventory.coordinate_reference_system
        coords = path.coordinates_at([200.0], crs)
        assert np.allclose(coords, [[2.0, 0.0, 1.0]])

    def test_uncovered_distance_is_nan(self):
        """Uncovered distance is nan."""
        inventory = build_inventory()
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        crs = inventory.coordinate_reference_system
        assert np.all(np.isnan(path.coordinates_at([225.0], crs)))


class TestDistanceMap:
    """DistanceMap rules and mapping behavior."""

    def test_at_least_one_input_axis(self):
        """A map with no input axis maps nothing."""
        with pytest.raises(ValidationError, match="at least one input axis"):
            inv.DistanceMap(distance=(5.0,))

    def test_two_input_axes(self):
        """
        One set of control points may be stated in both coordinates.

        The same calibration then serves patches whose axes differ, without
        the two descriptions being able to disagree.
        """
        dmap = inv.DistanceMap(
            channel=(0.0, 100.0),
            instrument_distance=(10.0, 112.0),
            distance=(500.0, 600.0),
        )
        assert dmap.axes == ("channel", "instrument_distance")
        assert np.isclose(dmap.map_to_distance([50], axis="channel")[0], 550.0)
        out = dmap.map_to_distance([61], axis="instrument_distance")
        assert np.isclose(out[0], 550.0)

    def test_axis_lengths_must_match(self):
        """Every axis states the same control points."""
        with pytest.raises(ValidationError, match="same"):
            inv.DistanceMap(
                channel=(0.0, 1.0), instrument_distance=(0.0,), distance=(5.0, 6.0)
            )

    def test_nonfinite_distance_raises(self):
        """A control point at NaN places nothing."""
        with pytest.raises(ValidationError, match="must be finite"):
            inv.DistanceMap(channel=(0.0, 1.0), distance=(5.0, np.inf))

    def test_unwritten_axis_raises(self):
        """A map cannot be read on an axis it was not written in."""
        dmap = inv.DistanceMap(channel=(0.0, 1.0), distance=(5.0, 6.0))
        with pytest.raises(InvalidInventoryError, match="not written in"):
            dmap.map_to_distance([0.5], axis="instrument_distance")

    def test_two_point_map(self):
        """Two point map."""
        dmap = inv.DistanceMap(channel=(512.0, 1710.0), distance=(500.0, 1698.0))
        out = dmap.map_to_distance([512, 1111, 1710])
        assert np.isclose(out[0], 500.0)
        assert np.isclose(out[-1], 1698.0)

    def test_outside_coverage_is_nan(self):
        """Outside coverage is nan."""
        dmap = inv.DistanceMap(channel=(512.0, 1710.0), distance=(500.0, 1698.0))
        assert np.isnan(dmap.map_to_distance([2000.0])[0])

    def test_single_point_needs_slope(self):
        """Single point needs slope."""
        dmap = inv.DistanceMap(channel=(512.0,), distance=(500.0,))
        with pytest.raises(InvalidInventoryError, match="slope"):
            dmap.map_to_distance([600.0])
        out = dmap.map_to_distance([612.0], slope=1.0)
        assert np.isclose(out[0], 600.0)

    def test_instrument_distance_axis(self):
        """Instrument distance axis."""
        dmap = inv.DistanceMap(
            instrument_distance=(0.0, 2000.0), distance=(-12.4, 1985.2)
        )
        out = dmap.map_to_distance([0.0])
        assert np.isclose(out[0], -12.4)


class TestAcquisition:
    """Acquisition resolution mechanisms and code rules."""

    def test_single_point_channel_map_is_affine(self):
        """One point states an origin; spatial_interval is the slope."""
        dmap = inv.DistanceMap(channel=(0.0,), distance=(10.0,))
        acq = inv.Acquisition(code="RAW", spatial_interval=2.0, distance_map=dmap)
        assert np.allclose(acq.channel_to_distance([0, 5]), [10.0, 20.0])

    def test_single_point_instrument_map_is_an_offset(self):
        """Interrogator meters map onto path meters one for one."""
        dmap = inv.DistanceMap(instrument_distance=(-120.5,), distance=(0.0,))
        acq = inv.Acquisition(code="RAW", spatial_interval=2.0, distance_map=dmap)
        assert np.allclose(acq.channel_to_distance([-120.5, -20.5]), [0.0, 100.0])

    def test_map_mapping_used_when_present(self):
        """Map mapping used when present."""
        dmap = inv.DistanceMap(channel=(0.0, 10.0), distance=(100.0, 120.0))
        acq = inv.Acquisition(code="RAW", distance_map=dmap)
        assert np.isclose(acq.channel_to_distance([5])[0], 110.0)

    def test_no_map_raises(self):
        """An acquisition with no map places no channels."""
        acq = inv.Acquisition(code="RAW")
        with pytest.raises(InvalidInventoryError, match="no distance_map"):
            acq.channel_to_distance([0])

    def test_code_charset(self):
        """Code charset."""
        with pytest.raises(ValidationError, match="Invalid"):
            inv.Acquisition(code="MY_RAW")
        with pytest.raises(ValidationError, match="Invalid"):
            inv.Acquisition(code="")

    def test_blank_location_legal(self):
        """Blank location legal."""
        acq = inv.Acquisition(code="RAW", location_code="")
        assert acq.location_code == ""


class TestEpochContainment:
    """An entry may not describe time its container does not."""

    @staticmethod
    def _array(array_kwargs, path_kwargs):
        """A fiber array holding one optical path, each with its own epoch."""
        path = inv.OpticalPath(
            name="main",
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            **path_kwargs,
        )
        return inv.FiberArray(code="L001", optical_paths=(path,), **array_kwargs)

    def test_a_path_starting_before_its_array(self):
        """The array did not exist then, so nothing can resolve to the path."""
        array = self._array({"start_time": "2024-01-01"}, {"start_time": "2020-01-01"})
        with pytest.raises(InvalidInventoryError, match="outside its container"):
            array.check()

    def test_a_path_ending_after_its_array(self):
        """The far bound is checked the same way as the near one."""
        array = self._array(
            {"start_time": "2024-01-01", "end_time": "2025-01-01"},
            {"start_time": "2024-06-01", "end_time": "2030-01-01"},
        )
        with pytest.raises(InvalidInventoryError, match="outside its container"):
            array.check()

    def test_a_path_entirely_after_its_array(self):
        """Neither bound is out of order, and the whole epoch is elsewhere."""
        array = self._array(
            {"start_time": "2024-01-01", "end_time": "2025-01-01"},
            {"start_time": "2026-01-01"},
        )
        with pytest.raises(InvalidInventoryError, match="outside its container"):
            array.check()

    def test_an_unset_bound_defers_to_the_container(self):
        """A path stating no start begins when its fiber array does."""
        array = self._array({"start_time": "2024-01-01"}, {})
        assert array.check() is array

    def test_a_contained_epoch_is_legal(self):
        """The ordinary case: a path living inside its array's lifetime."""
        array = self._array(
            {"start_time": "2024-01-01", "end_time": "2026-01-01"},
            {"start_time": "2024-06-01", "end_time": "2025-01-01"},
        )
        assert array.check() is array

    def test_an_acquisition_outside_its_array(self):
        """Acquisitions are contained by the same rule as paths."""
        array = inv.FiberArray(
            code="L001",
            start_time="2024-01-01",
            acquisitions=(inv.Acquisition(code="RAW", start_time="2020-01-01"),),
        )
        with pytest.raises(InvalidInventoryError, match="outside its container"):
            array.check()

    def test_a_fiber_array_outside_its_network(self):
        """And the rule holds one level up, for what a network contains."""
        array = inv.FiberArray(code="L001", start_time="2020-01-01")
        network = inv.Network(code="XX", start_time="2024-01-01", fiber_arrays=(array,))
        with pytest.raises(InvalidInventoryError, match="outside its container"):
            network.check()

    def test_a_channel_outside_its_station(self):
        """Stations contain channels the same way."""
        station = inv.Station(
            code="STA1",
            start_time="2024-01-01",
            channels=(inv.Channel(code="BHZ", start_time="2020-01-01"),),
        )
        with pytest.raises(InvalidInventoryError, match="outside its container"):
            station.check()

    def test_the_error_names_both_epochs(self):
        """A reader has to see which range reaches outside which."""
        array = self._array({"start_time": "2024-01-01"}, {"start_time": "2020-01-01"})
        with pytest.raises(InvalidInventoryError) as info:
            array.check()
        message = str(info.value)
        assert "2020-01-01" in message and "2024-01-01" in message
        assert "'main'" in message


class TestEpochs:
    """Half-open epoch rules across the tree."""

    def test_concurrent_paths_same_location_raise(self):
        """Concurrent paths same location raise."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(name="a", start_time="2020-01-01"),
                inv.OpticalPath(name="b", start_time="2021-01-01"),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="overlap in time"):
            array.check()

    def test_concurrent_paths_different_locations_legal(self):
        """Location-scoped paths: similar co-located fibers share an array."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(name="a", location_code="00", start_time="2020-01-01"),
                inv.OpticalPath(name="b", location_code="01", start_time="2020-01-01"),
            ),
        )
        assert array.check() is array

    def test_abutting_epochs_legal(self):
        """Half-open handoff: end == next start is legal."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(
                    name="a", start_time="2020-01-01", end_time="2021-01-01"
                ),
                inv.OpticalPath(name="b", start_time="2021-01-01"),
            ),
        )
        assert array.check() is array

    def test_acquisition_epoch_overlap_raises(self):
        """Acquisition epoch overlap raises."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(
                inv.Acquisition(code="RAW", start_time="2020-01-01"),
                inv.Acquisition(code="RAW", start_time="2020-06-01"),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="overlap in time"):
            array.check()

    def test_acquisition_gap_legal(self):
        """Acquisition gap legal."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(
                inv.Acquisition(
                    code="RAW", start_time="2020-01-01", end_time="2020-06-01"
                ),
                inv.Acquisition(code="RAW", start_time="2020-07-15"),
            ),
        )
        assert array.check() is array

    def test_station_fiber_code_collision_raises(self):
        """Station fiber code collision raises."""
        network = inv.Network(
            code="DAS",
            fiber_arrays=(inv.FiberArray(code="L001", start_time="2020-01-01"),),
            stations=(inv.Station(code="L001", start_time="2020-06-01"),),
        )
        with pytest.raises(InvalidInventoryError, match="share code"):
            network.check()


class TestResolution:
    """acquisition_key + time resolution."""

    def test_happy_path(self):
        """Happy path."""
        inventory = build_inventory()
        ctx = inventory.resolve("DAS.L001.00.RAW", time="2026-07-01")
        assert ctx.acquisition.code == "RAW"
        assert ctx.optical_path.name == "main"
        assert ctx.fiber_array.code == "L001"

    def test_time_selects_epoch(self):
        """Time selects epoch."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(
                inv.Acquisition(
                    code="RAW",
                    start_time="2020-01-01",
                    end_time="2021-01-01",
                    gauge_length=10.0,
                ),
                inv.Acquisition(code="RAW", start_time="2021-01-01", gauge_length=20.0),
            ),
        )
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        early = inventory.resolve("DAS.L001..RAW", time="2020-06-01")
        late = inventory.resolve("DAS.L001..RAW", time="2021-01-01")
        assert early.acquisition.gauge_length == 10.0
        # The boundary instant belongs to the newer epoch (half-open).
        assert late.acquisition.gauge_length == 20.0

    def test_blank_location_id(self):
        """DAS.L001..RAW is a legal identifier with a blank location."""
        array = inv.FiberArray(code="L001", acquisitions=(inv.Acquisition(code="RAW"),))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        ctx = inventory.resolve("DAS.L001..RAW")
        assert ctx.acquisition.location_code == ""

    def test_missing_raises(self):
        """Missing raises."""
        inventory = build_inventory()
        with pytest.raises(InvalidInventoryError, match="0 acquisitions"):
            inventory.resolve("DAS.L001.00.NOPE")

    def test_wrong_shape_raises(self):
        """Wrong shape raises."""
        with pytest.raises(InvalidInventoryError, match="dot separated codes"):
            build_inventory().resolve("DAS.L001.RAW")


class TestPathOperations:
    """select / split_at / reverse / concatenation."""

    @pytest.fixture()
    def path(self):
        """Return the example path."""
        return build_inventory().networks[0].fiber_arrays[0].optical_paths[0]

    def test_component_intervals(self, path):
        """Components tile the absolute axis cumulatively."""
        two = inv.OpticalPath(
            start_distance=100.0,
            optical_components=(
                inv.FiberSegment(optical_length=50.0),
                inv.Splice(optical_length=0.5),
            ),
        )
        assert two.component_intervals() == ((100.0, 150.0), (150.0, 150.5))

    def test_select_preserves_absolute_distances(self, path):
        """Select preserves absolute distances."""
        piece = path.select(distance=(50.0, 150.0))
        assert piece.start_distance == 50.0
        assert np.isclose(piece.optical_length, 100.0)
        assert piece.geometry[0].distance[0] == 50.0
        piece.check()

    def test_split_and_rejoin(self, path):
        """Split and rejoin."""
        left, right = path.split_at(100.0)
        assert left.end_distance == right.start_distance == 100.0
        joined = left + right
        assert np.isclose(joined.optical_length, path.optical_length)
        joined.check()

    def test_reverse_involution(self, path):
        """Reverse involution."""
        twice = path.reverse().reverse()
        assert twice.model_dump(mode="json") == path.model_dump(mode="json")

    def test_reverse_rewrites_all_tracks(self, path):
        """Reverse rewrites all tracks."""
        rev = path.reverse()
        # 0-100 label on a 0-250 path becomes 150-250.
        assert rev.labels[0].interval == (150.0, 250.0)
        # 0-200 coupling becomes 50-250.
        assert rev.coupling[0].interval == (50.0, 250.0)
        rev.check()

    def test_empty_selection_raises(self, path):
        """Empty selection raises."""
        with pytest.raises(Exception, match="Empty distance selection"):
            path.select(distance=(300.0, 400.0))


class TestInventory:
    """Whole-inventory behaviors."""

    def test_validate(self):
        """Validate."""
        assert build_inventory().check() is not None

    def test_duplicate_network_codes_raise(self):
        """Two networks sharing a code and a time cannot be told apart."""
        net = inv.Network(code="DAS")
        with pytest.raises(InvalidInventoryError, match="Duplicate network code"):
            inv.Inventory(networks=(net, net)).check()

    def test_yaml_roundtrip(self, tmp_path):
        """Yaml roundtrip."""
        inventory = build_inventory()
        path = tmp_path / "inventory.yaml"
        inventory.io.to_yaml(path)
        loaded = dc.inventory(path)
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_resources_accept_iterable(self):
        """Resources accept iterable."""
        cable = inv.Cable(resource_id="cable_01")
        inventory = inv.Inventory(resources=[cable])
        assert inventory.resources["cable_01"] == cable

    def test_replace_is_a_correction(self):
        """Replace is a correction."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        new = old.new(gauge_length=12.0)
        updated = inventory.replace(old, new)
        acq = updated.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.gauge_length == 12.0
        # Original is untouched (immutability).
        assert inventory.networks[0].fiber_arrays[0].acquisitions[0] == old

    def test_replace_missing_raises(self):
        """Replace missing raises."""
        inventory = build_inventory()
        with pytest.raises(InvalidInventoryError, match="not found"):
            inventory.replace(inv.Network(code="XX"), inv.Network(code="YY"))

    def test_dc_namespace(self, tmp_path):
        """Dc namespace."""
        assert isinstance(dc.inventory(), inv.Inventory)
        path = tmp_path / "inv.yaml"
        build_inventory().io.to_yaml(path)
        assert isinstance(dc.inventory(path), inv.Inventory)

    def test_crs_vocabulary_enforced(self):
        """Structural column names are outside the coordinate vocabulary."""
        with pytest.raises(ValidationError):
            inv.CoordinateReferenceSystem(coordinate_labels=("x", "distance"))


class TestReviewRegressions:
    """Regression tests for counterpart-review findings."""

    def test_far_future_open_epochs_overlap(self):
        """Ongoing epochs overlap finite epochs beyond any sentinel date."""
        a = inv.OpticalPath(name="a", start_time="2259-01-01")
        b = inv.OpticalPath(name="b", start_time="2261-01-01", end_time="2262-01-01")
        assert a.overlaps(b) and b.overlaps(a)

    def test_reversed_epoch_raises(self):
        """End before start fails at construction."""
        with pytest.raises(ValidationError, match="must be after"):
            inv.Acquisition(code="RAW", start_time="2022-01-01", end_time="2021-01-01")

    def test_replace_type_mismatch_raises(self):
        """Replace type mismatch raises."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        with pytest.raises(InvalidInventoryError, match="does not match"):
            inventory.replace(old, "not-an-acquisition")

    def test_full_select_keeps_terminal_zero_length_component(self):
        """A terminator at the path end survives full selection and splits."""
        path = inv.OpticalPath(
            optical_components=(
                inv.FiberSegment(optical_length=100.0),
                inv.Terminator(optical_length=0.0),
            ),
        )
        full = path.select(distance=(None, None))
        assert len(full.optical_components) == 2
        _, right = path.split_at(50.0)
        assert isinstance(right.optical_components[-1], inv.Terminator)

    def test_resource_key_mismatch_raises(self):
        """Resource key mismatch raises."""
        cable = inv.Cable(resource_id="right")
        with pytest.raises(ValidationError, match="disagrees"):
            inv.Inventory(resources={"wrong": cable})

    def test_duplicate_resource_ids_raise(self):
        """Duplicate resource ids raise."""
        with pytest.raises(ValidationError, match="Duplicate resource_id"):
            inv.Inventory(
                resources=[inv.Cable(resource_id="x"), inv.Cable(resource_id="x")]
            )

    def test_columnless_geometry_raises(self):
        """A segment which measures nothing describes nothing."""
        with pytest.raises(ValidationError, match="states no columns"):
            inv.Geometry(distance=(0.0, 1.0), coordinates={})

    def test_station_extra_fields_forbidden(self):
        """Coordinates are canonical (x, y, z); label fields are not stored."""
        with pytest.raises(ValidationError):
            inv.Station(code="VA01", latitude=48.8)


class TestCanonicalAxes:
    """Canonical (x, y, z) storage with CRS-resolved aliases."""

    def test_canonical_axes_always_resolve(self):
        """Canonical axes always resolve."""
        crs = inv.CoordinateReferenceSystem()
        assert [crs.axis_index(x) for x in ("x", "y", "z")] == [0, 1, 2]

    def test_alias_resolves_when_crs_defines_it(self):
        """Alias resolves when crs defines it."""
        crs = inv.CoordinateReferenceSystem()
        assert crs.axis_index("latitude") == 1
        assert crs.axis_index("longitude") == 0

    def test_alias_not_defined_raises(self):
        """An alias absent from the CRS labels raises."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("easting", "northing", "elevation"),
            units=("meter", "meter", "meter"),
        )
        assert crs.axis_index("easting") == 0
        with pytest.raises(InvalidInventoryError, match="not defined"):
            crs.axis_index("latitude")

    def test_labels_outside_vocabulary_raise(self):
        """Labels outside vocabulary raise."""
        with pytest.raises(ValidationError):
            inv.CoordinateReferenceSystem(coordinate_labels=("depth", "along"))

    def test_two_axis_crs_has_no_z(self):
        """Two axis crs has no z."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("x", "y"), units=("meter", "meter")
        )
        with pytest.raises(InvalidInventoryError, match="no 'z' axis"):
            crs.axis_index("z")


class TestResourcePool:
    """Shareable resources normalize into the flat pool with id references."""

    @staticmethod
    def _inventory_with(component, **acq_kwargs):
        """Wrap a component (and optional acquisition) into an inventory."""
        acq = (inv.Acquisition(code="RAW", **acq_kwargs),) if acq_kwargs else ()
        path = inv.OpticalPath(optical_components=(component,))
        array = inv.FiberArray(code="L001", acquisitions=acq, optical_paths=(path,))
        return inv.Inventory(networks=(inv.Network(code="DAS", fiber_arrays=(array,)),))

    def test_inline_resource_normalizes_to_pool(self):
        """An inline cable moves to the pool; the field keeps its id."""
        cable = inv.Cable(resource_id="cable-01", name="c")
        seg = inv.FiberSegment(optical_length=100.0, container=cable)
        inventory = self._inventory_with(seg)
        stored = (
            inventory.networks[0].fiber_arrays[0].optical_paths[0].optical_components[0]
        )
        assert stored.container == "cable-01"
        assert inventory.get_resource("cable-01") == cable

    def test_shared_resource_registered_once(self):
        """Two components sharing one enclosure yield one pool entry."""
        coupler = inv.Enclosure(resource_id="coupler-01")
        path = inv.OpticalPath(
            optical_components=(
                inv.Connector(container=coupler),
                inv.Connector(container=coupler),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        assert list(inventory.resources) == ["coupler-01"]

    def test_conflicting_inline_definitions_raise(self):
        """One id, two different contents: agree-or-raise."""
        a = inv.Cable(resource_id="cable-01", fiber_count=1)
        b = inv.Cable(resource_id="cable-01", fiber_count=4)
        path = inv.OpticalPath(
            optical_components=(
                inv.FiberSegment(optical_length=1.0, container=a),
                inv.FiberSegment(optical_length=1.0, container=b),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        with pytest.raises(ValidationError, match="defined twice"):
            inv.Inventory(networks=(inv.Network(code="DAS", fiber_arrays=(array,)),))

    def test_dangling_reference_raises(self):
        """Dangling reference raises."""
        seg = inv.FiberSegment(optical_length=1.0, container="no-such-cable")
        with pytest.raises(ValidationError, match="Dangling"):
            self._inventory_with(seg)

    def test_nested_resources_normalize(self):
        """A cable inside a pipe: both land in the pool, linked by id."""
        pipe = inv.Enclosure(resource_id="pipe-01", enclosure_type="pipe")
        cable = inv.Cable(resource_id="cable-01", container=pipe)
        seg = inv.FiberSegment(optical_length=1.0, container=cable)
        inventory = self._inventory_with(seg)
        assert inventory.get_resource("cable-01").container == "pipe-01"
        assert inventory.get_resource("pipe-01") == pipe

    def test_interrogator_normalizes(self):
        """Interrogator normalizes."""
        unit = inv.Interrogator(resource_id="int-01", model="DAS-1000")
        seg = inv.FiberSegment(optical_length=1.0)
        inventory = self._inventory_with(seg, interrogator=unit)
        acq = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.interrogator == "int-01"
        assert inventory.get_resource("int-01") == unit

    def test_resource_correction_is_single_site(self):
        """Replacing a pooled resource touches only the pool."""
        cable = inv.Cable(resource_id="cable-01", fiber_count=1)
        seg = inv.FiberSegment(optical_length=1.0, container=cable)
        inventory = self._inventory_with(seg)
        fixed = cable.new(fiber_count=4)
        updated = inventory.replace(cable, fixed)
        assert updated.get_resource("cable-01").fiber_count == 4
        stored = (
            updated.networks[0].fiber_arrays[0].optical_paths[0].optical_components[0]
        )
        assert stored.container == "cable-01"

    def test_resource_correction_must_keep_id(self):
        """Resource correction must keep id."""
        cable = inv.Cable(resource_id="cable-01")
        seg = inv.FiberSegment(optical_length=1.0, container=cable)
        inventory = self._inventory_with(seg)
        renamed = cable.new(resource_id="cable-02")
        with pytest.raises(InvalidInventoryError, match="same resource_id"):
            inventory.replace(cable, renamed)

    def test_yaml_roundtrip_stays_flat(self, tmp_path):
        """Serialized form holds ids, not inline copies, and round-trips."""
        cable = inv.Cable(resource_id="cable-01", name="c")
        seg = inv.FiberSegment(optical_length=100.0, container=cable)
        inventory = self._inventory_with(seg)
        text = inventory.io.to_yaml()
        assert text.count("cable-01") >= 2
        loaded = inv.Inventory.from_yaml(text)
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_get_resource_missing_raises(self):
        """Get resource missing raises."""
        with pytest.raises(InvalidInventoryError, match="No resource"):
            build_inventory().get_resource("nope")


class TestInternalReviewRegressions:
    """Regression tests for internal adversarial/convention review findings."""

    def test_new_preserves_union_discriminators(self):
        """new() works on models holding discriminated unions."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=10.0),)
        )
        renamed = path.new(name="renamed")
        assert renamed.name == "renamed"
        assert isinstance(renamed.optical_components[0], inv.FiberSegment)

    def test_new_preserves_normalized_pool(self):
        """new() keeps resources that normalization moved into the pool."""
        acq = inv.Acquisition(
            code="RAW", interrogator=inv.Interrogator(resource_id="int-1")
        )
        array = inv.FiberArray(code="L001", acquisitions=(acq,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        updated = inventory.new(resource_id="doc-1")
        assert updated.get_resource("int-1").resource_id == "int-1"

    def test_equality_with_nested_nat(self):
        """Structurally identical trees with unset times compare equal."""

        def make():
            return inv.FiberArray(code="F", acquisitions=(inv.Acquisition(code="A"),))

        assert make() == make()

    def test_replace_finds_rebuilt_handle(self):
        """replace() matches an equal-but-rebuilt object, not just identity."""
        inventory = build_inventory()
        old = build_inventory().networks[0].fiber_arrays[0].acquisitions[0]
        updated = inventory.replace(old, old.new(gauge_length=99.0))
        acq = updated.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.gauge_length == 99.0

    def test_replace_normalizes_inline_resources(self):
        """A replacement carrying an inline resource gets normalized."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        new = old.new(interrogator=inv.Interrogator(resource_id="int-9"))
        updated = inventory.replace(old, new)
        assert updated.get_resource("int-9").resource_id == "int-9"
        acq = updated.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.interrogator == "int-9"

    def test_replace_rejects_dangling_reference(self):
        """Replace rejects dangling reference."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        with pytest.raises(ValidationError, match="Dangling"):
            inventory.replace(old, old.new(interrogator="no-such-id"))

    def test_replace_finds_channel(self):
        """Replace finds channel."""
        chan = inv.Channel(code="HHZ")
        station = inv.Station(code="S1", channels=(chan,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", stations=(station,)),)
        )
        updated = inventory.replace(chan, chan.new(data_type="strain"))
        got = updated.networks[0].stations[0].channels[0]
        assert got.data_type == "strain"

    def test_keyless_dict_resource_adopts_key(self):
        """A dict resource without resource_id adopts its pool key."""
        inventory = inv.Inventory(
            resources={"cab-1": {"object_type": "Cable", "name": "mycable"}}
        )
        assert inventory.get_resource("cab-1").resource_id == "cab-1"
        loaded = inv.Inventory.from_yaml(inventory.io.to_yaml())
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_duplicate_array_codes_raise(self):
        """Same-code overlapping fiber arrays fail network check."""
        net = inv.Network(
            code="DAS",
            fiber_arrays=(
                inv.FiberArray(code="L001", start_time="2020-01-01"),
                inv.FiberArray(code="L001", start_time="2021-01-01"),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Duplicate fiber array"):
            net.check()

    def test_partial_axes_fail_the_inventory_check(self):
        """A segment stating some axes and not others fails the check."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(
                    distance=(0.0, 10.0), coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)}
                ),
                inv.Geometry(
                    distance=(20.0, 30.0),
                    coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
                ),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),)
        )
        with pytest.raises(InvalidInventoryError, match="every axis or none"):
            inventory.check()

    def test_inventory_function_dispatch(self):
        """dc.inventory handles bad input with clear errors."""
        # However it is spelled, a path which is not there says so.
        for missing in ("does_not_exist.yaml", Path("does_not_exist.yaml")):
            with pytest.raises(InvalidInventoryError, match="No such inventory file"):
                dc.inventory(missing)
        with pytest.raises(InvalidInventoryError, match="Could not get"):
            dc.inventory(123)
        existing = build_inventory()
        assert dc.inventory(existing) is existing

    def test_core_namespace_export(self):
        """Core namespace export."""
        assert Inventory is inv.Inventory


class TestCodexReviewRegressions:
    """Regression tests for counterpart (Codex) review findings."""

    def test_replace_reaches_path_track_items(self):
        """A recalibrated geometry segment is replaceable in place."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].optical_paths[0].geometry[0]
        new = old.new(name="recalibrated")
        updated = inventory.replace(old, new)
        got = updated.networks[0].fiber_arrays[0].optical_paths[0].geometry[0]
        assert got.name == "recalibrated"

    def test_reference_type_mismatch_raises(self):
        """A string ref resolving to the wrong resource type raises."""
        cable = inv.Cable(resource_id="cab-1")
        acq = inv.Acquisition(code="RAW", interrogator="cab-1")
        array = inv.FiberArray(code="L001", acquisitions=(acq,))
        with pytest.raises(ValidationError, match="expected one of"):
            inv.Inventory(
                resources=[cable],
                networks=(inv.Network(code="DAS", fiber_arrays=(array,)),),
            )

    def test_nested_null_not_equal_to_value(self):
        """A nested null array is not equal to a non-null one."""
        a = dc.PatchAttrs(foo={"x": np.array([np.nan])})
        b = dc.PatchAttrs(foo={"x": np.array([1.0])})
        assert a != b

    def test_nonfinite_interval_values_raise(self):
        """Nonfinite interval values raise."""
        with pytest.raises(ValidationError):
            inv.CouplingCondition(
                start_distance=np.nan, end_distance=10.0, coupling_type="trench"
            )
        with pytest.raises(ValidationError, match="finite"):
            inv.Geometry(
                distance=(0.0, np.inf), coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)}
            )
        with pytest.raises(ValidationError, match="finite"):
            inv.DistanceMap(channel=(0.0, np.inf), distance=(0.0, 1.0))


class TestCoverageCompleteness:
    """Exercise remaining branches so the patch stays fully covered."""

    def test_attenuation_none_without_loss(self):
        """No loss value means no derivable attenuation rate."""
        seg = inv.FiberSegment(optical_length=100.0)
        assert seg.attenuation_db_per_km is None

    def test_attenuation_scalar(self):
        """A scalar loss over a known length gives a per-km rate."""
        seg = inv.FiberSegment(optical_length=2000.0, loss_db=0.8)
        assert seg.attenuation_db_per_km == pytest.approx(0.4)

    def test_interval_optical_length(self):
        """Interval items report their length from start/end distances."""
        cond = inv.CouplingCondition(
            start_distance=10.0, end_distance=60.0, coupling_type="trench"
        )
        assert cond.optical_length == 50.0

    def test_normalize_keeps_existing_id_tuple(self):
        """A tuple ref field already holding id strings is left untouched."""
        m1 = inv.OpticalMeasurement(resource_id="m1", method="otdr", wavelength=1550.0)
        m2 = inv.OpticalMeasurement(resource_id="m2", method="otdr", wavelength=1310.0)
        segment = inv.FiberSegment(
            optical_length=10.0,
            loss_db=(0.4, 0.5),
            loss_measurement=("m1", "m2"),
        )
        path = inv.OpticalPath(optical_components=(segment,))
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),),
            resources={"m1": m1, "m2": m2},
        )
        got = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        assert got.optical_components[0].loss_measurement == ("m1", "m2")

    def test_duplicate_coordinate_labels_raise(self):
        """Duplicate coordinate labels raise."""
        with pytest.raises(ValidationError, match="unique"):
            inv.CoordinateReferenceSystem(coordinate_labels=("x", "x"))

    def test_distance_map_length_mismatch_raises(self):
        """Distance map length mismatch raises."""
        with pytest.raises(ValidationError, match="same length"):
            inv.DistanceMap(channel=(1.0, 2.0), distance=(5.0,))

    def test_distance_map_empty_raises(self):
        """Distance map empty raises."""
        with pytest.raises(ValidationError, match="at least 1 control point"):
            inv.DistanceMap(channel=(), distance=())

    def test_distance_map_non_increasing_raises(self):
        """Distance map non increasing raises."""
        with pytest.raises(ValidationError, match="values must be strictly"):
            inv.DistanceMap(channel=(2.0, 1.0), distance=(5.0, 6.0))
        with pytest.raises(ValidationError, match="distance values"):
            inv.DistanceMap(channel=(1.0, 2.0), distance=(6.0, 5.0))

    def test_coordinates_at_without_geometry(self):
        """Coordinates at without geometry."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=10.0),)
        )
        crs = inv.CoordinateReferenceSystem()
        assert np.all(np.isnan(path.coordinates_at([5.0], crs)))

    def test_select_drops_out_of_range_geometry(self):
        """Selection drops segments entirely outside the clip."""
        seg = dict(coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)})
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(distance=(0.0, 20.0), **seg),
                inv.Geometry(distance=(80.0, 100.0), **seg),
            ),
        )
        piece = path.select(distance=(40.0, 60.0))
        assert piece.geometry == ()

    def test_add_rejects_non_path(self):
        """Add rejects non path."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=10.0),)
        )
        with pytest.raises(TypeError):
            _ = path + 5
        with pytest.raises(TypeError):
            _ = 5 + path

    def test_resolve_multiple_paths_raises(self):
        """An unchecked inventory with concurrent paths fails resolve."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(inv.Acquisition(code="RAW"),),
            optical_paths=(
                inv.OpticalPath(name="a", start_time="2020-01-01"),
                inv.OpticalPath(name="b", start_time="2021-01-01"),
            ),
        )
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        with pytest.raises(InvalidInventoryError, match="2 optical paths"):
            inventory.resolve("DAS.L001..RAW", time="2022-01-01")

    def test_replace_network_and_station_and_path(self):
        """Replace works at network, station, and path levels."""
        station = inv.Station(code="S1")
        path = inv.OpticalPath(name="p", start_time="2020-01-01")
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        net = inv.Network(code="DAS", fiber_arrays=(array,), stations=(station,))
        inventory = inv.Inventory(networks=(net,))
        got = inventory.replace(station, station.new(name="renamed"))
        assert got.networks[0].stations[0].name == "renamed"
        got = inventory.replace(path, path.new(name="renamed"))
        assert got.networks[0].fiber_arrays[0].optical_paths[0].name == "renamed"
        got = inventory.replace(net, net.new(name="renamed"))
        assert got.networks[0].name == "renamed"

    def test_replace_missing_resource_raises(self):
        """Replace missing resource raises."""
        inventory = build_inventory()
        cable = inv.Cable(resource_id="ghost")
        with pytest.raises(InvalidInventoryError, match="not found"):
            inventory.replace(cable, cable.new(name="x"))

    def test_from_yaml_non_mapping_raises(self):
        """From yaml non mapping raises."""
        with pytest.raises(InvalidInventoryError, match="mapping"):
            inv.Inventory.from_yaml("- 1\n- 2\n")

    def test_model_equality_foreign_types(self):
        """Models compare False to unrelated types and True to equal dumps."""
        acq = inv.Acquisition(code="RAW")
        assert acq != 5
        assert acq == acq.model_dump()

    def test_equality_array_shape_mismatch(self):
        """Nested arrays of different shapes compare unequal."""
        a = dc.PatchAttrs(foo={"x": np.array([1.0, 2.0])})
        b = dc.PatchAttrs(foo={"x": np.array([1.0])})
        assert a != b

    def test_equality_sequence_length_mismatch(self):
        """Nested sequences of different lengths compare unequal."""
        a = inv.Network(code="DAS", fiber_arrays=(inv.FiberArray(code="A"),))
        b = inv.Network(code="DAS")
        assert a != b

    def test_replace_fiber_array(self):
        """Replace fiber array."""
        inventory = build_inventory()
        array = inventory.networks[0].fiber_arrays[0]
        got = inventory.replace(array, array.new(name="renamed"))
        assert got.networks[0].fiber_arrays[0].name == "renamed"

    def test_values_equal_branches(self):
        """Null-pattern and key mismatches compare unequal."""
        assert not values_equal(np.array([np.nan]), np.array([1.0]))
        assert values_equal(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
        assert not values_equal({"a": 1}, {"b": 1})
        assert values_equal((1.0, np.nan), (1.0, np.nan))


class TestObjectTypeTag:
    """The union members' own tag field is invisible to users."""

    def test_hidden_from_repr(self):
        """Hidden from repr."""
        cable = inv.Cable(resource_id="c1", name="c")
        assert "object_type" not in repr(cable)

    def test_present_in_dump(self):
        """Present in dump, unlike the tag every other model is given."""
        assert inv.Cable(resource_id="c1").model_dump()["object_type"] == "Cable"

    def test_wrong_tag_rejected(self):
        """Wrong tag rejected."""
        with pytest.raises(ValidationError):
            inv.Cable(resource_id="c1", object_type="Enclosure")


class TestUniformAttachments:
    """Every inventory object carries description and extra_fields."""

    def test_description_on_previous_gaps(self):
        """Descriptions on previous gaps."""
        acq = inv.Acquisition(code="RAW", description="tap tested twice")
        path = inv.OpticalPath(description="post-repair epoch")
        assert acq.description and path.description

    def test_yaml_omits_empty_fields(self):
        """Empty strings, dicts, and tuples do not serialize."""
        inventory = build_inventory()
        text = inventory.io.to_yaml()
        assert "description:" not in text
        assert "extra_fields:" not in text
        loaded = inv.Inventory.from_yaml(text)
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_extra_fields_contents_survive(self):
        """User values inside extra_fields are kept verbatim, even empty."""
        acq = inv.Acquisition(code="RAW", extra_fields={"vendor_flag": ""})
        array = inv.FiberArray(code="L001", acquisitions=(acq,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        loaded = inv.Inventory.from_yaml(inventory.io.to_yaml())
        got = loaded.networks[0].fiber_arrays[0].acquisitions[0]
        assert got.extra_fields == {"vendor_flag": ""}


class TestImmutability:
    """Inventory fields cannot be written to."""

    @staticmethod
    def _stocked_inventory(resource_id="fixed"):
        """An inventory whose frozen mappings both carry contents."""
        cable = inv.Cable(resource_id="cable-01", name="c")
        segment = inv.FiberSegment(optical_length=100.0, container=cable)
        array = inv.FiberArray(
            code="L001",
            optical_paths=(inv.OpticalPath(optical_components=(segment,)),),
        )
        return inv.Inventory(
            resource_id=resource_id,
            extra_fields={"vendor": "x", "gain": 1.5},
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),),
        )

    def test_extra_fields_refuses_writes(self):
        """extra_fields rejects item assignment."""
        acquisition = inv.Acquisition(code="RAW", extra_fields={"vendor": "x"})
        with pytest.raises(TypeError, match="item assignment"):
            acquisition.extra_fields["vendor"] = "y"

    def test_nested_extra_fields_refuses_writes(self):
        """The refusal holds everywhere in the tree, not just at the top."""
        inventory = build_inventory()
        nested = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        with pytest.raises(TypeError, match="item assignment"):
            nested.extra_fields["vendor"] = "y"

    def test_extra_fields_still_checks_value_types(self):
        """AfterValidator keeps the declared value types enforced."""
        with pytest.raises(ValidationError):
            inv.Acquisition(code="RAW", extra_fields={"vendor": [1, 2]})

    def test_resource_pool_refuses_writes(self):
        """Inventory.resources rejects item assignment."""
        inventory = inv.Inventory(resources=[inv.Cable(resource_id="cable_01")])
        with pytest.raises(TypeError, match="item assignment"):
            inventory.resources["cable_02"] = inv.Cable(resource_id="cable_02")

    def test_pool_built_by_validation_is_frozen(self):
        """Inline resources hoisted into the pool land in a frozen mapping."""
        # Exercises _normalize_resources, which bypasses the field validator.
        inventory = self._stocked_inventory()
        assert list(inventory.resources) == ["cable-01"]
        with pytest.raises(TypeError, match="item assignment"):
            inventory.resources["anything"] = None

    def test_frozen_pool_accepted_as_input(self):
        """A FrozenDict pool is keyed, not iterated."""
        first = inv.Inventory(resources=[inv.Cable(resource_id="cable_01")])
        second = inv.Inventory(resources=first.resources)
        assert isinstance(second.resources["cable_01"], inv.Cable)
        assert second.resources == first.resources

    def test_frozen_record_disagreeing_with_its_key_is_refused(self):
        """A record is read as a record whatever kind of mapping it is."""
        # Read as an object instead, its resource_id goes unseen and the
        # mismatch only surfaces on the next load.
        record = FrozenDict({"object_type": "Cable", "resource_id": "elsewhere"})
        with pytest.raises(ValidationError, match="disagrees with resource_id"):
            inv.Inventory(resources={"cable-01": record})

    def test_equal_inventories_hash_equally(self):
        """Equality and hashing agree, so an inventory works as a dict key."""
        # resource_id is pinned; it otherwise defaults to a fresh uuid.
        first, second = self._stocked_inventory(), self._stocked_inventory()
        assert first == second
        assert hash(first) == hash(second)
        assert {first: "found"}[second] == "found"

    def test_hash_survives_a_pickle_round_trip(self):
        """Pickle restores fields without validating, and spools are pickled."""
        inventory = self._stocked_inventory()
        loaded = pickle.loads(pickle.dumps(inventory))
        assert loaded == inventory
        assert hash(loaded) == hash(inventory)

    def test_hash_survives_a_yaml_round_trip(self):
        """Serializing and reloading does not move an inventory's hash."""
        inventory = self._stocked_inventory()
        loaded = inv.Inventory.from_yaml(inventory.io.to_yaml())
        assert loaded == inventory
        assert hash(loaded) == hash(inventory)

    def test_dumps_are_ordinary_dicts(self):
        """Frozen mappings serialize back to plain dicts at every depth."""
        inventory = inv.Inventory(
            resources=[inv.Cable(resource_id="cable_01", description="a cable")],
            extra_fields={"vendor": "x"},
        )
        dumped = inventory.model_dump()
        assert type(dumped["extra_fields"]) is dict
        assert type(dumped["resources"]) is dict
        assert type(dumped["resources"]["cable_01"]) is dict

    def test_json_mode_still_reaches_pooled_resources(self):
        """The pool's own serializer does not shadow what it holds."""
        run = inv.OpticalMeasurement(resource_id="otdr-1", time="2020-01-01")
        segment = inv.FiberSegment(optical_length=10.0, loss_measurement=run)
        array = inv.FiberArray(
            code="L001",
            optical_paths=(inv.OpticalPath(optical_components=(segment,)),),
        )
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        dumped = inventory.model_dump(mode="json")
        assert dumped["resources"]["otdr-1"]["time"].startswith("2020-01-01")
        assert json.dumps(dumped)


class TestStationXmlAlignment:
    """Fields added for FDSN StationXML import fidelity."""

    def test_successive_network_epochs_resolve(self):
        """A network code may be reused once its earlier epoch has ended."""
        acquisition = inv.Acquisition(code="RAW")
        array = inv.FiberArray(code="L001", acquisitions=(acquisition,))
        first = inv.Network(
            code="XX",
            start_time="2020-01-01",
            end_time="2021-01-01",
            fiber_arrays=(array,),
        )
        second = first.new(start_time="2021-01-01", end_time="")
        inventory = inv.Inventory(networks=(first, second)).check()
        context = inventory.resolve("XX.L001..RAW", time="2022-06-01")
        assert context.network.start_time == np.datetime64("2021-01-01")

    def test_network_epochs(self):
        """Networks carry validity epochs like every other container."""
        net = inv.Network(code="XX", start_time="2020-01-01", end_time="2022-01-01")
        assert net.is_effective_at("2021-06-01")
        assert not net.is_effective_at("2022-01-01")

    def test_identifiers_on_citable_levels(self):
        """Network, FiberArray, and Station carry citation identifiers."""
        doi = "doi:10.7914/SN/XX"
        net = inv.Network(code="XX", identifiers=(doi,))
        array = inv.FiberArray(code="L001", identifiers=(doi,))
        station = inv.Station(code="VA01", identifiers=(doi,))
        assert net.identifiers == array.identifiers == station.identifiers == (doi,)

    def test_channel_orientation_and_depth(self):
        """Channels hold azimuth, dip, and burial depth."""
        chan = inv.Channel(code="BHN", azimuth=0.0, dip=0.0, depth=100.0)
        assert (chan.azimuth, chan.dip, chan.depth) == (0.0, 0.0, 100.0)


class TestPointMarkers:
    """Zero-length intervals are point markers."""

    def test_aerial_coupling_type(self):
        """Aerial coupling type."""
        cond = inv.CouplingCondition(
            start_distance=0.0, end_distance=100.0, coupling_type="aerial"
        )
        assert cond.coupling_type == "aerial"

    def test_point_clamp_inside_span_is_legal(self):
        """A point marker inside a covered span does not count as overlap."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    start_distance=0.0, end_distance=80.0, coupling_type="trench"
                ),
                inv.CouplingCondition(
                    start_distance=40.0,
                    end_distance=40.0,
                    coupling_type="other",
                    description="clamp point",
                ),
            ),
        )
        assert path.check() is path

    def test_point_label(self):
        """Point label."""
        label = inv.OpticalPathLabel(
            start_distance=350.0, end_distance=350.0, group="wellhead"
        )
        assert label.interval == (350.0, 350.0)

    def test_point_markers_survive_select(self):
        """A clamp inside the clip is not coverage, but it is not nothing."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=50.0, end_distance=50.0, group="clamp"
                ),
            ),
            coupling=(
                inv.CouplingCondition(
                    start_distance=25.0, end_distance=25.0, coupling_type="other"
                ),
            ),
        )
        kept = path.select(distance=(10.0, 90.0))
        assert [x.interval for x in kept.labels] == [(50.0, 50.0)]
        assert [x.interval for x in kept.coupling] == [(25.0, 25.0)]

    def test_point_markers_outside_the_clip_are_dropped(self):
        """A marker beyond the requested window does not belong to the piece."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=95.0, end_distance=95.0, group="clamp"
                ),
            ),
        )
        assert path.select(distance=(10.0, 90.0)).labels == ()

    def test_point_marker_at_the_outer_endpoint_is_kept(self):
        """The outermost endpoint of the path is included, as everywhere."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=100.0, end_distance=100.0, group="end_cap"
                ),
            ),
        )
        kept = path.select(distance=(10.0, 100.0))
        assert [x.interval for x in kept.labels] == [(100.0, 100.0)]

    def test_reversed_interval_rejected(self):
        """An end before the start is rejected."""
        with pytest.raises(ValidationError, match="must not precede"):
            inv.CouplingCondition(
                start_distance=10.0, end_distance=5.0, coupling_type="trench"
            )


class TestOpticalLoss:
    """Unified loss/reflectance with measurement provenance."""

    def test_scalar_loss_no_provenance(self):
        """Plain numbers with no measurement records are legal."""
        splice = inv.Splice(loss_db=0.08, reflectance_db=-55.0)
        assert splice.loss_db == 0.08

    def test_shared_measurement_pooled_once(self):
        """One OTDR run backs many components through one pool entry."""
        run = inv.OpticalMeasurement(
            resource_id="otdr-1",
            method="otdr",
            wavelength=1550.0,
            pulse_width=1e-8,
            direction="forward",
        )
        path = inv.OpticalPath(
            optical_components=(
                inv.FiberSegment(
                    optical_length=1000.0, loss_db=0.3, loss_measurement=run
                ),
                inv.Splice(
                    loss_db=0.05,
                    loss_measurement=run,
                    reflectance_db=-60.0,
                    reflectance_measurement=run,
                ),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        assert list(inventory.resources) == ["otdr-1"]
        comps = (
            inventory.networks[0].fiber_arrays[0].optical_paths[0].optical_components
        )
        assert comps[0].loss_measurement == "otdr-1"
        assert comps[1].reflectance_measurement == "otdr-1"

    def test_multi_wavelength_pairs(self):
        """Tuple losses pair elementwise with their measurements."""
        sheet_1550 = inv.OpticalMeasurement(
            resource_id="ds-1550", method="datasheet", wavelength=1550.0
        )
        sheet_1310 = inv.OpticalMeasurement(
            resource_id="ds-1310", method="datasheet", wavelength=1310.0
        )
        seg = inv.FiberSegment(
            optical_length=2000.0,
            loss_db=(0.6, 0.7),
            loss_measurement=(sheet_1550, sheet_1310),
        )
        assert seg.attenuation_db_per_km == (0.3, 0.35)

    def test_tuple_loss_requires_tuple_measurements(self):
        """Tuple loss requires tuple measurements."""
        with pytest.raises(ValidationError, match="equal-length"):
            inv.FiberSegment(optical_length=10.0, loss_db=(0.1, 0.2))

    def test_length_mismatch_raises(self):
        """Length mismatch raises."""
        with pytest.raises(ValidationError, match="has 2 values"):
            inv.FiberSegment(
                optical_length=10.0,
                loss_db=(0.1, 0.2),
                loss_measurement=("m1", "m2", "m3"),
            )

    def test_dangling_measurement_ref_raises(self):
        """Dangling measurement ref raises."""
        seg = inv.FiberSegment(
            optical_length=10.0, loss_db=0.1, loss_measurement="no-such-run"
        )
        path = inv.OpticalPath(optical_components=(seg,))
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        with pytest.raises(ValidationError, match="Dangling"):
            inv.Inventory(networks=(inv.Network(code="DAS", fiber_arrays=(array,)),))

    def test_measurement_data_file_normalizes(self):
        """A measurement's trace-file resource lands in the pool."""
        trace = inv.ExternalResource(resource_id="trace-1", uri="file://x.sor")
        run = inv.OpticalMeasurement(resource_id="otdr-1", data=trace)
        inventory = inv.Inventory(resources=[run])
        assert inventory.get_resource("otdr-1").data == "trace-1"
        assert inventory.get_resource("trace-1").uri == "file://x.sor"

    def test_path_measurements_normalize(self):
        """Path-level measurement refs land in the pool."""
        run = inv.OpticalMeasurement(resource_id="otdr-1", method="otdr")
        path = inv.OpticalPath(measurements=(run,))
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        got = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        assert got.measurements == ("otdr-1",)


class TestCrsShape:
    """Units pair with axes; WKT carries non-registry definitions."""

    def test_default_units_match_default_labels(self):
        """The default CRS has honest per-axis units."""
        crs = inv.CoordinateReferenceSystem()
        assert crs.units == ("degree", "degree", "meter")

    def test_units_length_mismatch_raises(self):
        """Units length mismatch raises."""
        with pytest.raises(ValidationError, match="one entry per"):
            inv.CoordinateReferenceSystem(units=("degree",))

    def test_local_frame_with_wkt(self):
        """Local frame with wkt."""
        crs = inv.CoordinateReferenceSystem(
            authority="LOCAL",
            code="mine-grid-7",
            coordinate_labels=("x", "y", "z"),
            units=("meter", "meter", "meter"),
            wkt='ENGCRS["mine grid",EDATUM["portal"],CS[Cartesian,3]]',
        )
        assert crs.wkt.startswith("ENGCRS")


class TestPrReviewFindings:
    """Regressions for findings raised while reviewing the model PR."""

    def test_non_finite_spatial_interval_rejected(self):
        """A nan interval would silently poison every resolved distance."""
        with pytest.raises(ValidationError):
            inv.Acquisition(code="RAW", spatial_interval=np.nan)

    def test_start_distance_explains_its_removal(self):
        """An inventory written against the affine form says what to do."""
        with pytest.raises(ValidationError, match="no longer takes start_distance"):
            inv.Acquisition(code="RAW", start_distance=100.0)

    def test_duplicate_channel_identity_raises(self):
        """Channel (location_code, code) names a stream; it must be unique."""
        channel = inv.Channel(code="BHZ", location_code="00")
        station = inv.Station(code="VA01", channels=(channel, channel))
        with pytest.raises(InvalidInventoryError, match="Duplicate channel identity"):
            station.check()

    def test_station_channels_checked_from_network(self):
        """Station rules are reached by a whole-tree check."""
        channel = inv.Channel(code="BHZ")
        station = inv.Station(code="VA01", channels=(channel, channel))
        network = inv.Network(code="XX", stations=(station,))
        with pytest.raises(InvalidInventoryError, match="Duplicate channel identity"):
            inv.Inventory(networks=(network,)).check()

    def test_distinct_channel_epochs_are_legal(self):
        """The same stream may be described by successive epochs."""
        station = inv.Station(
            code="VA01",
            channels=(
                inv.Channel(code="BHZ", start_time="2020-01-01", end_time="2021-01-01"),
                inv.Channel(code="BHZ", start_time="2021-01-01"),
            ),
        )
        assert station.check() is station

    def test_add_rejects_different_lineage(self):
        """Concatenating unrelated paths would misattribute the result."""
        left = inv.OpticalPath(
            location_code="00",
            optical_components=(inv.FiberSegment(optical_length=10.0),),
        )
        right = left.new(location_code="01")
        with pytest.raises(InvalidInventoryError, match="one lineage and"):
            _ = left + right

    def test_add_rejects_different_epoch(self):
        """Concatenating across epochs would advertise the wrong validity."""
        left = inv.OpticalPath(
            start_time="2020-01-01",
            optical_components=(inv.FiberSegment(optical_length=10.0),),
        )
        right = left.new(start_time="2021-01-01")
        with pytest.raises(InvalidInventoryError, match="one lineage and"):
            _ = left + right

    def test_add_allows_matching_ongoing_epochs(self):
        """Two unset end times are the same epoch, not two unknowns."""
        path = inv.OpticalPath(
            start_time="2020-01-01",
            optical_components=(inv.FiberSegment(optical_length=10.0),),
        )
        assert (path + path).optical_length == 20.0

    def test_replace_rejects_ambiguous_match(self):
        """Equal items are indistinguishable, so replacing one is undefined."""
        connector = inv.Connector(connector_type="E2000")
        path = inv.OpticalPath(
            optical_components=(
                connector,
                inv.FiberSegment(optical_length=10.0),
                connector,
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),)
        )
        with pytest.raises(InvalidInventoryError, match="matches 2 items"):
            inventory.replace(connector, connector.new(name="first"))

    def test_replace_resource_addresses_one_id(self):
        """Resources are addressed by resource_id; look-alikes are untouched."""
        first = inv.Enclosure(resource_id="e1", name="box")
        twin = inv.Enclosure(resource_id="e2", name="box")
        inventory = inv.Inventory(resources={"e1": first, "e2": twin})
        out = inventory.replace(first, first.new(material="steel"))
        assert out.get_resource("e1").material == "steel"
        assert out.get_resource("e2").material == ""

    def test_replace_resource_rejects_stale_content(self):
        """An old which is not what the pool holds is not a correction."""
        stored = inv.Enclosure(resource_id="e1", name="box")
        inventory = inv.Inventory(resources={"e1": stored})
        stale = stored.new(name="crate")
        with pytest.raises(InvalidInventoryError, match="was not found"):
            inventory.replace(stale, stale.new(material="steel"))

    def test_replace_reaches_optical_measurements(self):
        """Measurements are pooled resources like any other."""
        measurement = inv.OpticalMeasurement(resource_id="m1", method="otdr")
        inventory = inv.Inventory(resources={"m1": measurement})
        out = inventory.replace(measurement, measurement.new(wavelength=1550.0))
        assert out.get_resource("m1").wavelength == 1550.0

    def test_coordinates_at_rejects_mixed_dimensions(self):
        """An unchecked, mixed-dimension path fails loudly, not by broadcast."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(
                    distance=(0.0, 10.0), coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)}
                ),
                inv.Geometry(
                    distance=(20.0, 30.0),
                    coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
                ),
            ),
        )
        crs = inv.CoordinateReferenceSystem()
        with pytest.raises(InvalidInventoryError, match="every axis or none"):
            path.coordinates_at([5.0], crs)

    def test_coordinate_width_must_match_crs(self):
        """Coordinates are read through the CRS, so they must fit its axes."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(
                    name="flat",
                    distance=(0.0, 10.0),
                    coordinates={"x": (0.0, 1.0), "y": (0.0, 1.0)},
                ),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),)
        )
        with pytest.raises(InvalidInventoryError, match="every axis or none"):
            inventory.check()

    def test_station_coordinate_width_must_match_crs(self):
        """Point coordinates answer to the same frame as fiber geometry."""
        station = inv.Station(code="VA01", coordinates=(1.0, 2.0))
        channel = inv.Channel(code="BHZ", coordinates=(1.0, 2.0))
        network = inv.Network(
            code="XX", stations=(station, inv.Station(code="VA02", channels=(channel,)))
        )
        with pytest.raises(InvalidInventoryError, match="coordinate values"):
            inv.Inventory(networks=(network,)).check()

    def test_two_axis_crs_accepts_two_axis_coordinates(self):
        """A declared 2D frame makes 2D coordinates correct, not deficient."""
        station = inv.Station(code="VA01", coordinates=(1.0, 2.0))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", stations=(station,)),),
            coordinate_reference_system=inv.CoordinateReferenceSystem(
                coordinate_labels=("x", "y"), units=("meter", "meter")
            ),
        )
        assert inventory.check() is inventory


class TestConstraintsMatchDescriptions:
    """Fields which promise a range or a real value must enforce it."""

    def test_azimuth_range(self):
        """Azimuth is a compass bearing."""
        assert inv.Channel(code="BHN", azimuth=359.9).azimuth == 359.9
        with pytest.raises(ValidationError):
            inv.Channel(code="BHN", azimuth=360.0)

    def test_dip_range(self):
        """Dip runs from straight down to straight up."""
        assert inv.Channel(code="BHZ", dip=-90.0).dip == -90.0
        with pytest.raises(ValidationError):
            inv.Channel(code="BHZ", dip=-91.0)

    def test_geometry_coordinates_must_be_finite(self):
        """A nan control point would read as uncovered distance."""
        with pytest.raises(ValidationError, match="must hold finite"):
            inv.Geometry(
                distance=(0.0, 1.0), coordinates={"x": (np.nan, 1.0), "y": (0.0, 1.0)}
            )

    def test_point_coordinates_must_be_finite(self):
        """Stations and channels name real positions."""
        with pytest.raises(ValidationError, match="must be finite"):
            inv.Station(code="VA01", coordinates=(1.0, np.inf))
        with pytest.raises(ValidationError, match="must be finite"):
            inv.Channel(code="BHZ", coordinates=(np.nan, 1.0))

    def test_crs_axis_count(self):
        """Canonical storage is (x, y, z), so a CRS declares one to three."""
        with pytest.raises(ValidationError, match="one to three axes"):
            inv.CoordinateReferenceSystem(coordinate_labels=(), units=())
        with pytest.raises(ValidationError, match="one to three axes"):
            inv.CoordinateReferenceSystem(
                coordinate_labels=("x", "y", "z", "depth"),
                units=("meter",) * 4,
            )

    def test_label_value_keeps_numpy_type(self):
        """A mask element is a flag, not the number one."""
        label = inv.OpticalPathLabel(
            start_distance=0.0, end_distance=1.0, group="noisy", value=np.bool_(True)
        )
        assert label.value is True
        counted = inv.OpticalPathLabel(
            start_distance=0.0, end_distance=1.0, group="shots", value=np.int64(5)
        )
        assert isinstance(counted.value, int) and not isinstance(counted.value, bool)

    def test_physical_quantities_must_be_finite(self):
        """A nan quantity is not a measurement."""
        with pytest.raises(ValidationError):
            inv.Acquisition(code="RAW", sample_rate=np.nan)
        with pytest.raises(ValidationError):
            inv.FiberSegment(optical_length=10.0, loss_db=(0.4, np.inf))

    def test_label_value_must_be_finite(self):
        """A non-finite value cannot survive a JSON round trip."""
        with pytest.raises(ValidationError, match="must be finite"):
            inv.OpticalPathLabel(
                start_distance=0.0, end_distance=1.0, group="g", value=np.inf
            )


def _sample_inventories() -> dict[str, inv.Inventory]:
    """Build inventories covering the ways serialization could lose a value."""
    blank_crs = inv.CoordinateReferenceSystem(
        authority="",
        code="",
        name="",
        wkt='LOCAL_CS["mine"]',
        coordinate_labels=("x", "y"),
        units=("m", "m"),
    )
    cable = inv.Cable(resource_id="cable-1", name="trunk")
    segment = inv.FiberSegment(name="run", optical_length=100.0, container=cable)
    labels = (
        inv.OpticalPathLabel(
            start_distance=0.0, end_distance=50.0, group="zone", value="east"
        ),
        inv.OpticalPathLabel(
            start_distance=0.0, end_distance=50.0, group="noisy", value=True
        ),
        inv.OpticalPathLabel(
            start_distance=0.0, end_distance=50.0, group="masked", value=False
        ),
        inv.OpticalPathLabel(
            start_distance=0.0, end_distance=50.0, group="shots", value=0
        ),
        inv.OpticalPathLabel(
            start_distance=50.0, end_distance=100.0, group="offset", value=0.0
        ),
    )

    def array(code, paths=(), **kwargs):
        return inv.FiberArray(code=code, optical_paths=tuple(paths), **kwargs)

    return {
        "defaults": inv.Inventory(),
        "blank_crs": inv.Inventory(coordinate_reference_system=blank_crs),
        "blank_resources": inv.Inventory(
            resources=(
                inv.Interrogator(resource_id="int-1", instrument_type=""),
                inv.Cable(resource_id="cable-2", name=""),
            )
        ),
        "explicit_empties": inv.Inventory(
            networks=(
                inv.Network(
                    code="XX",
                    name="",
                    description="",
                    identifiers=(),
                    extra_fields={},
                    fiber_arrays=(array("L001", name="", identifiers=()),),
                ),
            )
        ),
        "depths": inv.Inventory(
            networks=(
                inv.Network(code="N0"),
                inv.Network(code="N1", fiber_arrays=(array("L001"),)),
                inv.Network(
                    code="N2",
                    fiber_arrays=(
                        array("L002"),
                        array(
                            "L003",
                            paths=(
                                inv.OpticalPath(location_code="00"),
                                inv.OpticalPath(
                                    location_code="01",
                                    optical_components=(segment,),
                                    labels=labels,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        ),
        "shared_resource": inv.Inventory(
            networks=(
                inv.Network(
                    code="XX",
                    fiber_arrays=(
                        array(
                            "L001",
                            paths=(
                                inv.OpticalPath(
                                    location_code="00", optical_components=(segment,)
                                ),
                                inv.OpticalPath(
                                    location_code="01", optical_components=(segment,)
                                ),
                            ),
                        ),
                    ),
                ),
            )
        ),
        "extra_fields": inv.Inventory(
            networks=(
                inv.Network(
                    code="XX",
                    extra_fields={"blank": "", "zero": 0, "off": False, "none": 0.0},
                ),
            )
        ),
        "full": build_full_inventory(),
    }


SAMPLE_INVENTORIES = _sample_inventories()


def _walk_models(obj):
    """Yield every inventory model in a tree, the tree itself included."""
    if isinstance(obj, InventoryModel):
        yield obj
        for name in type(obj).model_fields:
            yield from _walk_models(getattr(obj, name))
    elif isinstance(obj, tuple | list):
        for item in obj:
            yield from _walk_models(item)
    elif isinstance(obj, Mapping):
        for item in obj.values():
            yield from _walk_models(item)


def _swap_model(obj, old, new):
    """Rebuild a tree with one model instance, matched by identity, swapped."""
    if obj is old:
        return new
    if isinstance(obj, InventoryModel):
        updates = {}
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            if (swapped := _swap_model(value, old, new)) is not value:
                updates[name] = swapped
        return obj.new(**updates) if updates else obj
    if isinstance(obj, tuple | list):
        items = tuple(_swap_model(x, old, new) for x in obj)
        changed = any(a is not b for a, b in zip(items, obj, strict=True))
        return items if changed else obj
    if isinstance(obj, Mapping):
        items = {key: _swap_model(value, old, new) for key, value in obj.items()}
        return items if any(items[key] is not obj[key] for key in obj) else obj
    return obj


def _empty_like(value):
    """Return the empty value of this value's type, or None if it has none."""
    if isinstance(value, str):
        return ""
    if isinstance(value, Mapping):
        return {}
    if isinstance(value, tuple):
        return ()
    return None


def _empty_keys(data, out=None) -> set[str]:
    """Return the keys a document writes with an empty value."""
    out = set() if out is None else out
    if isinstance(data, Mapping):
        for key, value in data.items():
            if value in ("", [], {}):
                out.add(key)
            _empty_keys(value, out)
    elif isinstance(data, list):
        for item in data:
            _empty_keys(item, out)
    return out


class TestSerializationIsLossless:
    """Pruning empty values must not change what reloads."""

    @pytest.mark.parametrize("name", sorted(SAMPLE_INVENTORIES))
    def test_round_trip_equals(self, name):
        """Whatever was written comes back, in text and through a file."""
        inventory = SAMPLE_INVENTORIES[name]
        assert dc.inventory(inventory.io.to_yaml()) == inventory

    def test_a_label_value_of_one_survives(self):
        """`1 == True`, and the value's default is True, so it was dropped."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=0.0, end_distance=10.0, group="hole", value=1
                ),
                inv.OpticalPathLabel(
                    start_distance=20.0, end_distance=30.0, group="hole", value=2
                ),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),)
        )
        text = inventory.io.to_yaml()
        assert "value: 1" in text
        # Without the value, the group reloads holding a boolean beside a
        # number and is refused as mixing two kinds.
        assert dc.inventory(text) == inventory

    def test_a_label_still_names_its_class(self):
        """Restoring the value must not displace the document's tag."""
        label = inv.OpticalPathLabel(
            start_distance=0.0, end_distance=1.0, group="hole", value=2
        )
        dumped = label.model_dump(mode="json")
        assert dumped["object_type"] == "OpticalPathLabel"

    def test_a_deliberately_excluded_value_stays_out(self):
        """What a caller filtered is not what exclude_defaults dropped."""
        label = inv.OpticalPathLabel(
            start_distance=0.0, end_distance=1.0, group="hole", value=2
        )
        assert "value" not in label.model_dump(mode="json", exclude={"value"})
        assert "value" not in label.model_dump(mode="json", include={"group"})

    def test_a_flag_label_stays_terse(self):
        """A value which really is the default is still left out."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            labels=(
                inv.OpticalPathLabel(
                    start_distance=0.0, end_distance=10.0, group="noisy"
                ),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", fiber_arrays=(array,)),)
        )
        text = inventory.io.to_yaml()
        assert "value:" not in text
        assert dc.inventory(text) == inventory

    def test_round_trip_through_file(self, tmp_path):
        """The writer taking a path writes what the text form holds."""
        inventory = build_full_inventory()
        path = tmp_path / "inventory.yaml"
        inventory.io.to_yaml(path)
        assert dc.inventory(path) == inventory

    def test_blank_crs_fields_survive(self):
        """A frame described by WKT alone must not reload as EPSG:4979."""
        inventory = SAMPLE_INVENTORIES["blank_crs"]
        crs = dc.inventory(inventory.io.to_yaml()).coordinate_reference_system
        assert (crs.authority, crs.code, crs.name) == ("", "", "")

    def test_blank_instrument_type_survives(self):
        """A blanked field with a non-empty default is not a missing one."""
        inventory = SAMPLE_INVENTORIES["blank_resources"]
        loaded = dc.inventory(inventory.io.to_yaml())
        assert loaded.resources["int-1"].instrument_type == ""

    def test_every_blankable_field_survives(self):
        """Emptying any field of any model must survive a round trip.

        Sweeping the model tree, rather than listing the fields at risk,
        is what catches the next field added with a non-empty default.
        """
        inventory = build_full_inventory()
        checked = set()
        for model in _walk_models(inventory):
            for name in type(model).model_fields:
                empty = _empty_like(getattr(model, name))
                if empty is None:
                    continue
                try:
                    blanked = model.new(**{name: empty})
                    candidate = _swap_model(inventory, model, blanked).check()
                except (ValidationError, InvalidInventoryError):
                    continue  # a field a validator refuses to blank
                # Outside the catch: an inventory valid here but not once it
                # has been written is a document the pruning damaged.
                loaded = dc.inventory(candidate.io.to_yaml())
                assert loaded == candidate, f"{type(model).__name__}.{name} was lost"
                checked.add((type(model).__name__, name))
        # The fields the reported bug was found in must be among those swept.
        crs = "CoordinateReferenceSystem"
        assert {(crs, "authority"), (crs, "code"), (crs, "name")} <= checked
        assert ("Interrogator", "instrument_type") in checked

    def test_defaulted_fields_are_dropped(self):
        """A field still holding its default is left out of the document."""
        data = yaml.safe_load(build_inventory().io.to_yaml())
        assert not _empty_keys(data)
        assert "description" not in yaml.safe_dump(data)

    def test_document_states_its_schema_version(self):
        """The envelope version is written, default or not.

        Every other defaulted field is dropped, so this is what says which
        envelope a document was written against.
        """
        default = inv.Inventory().schema_version
        data = yaml.safe_load(build_inventory().io.to_yaml())
        assert data["schema_version"] == default
        other = inv.Inventory(schema_version=default + 1)
        assert dc.inventory(other.io.to_yaml()).schema_version == default + 1

    def test_empty_label_value_is_rejected(self):
        """An empty value would have to survive serialization to mean anything.

        It used to be legal, and this guarded it against being pruned and
        reloading as a boolean flag. It is now rejected outright: it states
        nothing, and a string coordinate spells an uncovered channel with
        the empty string, so a covered one could not be told apart.
        """
        with pytest.raises(ValidationError, match="may not be the empty string"):
            inv.OpticalPathLabel(
                start_distance=0.0, end_distance=50.0, group="rock", value=""
            )


class TestLoadingValidates:
    """Reading a document asks whether it is a valid inventory."""

    def test_invalid_document_raises_on_load(self):
        """A document violating a whole-tree rule fails at its source."""
        station = inv.Station(code="VA01", coordinates=(1.0, 2.0))
        inventory = inv.Inventory(
            networks=(inv.Network(code="XX", stations=(station,)),)
        )
        text = inventory.io.to_yaml()
        with pytest.raises(InvalidInventoryError, match="coordinate values"):
            inv.Inventory.from_yaml(text)
        with pytest.raises(InvalidInventoryError, match="coordinate values"):
            dc.inventory(text)

    def test_in_memory_construction_stays_unchecked(self):
        """An inventory can be assembled a piece at a time."""
        station = inv.Station(code="VA01", coordinates=(1.0, 2.0))
        assert inv.Inventory(networks=(inv.Network(code="XX", stations=(station,)),))


class TestFiberSegmentFields:
    """Fiber-level naming and optical calibration fields."""

    def test_fiber_number_and_color(self):
        """Fiber identity within a cable uses telecom naming."""
        segment = inv.FiberSegment(
            optical_length=10.0, fiber_number=3, fiber_color="blue"
        )
        assert segment.fiber_number == 3
        assert segment.fiber_color == "blue"

    def test_refractive_index(self):
        """The group index converts time of flight into distance."""
        segment = inv.FiberSegment(optical_length=10.0, refractive_index=1.4682)
        assert segment.refractive_index == 1.4682

    def test_refractive_index_must_be_finite(self):
        """A non-finite index would poison every distance it scales."""
        with pytest.raises(ValidationError):
            inv.FiberSegment(optical_length=10.0, refractive_index=np.nan)


class TestDepthLabel:
    """Boreholes and mines measure down, not up."""

    def test_depth_is_in_the_vocabulary(self):
        """Depth is a canonical vertical axis label."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("easting", "northing", "depth"),
            units=("meter", "meter", "meter"),
        )
        assert crs.axis_index("depth") == 2


class TestDistanceMapAxisAgreement:
    """Two input axes describe one interrogator."""

    def test_varying_spacing_raises(self):
        """No interrogator samples at two different spacings."""
        with pytest.raises(ValidationError, match="varies along the fiber"):
            inv.DistanceMap(
                channel=(0.0, 100.0, 200.0),
                instrument_distance=(0.0, 150.0, 200.0),
                distance=(0.0, 100.0, 200.0),
            )

    def test_constant_spacing_is_fine(self):
        """A fixed spacing between the axes is the normal case."""
        dmap = inv.DistanceMap(
            channel=(0.0, 100.0, 200.0),
            instrument_distance=(0.0, 50.0, 100.0),
            distance=(0.0, 100.0, 200.0),
        )
        assert dmap.axes == ("channel", "instrument_distance")

    def test_bad_axis_name_raises(self):
        """A map is read on an input axis, not on any of its fields."""
        dmap = inv.DistanceMap(channel=(0.0, 1.0), distance=(5.0, 6.0))
        with pytest.raises(InvalidInventoryError, match="not a DistanceMap input"):
            dmap.map_to_distance([0.5], axis="distance")


class TestGetNames:
    """The names an inventory could contribute to a patch."""

    @pytest.fixture(scope="class")
    def names(self):
        """The names of the shared test inventory."""
        return build_inventory().get_names()

    def test_attrs_match_the_reader_vocabulary(self, names):
        """
        The names read off the models are the ones readers write.

        INVENTORY_ATTRS is what a reader puts in patch attrs and this is
        what enrichment could copy there; the two being one vocabulary is
        what keeps a header value and an enriched value one attr. The
        data-state trio is the only difference: enrichment copies it when
        asked by name rather than in the blanket form.
        """
        data_state = {"data_type", "data_category", "data_units"}
        assert set(names.attrs) == set(INVENTORY_ATTRS) | data_state

    def test_attrs_are_model_fields(self, names):
        """Every attr name is a field of the model it is read from."""
        for name in names.attrs:
            prefix, _, field = name.rpartition(".")
            model = inv.Interrogator if prefix == "interrogator" else inv.Acquisition
            assert not prefix or prefix == "interrogator", name
            assert field in model.model_fields, name

    def test_attrs_exclude_structure(self, names):
        """
        What locates an entry is not a fact about the system.

        The codes and epoch bounds resolve a patch to its acquisition, and
        the distance map and interrogator hold whole records rather than
        values; none of them is something a patch could carry as an attr.
        """
        excluded = {"code", "location_code", "start_time", "end_time"}
        excluded |= {"distance_map", "interrogator", "extra_fields", "description"}
        assert not set(names.attrs) & excluded

    def test_a_declaration_naming_nothing_is_refused(self, monkeypatch):
        """
        The map is derived, and the derivation checks what it derives.

        A model declaring an identity field it does not have would build
        an entry pointing at nothing, so it fails where it is built
        rather than somewhere a bare track name quietly resolves to NaN.
        """
        monkeypatch.setattr(inv.CouplingCondition, "_identity_field", "not_a_field")
        with pytest.raises(AssertionError):
            inv._track_identity_fields()

    def test_coords_hold_the_tracks_and_groups(self, names):
        """The path's tracks, their fields, and its label groups."""
        assert "coupling" in names.coords  # bare: the identity field
        assert "coupling.medium" in names.coords
        assert "geometry" in names.coords
        assert "optical_components.fiber_type" in names.coords
        assert "zone" in names.coords  # the label group

    def test_coords_hold_both_axis_spellings(self, names):
        """A CRS axis is selectable as stored and as this CRS reads it."""
        crs = build_inventory().coordinate_reference_system
        assert set("xyz") <= set(names.coords)
        assert set(crs.coordinate_labels) <= set(names.coords)

    def test_coords_hold_optical_distance(self, names):
        """Optical distance is a coordinate enrich can be asked for."""
        assert "distance" in names.coords

    def test_coords_exclude_track_structure(self, names):
        """
        A track's placement is not a value its channels take.

        The interval bounds say where a coupling condition applies, and a
        geometry's control points are the segment itself; neither is a
        number a channel inside it carries.
        """
        excluded = {"coupling.start_distance", "coupling.end_distance"}
        excluded |= {"geometry.distance", "geometry.coordinates"}
        assert not set(names.coords) & excluded

    def test_coords_omit_absent_tracks(self):
        """An inventory with no coupling has no coupling to select on."""
        base = build_inventory()
        path = base.networks[0].fiber_arrays[0].optical_paths[0]
        bare = base.replace(path, path.new(coupling=(), labels=()))
        coords = set(bare.get_names().coords)
        assert not {x for x in coords if x.startswith("coupling")}
        assert "zone" not in coords
        assert "geometry" in coords  # the tracks it does have remain

    def test_names_are_unique(self, names):
        """A name means one thing, so it is listed once."""
        assert len(set(names.attrs)) == len(names.attrs)
        assert len(set(names.coords)) == len(names.coords)

    def test_attrs_do_not_depend_on_contents(self):
        """The models decide the attrs, so every inventory has the same."""
        assert Inventory().get_names().attrs == build_inventory().get_names().attrs


class TestGetNamesRoundTrip:
    """Every name the accessor lists is one enrichment can be asked for."""

    def test_every_coord_name_projects(self):
        """
        Enriching by each listed name gives a coordinate, or says nothing.

        Contributing a name is not promising a value for it, but it is
        promising the name is one enrichment understands: a name it
        cannot project at all would be listed and unusable.
        """
        patch, inventory = dc.examples.inventory_patch_pair()
        for name in inventory.get_names().coords:
            if name in patch.coords.coord_map:
                continue  # the patch's own axis, which enrich will not touch
            out = patch.enrich(
                inventory, attrs=False, coords=(name,), on_missing="null"
            )
            assert name in out.coords.coord_map, name

    def test_bare_track_name_is_its_identity(self):
        """
        A bare track name means the field the blessed map points at.

        The map is what makes `coupling` a name at all, so a bare use of
        it has to give the same values as the qualified one.
        """
        patch, inventory = dc.examples.inventory_patch_pair()
        for track, field in inv.TRACK_IDENTITY_FIELDS.items():
            bare = patch.enrich(inventory, attrs=False, coords=(track,))
            qualified = patch.enrich(
                inventory, attrs=False, coords=(f"{track}.{field}",)
            )
            assert np.array_equal(
                bare.get_coord(track).values,
                qualified.get_coord(f"{track}.{field}").values,
            ), track

    def test_multi_valued_field_is_not_listed(self):
        """
        A field stated as several values has none to give a channel.

        A multi-wavelength loss is a legitimate thing for a component to
        record and an impossible thing to project, so the name is not one
        this inventory contributes however scalar its annotation reads.
        """
        base = build_inventory()
        path = base.networks[0].fiber_arrays[0].optical_paths[0]
        component = path.optical_components[0]
        measured = base.new(
            resources={"m1": inv.OpticalMeasurement(resource_id="m1")}
        ).replace(
            component,
            component.new(loss_db=(0.2, 0.25), loss_measurement=("m1", "m1")),
        )
        coords = set(measured.get_names().coords)
        assert "optical_components.loss_db" not in coords
        assert "optical_components.name" in coords
        assert "optical_components.loss_db" in set(base.get_names().coords)

    def test_optional_collection_is_not_a_value(self):
        """
        "A tuple or nothing" is still a tuple, not a value.

        `tuple[float, ...] | None` is how this file spells several of its
        fields, so reading the `None` arm as the scalar would list them.
        """

        class _Probe(inv.InventoryModel):
            """A model shaped like the optional collections here."""

            wavelengths: tuple[float, ...] | None = None
            plain: float | None = None

        assert inv._value_field_names(_Probe) == ("plain",)

    def test_annotated_unions_are_transparent(self):
        """
        A discriminated union is written Annotated, and holds models.

        The models here spell their unions that way, so a field holding
        one must read as a reference rather than as a value.
        """

        class _Probe(inv.InventoryModel):
            """A model whose reference is spelled the way this file does."""

            resource: inv._Resource | None = None
            plain: float | None = None

        assert inv._value_field_names(_Probe) == ("plain",)


class TestInventoryNamespaces:
    """An inventory hosts method namespaces, as a patch and a spool do."""

    def test_io_namespace(self):
        """The io namespace DASCore registers is reachable."""
        inventory = build_inventory()
        assert inventory.io.to_yaml() == inv.inventory_to_yaml(inventory)

    def test_copy_gets_its_own_binding(self):
        """A copied inventory hands out a namespace bound to the copy."""
        inventory = build_inventory()
        inventory.io  # a namespace kept on the host would ride along
        other = inventory.model_copy(update={"schema_version": 7})
        assert other.io.to_yaml().startswith("schema_version: 7")

    def test_local_namespace_attaches(self):
        """A namespace defined without an entry point still attaches."""

        class _Local(InventoryNameSpace):
            name = "some_local_namespace"

            def network_count(inventory) -> int:  # noqa: N805
                """Return how many networks the inventory holds."""
                return len(inventory.networks)

        inventory = build_inventory()
        assert inventory.some_local_namespace.network_count() == len(inventory.networks)

    def test_unknown_attr_raises(self):
        """A name no namespace claims raises DASCore's message."""
        inventory = build_inventory()
        with pytest.raises(AttributeError, match="Inventory has no attribute 'nope'"):
            inventory.nope

    def test_private_attrs_still_resolve(self):
        """The namespace search does not stand in front of pydantic's own."""
        # _cache is a private attribute pydantic resolves in __getattr__,
        # which the namespace search would otherwise answer first.
        assert build_inventory()._cache == {}

    def test_unknown_private_attr_raises(self):
        """A private name neither pydantic nor a namespace knows still fails."""
        with pytest.raises(AttributeError, match="_not_a_field"):
            build_inventory()._not_a_field

    def test_cached_namespace_is_not_state(self):
        """An attached namespace changes neither equality nor a dump."""
        inventory = build_inventory()
        other = inv.Inventory(**inventory.model_dump())
        inventory.io.to_yaml()
        assert inventory == other
        assert hash(inventory) == hash(other)
        assert inventory.model_dump(mode="json") == other.model_dump(mode="json")
