"""Tests for example fetching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.examples as dc_examples
from dascore.examples import EXAMPLE_INVENTORIES, EXAMPLE_PATCHES
from dascore.exceptions import UnknownExampleError
from dascore.utils.intervals import normalize_value, value_kind
from dascore.utils.time import to_float


class TestGetExamplePatch:
    """Test suite for `get_example_patch`."""

    def test_default(self):
        """Ensure calling get_example_patch with no args returns patch."""
        patch = dc.get_example_patch()
        assert isinstance(patch, dc.Patch)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example patch"):
            dc.get_example_patch("NotAnExampleRight????")

    def test_data_file_name(self):
        """Ensure get_example_spool works on a datafile."""
        spool = dc.get_example_spool("dispersion_event.h5")
        assert isinstance(spool, dc.BaseSpool)

    def test_get_example_patch_data_file_name(self):
        """Ensure get_example_patch can load file-backed registry entries."""
        patch = dc.get_example_patch("dispersion_event.h5")
        assert isinstance(patch, dc.Patch)

    @pytest.mark.parametrize("name", EXAMPLE_PATCHES)
    def test_load_example_patch(self, name):
        """Ensure the registered example patches can all be loaded."""
        patch = dc.get_example_patch(name)
        assert isinstance(patch, dc.Patch)

    def test_file_backed_examples_use_direct_read(self, monkeypatch):
        """File-backed examples should not depend on file-spool patch resolution."""
        patch = dc.get_example_patch()

        def _fetch(_name):
            return "ignored-path"

        def _read(_path):
            return [patch]

        def _spool(_path):
            raise AssertionError("file-backed examples should load via dc.read")

        monkeypatch.setattr(dc_examples, "fetch", _fetch)
        monkeypatch.setattr(dc_examples.dc, "read", _read)
        monkeypatch.setattr(dc_examples.dc, "spool", _spool)

        out = dc_examples.example_event_1()
        assert isinstance(out, dc.Patch)

    def test_file_backed_registry_examples_use_direct_read(self, monkeypatch):
        """Registry-backed patch examples should load via direct read."""
        patch = dc.get_example_patch()

        def _fetch(_name):
            return "ignored-path"

        def _read(_path):
            return [patch]

        def _spool(_path):
            raise AssertionError("registry-backed examples should load via dc.read")

        monkeypatch.setattr(dc_examples, "fetch", _fetch)
        monkeypatch.setattr(dc_examples.dc, "read", _read)
        monkeypatch.setattr(dc_examples.dc, "spool", _spool)

        out = dc.get_example_patch("dispersion_event.h5")
        assert isinstance(out, dc.Patch)


class TestGetExampleSpool:
    """Test suite for `get_example_spool`."""

    def test_default(self):
        """Ensure calling get_example_spool with no args returns a Spool."""
        patch = dc.get_example_spool()
        assert isinstance(patch, dc.BaseSpool)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example spool"):
            dc.get_example_spool("NotAnExampleRight????")

    def test_data_file_name(self):
        """Ensure get_example_spool works on a datafile."""
        spool = dc.get_example_spool("dispersion_event.h5")
        assert isinstance(spool, dc.BaseSpool)


class TestGetExampleInventory:
    """Test suite for `get_example_inventory`."""

    def test_default(self):
        """Ensure calling get_example_inventory with no args returns one."""
        inventory = dc.get_example_inventory()
        assert isinstance(inventory, dc.Inventory)

    def test_raises_on_bad_key(self):
        """Ensure a bad key raises expected error."""
        with pytest.raises(UnknownExampleError, match="No example inventory"):
            dc.get_example_inventory("NotAnExampleRight????")

    @pytest.mark.parametrize("name", EXAMPLE_INVENTORIES)
    def test_load_example_inventory(self, name):
        """Each registered inventory loads and passes its own checks."""
        inventory = dc.get_example_inventory(name)
        assert isinstance(inventory, dc.Inventory)
        # check returns self, so this both validates and pins that.
        assert inventory.check() == inventory


class TestDiverseInventory:
    """The diverse example exists to exercise the things plots need."""

    @pytest.fixture(scope="class")
    def inventory(self):
        """The diverse example inventory."""
        return dc.get_example_inventory("diverse_das")

    @pytest.fixture(scope="class")
    def tunnel_path(self, inventory):
        """The tunnel path as it was before the repair."""
        return inventory.networks[0].fiber_arrays[0].optical_paths[0]

    def test_two_networks_and_arrays(self, inventory):
        """Both spellings of breadth are present."""
        assert len(inventory.networks) == 2
        assert {x.code for x in inventory.networks} == {"XT", "XB"}
        assert len(list(inventory._optical_paths())) == 3

    def test_repair_is_two_epochs_of_one_location(self, inventory):
        """One location code carries two paths which do not overlap."""
        paths = inventory.networks[0].fiber_arrays[0].optical_paths
        assert len({x.location_code for x in paths}) == 1
        assert not paths[0].overlaps(paths[1])
        # The repair spliced fiber in, so the later path is longer.
        assert paths[1].optical_length > paths[0].optical_length

    def test_epochs_span_the_open_and_closed_cases(self, inventory):
        """A timeline needs an ongoing epoch, a closed one, and an unset one."""
        acquisitions = inventory.networks[0].fiber_arrays[0].acquisitions
        ongoing, closed = acquisitions
        assert pd.isnull(ongoing.end_time) and not pd.isnull(ongoing.start_time)
        assert not pd.isnull(closed.end_time)
        unset = inventory.networks[1].fiber_arrays[0].acquisitions[0]
        assert pd.isnull(unset.start_time) and pd.isnull(unset.end_time)

    def test_geometry_gap_is_a_real_gap(self, inventory, tunnel_path):
        """The slack coil states no position, so its channels get NaN."""
        crs = inventory.coordinate_reference_system
        coords = tunnel_path.coordinates_at(np.array([250.0, 320.0, 375.0]), crs)
        assert not np.isnan(coords[0]).any()
        assert np.isnan(coords[1]).all()
        assert not np.isnan(coords[2]).any()

    def test_states_a_column_which_is_not_a_position(self, tunnel_path):
        """A non-axis geometry column gives the line panels something to draw."""
        assert "chainage" in tunnel_path.geometry_columns()
        values = tunnel_path.column_at("chainage", np.array([150.0, 320.0]))
        assert values[0] == 1250.0
        assert np.isnan(values[1])

    def test_every_label_kind_appears(self, tunnel_path):
        """One group of each kind, which is what decides a color treatment."""
        kinds = {}
        for label in tunnel_path.labels:
            value = normalize_value(label.value)
            kinds.setdefault(label.group, set()).add(value_kind(value))
        assert kinds == {
            "zone": {"string"},
            "noisy": {"boolean"},
            "borehole": {"numeric"},
        }

    def test_holds_a_point_marker(self, tunnel_path):
        """A zero length component is a point, which a plot must still show."""
        intervals = tunnel_path.component_intervals()
        assert any(start == end for start, end in intervals)

    def test_coupling_covers_only_part_of_the_path(self, tunnel_path):
        """Partial coverage is legal and is what a coverage plot must show."""
        assert len({x.coupling_type for x in tunnel_path.coupling}) > 1
        covered = sum(x.optical_length for x in tunnel_path.coupling)
        assert covered < tunnel_path.optical_length


class TestRickerMoveout:
    """Tests for Ricker moveout patch."""

    def test_moveout(self):
        """Ensure peaks of ricker wavelet line up with expected moveout."""
        velocity = 100
        patch = dc.get_example_patch("ricker_moveout", velocity=velocity)
        argmaxes = np.argmax(patch.data, axis=0)
        peak_times = patch.get_coord("time").values[argmaxes]
        moveout = to_float(peak_times - np.min(peak_times))
        distances = patch.get_coord("distance").values
        expected_moveout = distances / velocity
        assert np.allclose(moveout, expected_moveout)


class TestDeltaPatch:
    """Tests for the delta_patch example."""

    @pytest.mark.parametrize("invalid_dim", ["inv_dim", "", None, 123, 1.1])
    def test_delta_patch_invalid_dim(self, invalid_dim):
        """
        Test that passing an invalid dimension value raises a ValueError.
        """
        msg = "with 'time' and 'distance'"
        with pytest.raises(ValueError, match=msg):
            dc.get_example_patch("delta_patch", dim=invalid_dim)

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_delta_patch_structure(self, dim):
        """Test that the delta_patch returns a Patch with correct structure."""
        patch = dc.get_example_patch("delta_patch", dim=dim)
        assert isinstance(patch, dc.Patch), "delta_patch should return a Patch instance"

        dims = patch.dims
        assert "time" in dims and "distance" in dims, (
            "Patch must have 'time' and 'distance' dimensions"
        )

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_delta_patch_delta_location(self, dim):
        """
        Ensures the delta is at the center of the chosen dimension and zeros elsewhere.
        """
        # The default shape from the function signature: shape=(10, 200)
        # If dim="time", we end up with a single (distance=0) trace => shape (200,)
        # If dim="distance", we end up with a single (time=0) trace => shape (10,)
        patch = dc.get_example_patch("delta_patch", dim=dim)
        data = patch.squeeze().data

        # The expected midpoint and verify single delta at center
        mid_idx = len(data) // 2

        assert data[mid_idx] == 1.0, "Expected a unit delta at the center"
        # Check all other samples are zero
        # Replace the center value with zero and ensure all zeros remain
        test_data = np.copy(data)
        test_data[mid_idx] = 0
        assert np.allclose(test_data, 0), (
            "All other samples should be zero except the center"
        )

    @pytest.mark.parametrize("dim", ["time", "distance"])
    def test_delta_patch_with_patch(self, dim):
        """Test passing an existing patch to delta_patch and ensure delta is applied."""
        # Create a base patch
        base_patch = dc.get_example_patch("random_das", shape=(5, 50))
        # Apply the delta_patch function with the existing patch
        delta_applied_patch = dc.get_example_patch(
            "delta_patch", dim=dim, patch=base_patch
        )

        assert isinstance(delta_applied_patch, dc.Patch), "Should return a Patch"
        data = delta_applied_patch.squeeze().data

        # Check that only the center value is one and others are zero
        mid_idx = len(data) // 2
        assert data[mid_idx] == 1.0, "Center sample should be 1.0"
        test_data = np.copy(data)
        test_data[mid_idx] = 0
        assert np.allclose(test_data, 0), (
            "All other samples should be zero except the center"
        )

    @pytest.mark.parametrize("dim", ["lag_time", "distance"])
    def test_delta_patch_with_3d_patch(self, dim):
        """Test passing a 3D patch."""
        # Create a base patch
        base_patch = dc.get_example_patch("sin_wav")
        base_patch_3d = base_patch.correlate(distance=[2], samples=True)
        # Apply the delta_patch function with the existing patch
        delta_applied_patch = dc.get_example_patch(
            "delta_patch", dim=dim, patch=base_patch_3d
        )
        assert isinstance(delta_applied_patch, dc.Patch), "Should return a Patch"
        data = delta_applied_patch.squeeze().data
        # Check that only the center value is one and others are zero
        mid_idx = len(data) // 2
        assert data[mid_idx] == 1.0, "Center sample should be 1.0"
        test_data = np.copy(data)
        test_data[mid_idx] = 0
        assert np.allclose(test_data, 0), (
            "All other samples should be zero except the center"
        )
