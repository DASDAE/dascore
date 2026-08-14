"""Tests for loading an inventory from an authoring directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core import inventory as inv
from dascore.core import inventory_loader as loader
from dascore.exceptions import (
    InvalidInventoryError,
    MissingOptionalDependencyError,
)
from dascore.models import InventoryModel, TimeRangedModel
from dascore.models.registry import TAG_FIELD

pytest.importorskip("yaml")


# A minimal directory which loads: one acquisition names everything above it.
MINIMAL = {
    "acquisitions/DAS.L001..RAW.yaml": "object_type: Acquisition\ndata_category: DAS\n",
}


def write_inventory(root, files) -> object:
    """Write a mapping of relative path to text under root."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _folds_case() -> bool:
    """Return True if this filesystem holds two case variants as one file."""
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        (directory / "CaseProbe").write_text("")
        return (directory / "caseprobe").exists()


# Asked once, at collection. Windows and most macOS checkouts fold case, so
# a name differing from another only by case is the same file there.
FOLDS_CASE = _folds_case()


def _keeps_trailing_dot() -> bool:
    """Return True if this filesystem keeps a trailing dot in a name."""
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        try:
            (directory / "probe.").mkdir()
        except OSError:  # pragma: no cover - no filesystem here refuses it
            return False
        return (directory / "probe.").exists() and not (directory / "probe").exists()


# Windows strips a trailing dot, so `path.` and `path` are one directory
# there and the blank location cannot be spelled twice.
KEEPS_TRAILING_DOT = _keeps_trailing_dot()


@pytest.fixture
def make_inventory(tmp_path):
    """Return a function which writes a directory and loads it."""

    def build(files, name="my_inventory"):
        return dc.inventory(write_inventory(tmp_path / name, files))

    return build


class TestRegistry:
    """The container registry must stay pinned to the models."""

    def test_containers_are_the_ones_the_format_defines(self):
        """Pinned so the loops below cannot pass over an empty registry."""
        assert set(loader._CONTAINERS) == {
            "resources",
            "networks",
            "fiber_arrays",
            "stations",
            "acquisitions",
        }

    def test_identity_tokens_are_fields_or_levels(self):
        """Every name token states a field or names a containing entity."""
        checked = []
        for container in loader._CONTAINERS.values():
            for model in container.models:
                for token in container.identity:
                    known = token in model.model_fields
                    assert known or token in loader._ADDRESS_LEVELS
                    checked.append((model, token))
        assert len(checked) == 14

    def test_resource_container_matches_the_union(self):
        """The resources container holds exactly the resource union."""
        union = inv.get_args(inv.get_args(inv._Resource)[0])
        assert set(loader._CONTAINERS["resources"].models) == set(union)
        assert len(union) == 5

    def test_every_model_name_is_known(self):
        """A model this loader can build counts as a model for near-misses."""
        names = loader._model_names()
        built = {x.__name__ for c in loader._CONTAINERS.values() for x in c.models}
        assert built <= names
        assert len(built) == 9
        assert "Inventory" in names


class TestLoadDirectory:
    """Tests for loading a well-formed directory."""

    def test_minimal_directory(self, make_inventory):
        """One acquisition file names the network and array which hold it."""
        out = make_inventory(MINIMAL)
        assert [x.code for x in out.networks] == ["DAS"]
        array = out.networks[0].fiber_arrays[0]
        assert array.code == "L001"
        assert [x.code for x in array.acquisitions] == ["RAW"]
        # The blank token is a location code, which may be blank.
        assert array.acquisitions[0].location_code == ""

    def test_resolves_like_any_inventory(self, make_inventory):
        """A loaded inventory answers the key its directory spells."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW.yaml": (
                    "object_type: Acquisition\ndata_category: DAS\ngauge_length: 10.0\n"
                ),
            }
        )
        assert out.resolve("DAS.L001..RAW").acquisition.gauge_length == 10.0

    def test_full_directory(self, make_inventory):
        """Every container contributes to one inventory."""
        out = make_inventory(
            {
                "inventory.yaml": "object_type: Inventory\nschema_version: 1\n",
                "resources/int_01.yaml": (
                    "object_type: Interrogator\nmanufacturer: Fake\nmodel: FI-1\n"
                ),
                "networks/DAS.yaml": "object_type: Network\nname: test network\n",
                "fiber_arrays/DAS.L001.yaml": "object_type: FiberArray\nname: first\n",
                "stations/DAS.STA1.yaml": "object_type: Station\nname: a station\n",
                "acquisitions/DAS.L001..RAW.yaml": (
                    "object_type: Acquisition\ndata_category: DAS\n"
                    "interrogator: int_01\n"
                ),
            }
        )
        network = out.networks[0]
        assert network.name == "test network"
        assert [x.code for x in network.stations] == ["STA1"]
        assert network.fiber_arrays[0].name == "first"
        # The reference resolved against the pool the resources dir filled.
        assert out.get_resource("int_01").model == "FI-1"

    def test_json_and_yaml_are_interchangeable(self, tmp_path):
        """One data model stands behind both spellings."""
        as_yaml = dc.inventory(
            write_inventory(
                tmp_path / "yaml_form",
                {
                    "acquisitions/DAS.L001..RAW.yaml": (
                        "object_type: Acquisition\ndata_category: DAS\n"
                        "gauge_length: 4.0\n"
                    )
                },
            )
        )
        as_json = dc.inventory(
            write_inventory(
                tmp_path / "json_form",
                {
                    "acquisitions/DAS.L001..RAW.json": (
                        '{"object_type": "Acquisition", "data_category": "DAS", '
                        '"gauge_length": 4.0}'
                    )
                },
            )
        )
        assert as_yaml.networks == as_json.networks

    def test_yml_suffix(self, make_inventory):
        """The short YAML suffix is the same spelling."""
        out = make_inventory(
            {"acquisitions/DAS.L001..RAW.yml": "object_type: Acquisition\n"}
        )
        assert out.networks[0].fiber_arrays[0].acquisitions[0].code == "RAW"

    def test_envelope_is_optional(self, make_inventory):
        """A bare directory loads with envelope defaults."""
        out = make_inventory(MINIMAL)
        assert out.schema_version == inv.Inventory().schema_version

    def test_envelope_states_the_singletons(self, make_inventory):
        """The envelope is where the document's own facts live."""
        out = make_inventory(
            {
                **MINIMAL,
                "inventory.yaml": (
                    "object_type: Inventory\n"
                    "resource_id: my-inventory\n"
                    "coordinate_reference_system:\n"
                    "  authority: EPSG\n"
                    "  code: '32611'\n"
                    "  coordinate_labels: [easting, northing, elevation]\n"
                ),
            }
        )
        assert out.resource_id == "my-inventory"
        assert out.coordinate_reference_system.code == "32611"

    def test_entity_directory_form(self, make_inventory):
        """An entity is a file until it needs tracks, then a directory."""
        out = make_inventory(
            {
                **MINIMAL,
                "fiber_arrays/DAS.L001/attrs.yaml": (
                    "object_type: FiberArray\nname: from a directory\n"
                ),
            }
        )
        assert out.networks[0].fiber_arrays[0].name == "from a directory"

    def test_non_participating_files_are_ignored(self, make_inventory):
        """Field material lives happily inside the inventory directory."""
        out = make_inventory(
            {
                **MINIMAL,
                "photos/wellhead.jpg": "not text",
                "notes/log.md": "# deployment log\n",
                # A YAML file which declares nothing is not an object.
                "notes/scratch.yaml": "just: a note\n",
                # Nor is one which does not parse at all.
                "notes/broken.yaml": "{[unclosed\n",
                ".hidden/DAS.yaml": "object_type: Network\n",
                "acquisitions/README.txt": "a note beside the objects\n",
            }
        )
        assert [x.code for x in out.networks] == ["DAS"]

    def test_resource_id_may_hold_dots(self, make_inventory):
        """A single-token identity is the whole name, dots and all."""
        out = make_inventory(
            {
                **MINIMAL,
                "resources/cable.01.yaml": "object_type: Cable\nname: a cable\n",
            }
        )
        assert out.get_resource("cable.01").name == "a cable"

    def test_restated_address_may_agree(self, make_inventory):
        """A name may be restated inside the file when the two agree."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001.01.RAW.yaml": (
                    "object_type: Acquisition\ncode: RAW\nlocation_code: '01'\n"
                )
            }
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert (acquisition.location_code, acquisition.code) == ("01", "RAW")


class TestEpochNames:
    """Tests for the ``@`` epoch suffix."""

    def test_date_only_means_midnight_utc(self, make_inventory):
        """A date-only suffix is valid when that precision suffices."""
        out = make_inventory(
            {"acquisitions/DAS.L001..RAW@2024-06-01.yaml": "object_type: Acquisition\n"}
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert acquisition.start_time == np.datetime64("2024-06-01T00:00:00", "ns")

    def test_basic_time_and_fractional_seconds(self, make_inventory):
        """The time portion is ISO basic, since ':' is not a legal filename."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW@2024-05-12T103000.12.yaml": (
                    "object_type: Acquisition\n"
                )
            }
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        expected = np.datetime64("2024-05-12T10:30:00.12", "ns")
        assert acquisition.start_time == expected

    def test_suffix_agrees_with_stated_start(self, make_inventory):
        """A restated start time is checked rather than overridden."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW@2024-06-01.yaml": (
                    "object_type: Acquisition\nstart_time: '2024-06-01'\n"
                )
            }
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert acquisition.start_time == np.datetime64("2024-06-01", "ns")

    def test_epochs_of_one_acquisition(self, make_inventory):
        """Two epochs of one acquisition are two entries under one array."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW.yaml": (
                    "object_type: Acquisition\nend_time: '2024-06-01'\n"
                    "gauge_length: 10.0\n"
                ),
                "acquisitions/DAS.L001..RAW@2024-06-01.yaml": (
                    "object_type: Acquisition\ngauge_length: 5.0\n"
                ),
            }
        )
        acquisitions = out.networks[0].fiber_arrays[0].acquisitions
        assert len(acquisitions) == 2
        assert out.resolve("DAS.L001..RAW", "2024-01-01").acquisition.gauge_length == 10
        assert out.resolve("DAS.L001..RAW", "2025-01-01").acquisition.gauge_length == 5

    def test_acquisition_lands_in_its_array_epoch(self, make_inventory):
        """A child is placed in the container epoch which held it."""
        out = make_inventory(
            {
                "fiber_arrays/DAS.L001.yaml": (
                    "object_type: FiberArray\nname: first\nend_time: '2024-06-01'\n"
                ),
                "fiber_arrays/DAS.L001@2024-06-01.yaml": (
                    "object_type: FiberArray\nname: second\n"
                ),
                "acquisitions/DAS.L001..RAW@2024-07-01.yaml": (
                    "object_type: Acquisition\n"
                ),
            }
        )
        arrays = {x.name: x for x in out.networks[0].fiber_arrays}
        assert not arrays["first"].acquisitions
        assert [x.code for x in arrays["second"].acquisitions] == ["RAW"]


class TestNearMisses:
    """Anything claiming to participate and getting it wrong raises."""

    def test_typo_in_container_name(self, make_inventory):
        """A typo must not quietly load an inventory with no acquisitions."""
        files = {"aquisitions/DAS.L001..RAW.yaml": "object_type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="nothing contains it"):
            make_inventory(files)

    def test_model_file_at_the_root(self, make_inventory):
        """An object at the root is outside every container."""
        files = {**MINIMAL, "DAS.L001.yaml": "object_type: FiberArray\n"}
        with pytest.raises(InvalidInventoryError, match="nothing contains it"):
            make_inventory(files)

    def test_missing_type(self, make_inventory):
        """A file which does not say what it is has not participated."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "data_category: DAS\n"}
        with pytest.raises(InvalidInventoryError, match="declares no object_type"):
            make_inventory(files)

    def test_wrong_container(self, make_inventory):
        """The container checks the declared type rather than supplying it."""
        files = {"fiber_arrays/DAS.L001.yaml": "object_type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="cannot hold"):
            make_inventory(files)

    def test_unknown_type(self, make_inventory):
        """A type which names no model is unknown rather than misfiled."""
        files = {"fiber_arrays/DAS.L001.yaml": "object_type: Telescope\n"}
        with pytest.raises(InvalidInventoryError, match="unknown"):
            make_inventory(files)

    def test_restated_address_disagrees(self, make_inventory):
        """There is never a precedence rule between two spellings."""
        files = {
            "acquisitions/DAS.L001..RAW.yaml": "object_type: Acquisition\ncode: DEC\n"
        }
        with pytest.raises(InvalidInventoryError, match="must agree with the name"):
            make_inventory(files)

    def test_restated_start_time_disagrees(self, make_inventory):
        """The epoch suffix is a restated address, so it must agree."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-06-01.yaml": (
                "object_type: Acquisition\nstart_time: '2024-06-02'\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="must agree with the name"):
            make_inventory(files)

    @pytest.mark.parametrize(
        ("name", "level"),
        # Both address levels, since each is checked by the same loop and
        # only one of them was ever exercised.
        [("DAS.L0_01..RAW.yaml", "fiber_array"), ("D_AS.L001..RAW.yaml", "network")],
    )
    def test_illegal_address_token_names_the_file(self, make_inventory, name, level):
        """The entity a token names is built from every address, not one file."""
        files = {f"acquisitions/{name}": "object_type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match=f"names {level}") as info:
            make_inventory(files)
        # The point of the check is which file to open, so assert the name.
        assert name in str(info.value)

    def test_wrong_token_count(self, make_inventory):
        """An acquisition name is an address of four tokens."""
        files = {"acquisitions/DAS.L001.RAW.yaml": "object_type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="address of 4"):
            make_inventory(files)

    def test_schema_version_outside_the_envelope(self, make_inventory):
        """The envelope versions the document exactly once."""
        files = {
            "acquisitions/DAS.L001..RAW.yaml": (
                "object_type: Acquisition\nschema_version: 1\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="envelope versions"):
            make_inventory(files)

    def test_invalid_field_names_the_file(self, make_inventory):
        """A model error says which file could not be read."""
        files = {
            "acquisitions/DAS.L001..RAW.yaml": (
                "object_type: Acquisition\ngauge_length: not a number\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match=r"DAS\.L001\.\.RAW\.yaml"):
            make_inventory(files)

    def test_unparsable_file_in_a_container(self, make_inventory):
        """A file which claims a place in a container must parse."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "{[unclosed\n"}
        with pytest.raises(InvalidInventoryError, match="Could not parse YAML"):
            make_inventory(files)

    def test_unparsable_json_in_a_container(self, make_inventory):
        """The JSON spelling is held to the same standard."""
        files = {"acquisitions/DAS.L001..RAW.json": "{not json\n"}
        with pytest.raises(InvalidInventoryError, match="Could not parse JSON"):
            make_inventory(files)

    def test_non_mapping_file_in_a_container(self, make_inventory):
        """A document which is not a mapping defines no object."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "- a list\n"}
        with pytest.raises(InvalidInventoryError, match="holds no mapping"):
            make_inventory(files)


class TestOverlookedInput:
    """Input the loader did not inspect, each of which loaded a wrong inventory."""

    def test_object_filed_inside_an_entity_directory(self, make_inventory):
        """An object one level too deep must not be silently dropped."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/misplaced/DAS.L001..RAW.yaml": (
                "object_type: Acquisition\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="nothing contains it"):
            make_inventory(files)

    @pytest.mark.parametrize("suffix", ["YAML", "YML", "JSON"])
    def test_upper_case_suffixes(self, make_inventory, suffix):
        """A case-insensitive filesystem holds one file, not two spellings."""
        text = (
            '{"object_type": "Acquisition"}'
            if suffix == "JSON"
            else "object_type: Acquisition\ndata_category: DAS\n"
        )
        out = make_inventory({f"acquisitions/DAS.L001..RAW.{suffix}": text})
        assert out.networks[0].fiber_arrays[0].acquisitions[0].code == "RAW"

    def test_upper_case_envelope(self, make_inventory):
        """The envelope is found however its suffix is spelled."""
        out = make_inventory(
            {
                **MINIMAL,
                "inventory.YAML": "object_type: Inventory\nresource_id: shouted\n",
            }
        )
        assert out.resource_id == "shouted"

    def test_child_outliving_its_parent_epoch(self, make_inventory):
        """Starting inside an epoch is not enough to belong to it."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "object_type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "fiber_arrays/DAS.L001@2024-06-01.yaml": "object_type: FiberArray\n",
            # Starts inside the first epoch and never ends, so resolution
            # after June would find the second array, which does not hold it.
            "acquisitions/DAS.L001..RAW@2024-05-01.yaml": "object_type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="runs past"):
            make_inventory(files)

    def test_child_ending_exactly_at_the_boundary_fits(self, make_inventory):
        """Half-open on both sides, so the shared instant belongs to neither."""
        out = make_inventory(
            {
                "fiber_arrays/DAS.L001.yaml": (
                    "object_type: FiberArray\nend_time: '2024-06-01'\n"
                ),
                "fiber_arrays/DAS.L001@2024-06-01.yaml": "object_type: FiberArray\n",
                "acquisitions/DAS.L001..RAW@2024-05-01.yaml": (
                    "object_type: Acquisition\nend_time: '2024-06-01'\n"
                ),
            }
        )
        first = out.networks[0].fiber_arrays[0]
        assert [x.code for x in first.acquisitions] == ["RAW"]

    def test_unreadable_envelope_value_names_the_file(self, make_inventory):
        """An envelope error reads like every other error this format raises."""
        files = {
            **MINIMAL,
            "inventory.yaml": "object_type: Inventory\nschema_version: nope\n",
        }
        with pytest.raises(InvalidInventoryError, match="Could not read the envelope"):
            make_inventory(files)

    @pytest.mark.parametrize(
        ("where", "field", "member"),
        [
            ("networks/DAS.yaml", "stations", "code: STA1"),
            ("networks/DAS.yaml", "fiber_arrays", "code: L001"),
            ("fiber_arrays/DAS.L001.yaml", "acquisitions", "code: RAW"),
        ],
    )
    def test_nested_collections_are_refused(self, make_inventory, where, field, member):
        """A file may not state what the directory supplies and would replace."""
        declared = "Network" if where.startswith("networks") else "FiberArray"
        text = f"object_type: {declared}\n{field}:\n  - {member}\n"
        with pytest.raises(InvalidInventoryError, match=field):
            make_inventory({**MINIMAL, where: text})

    def test_child_predating_its_parent_epoch(self, make_inventory):
        """Containment is checked at the near end as well as the far one."""
        files = {
            "networks/DAS@2024-01-01.yaml": "object_type: Network\n",
            # No start of its own, so it claims the unbounded past, which is
            # outside a network beginning in 2024.
            "stations/DAS.STA1.yaml": "object_type: Station\nend_time: '2020-01-01'\n",
        }
        with pytest.raises(InvalidInventoryError, match="starts before"):
            make_inventory(files)

    def test_child_ending_after_its_parent_epoch(self, make_inventory):
        """The far end refuses a real end time, not only an unset one."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "object_type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "fiber_arrays/DAS.L001@2024-06-01.yaml": "object_type: FiberArray\n",
            "acquisitions/DAS.L001..RAW@2024-05-01.yaml": (
                "object_type: Acquisition\nend_time: '2024-07-01'\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="runs past"):
            make_inventory(files)

    def test_symlink_pointing_at_the_inventory(self, tmp_path):
        """Walking a loop would find the inventory's own files uncontained."""
        root = write_inventory(tmp_path / "looped", MINIMAL)
        (root / "photos").mkdir()
        os.symlink(root, root / "photos" / "loop")
        # Without skipping the link this reports DAS.L001..RAW.yaml, reached
        # through it, as an object nothing contains.
        assert dc.inventory(root).networks[0].code == "DAS"

    def test_envelope_stating_only_its_type(self, tmp_path):
        """An envelope says the directory is an inventory, whatever it holds."""
        root = write_inventory(
            tmp_path / "bare", {"inventory.yaml": "object_type: Inventory\n"}
        )
        assert not dc.inventory(root).networks

    def test_json_inventory_without_pyyaml(self, tmp_path, monkeypatch):
        """A JSON inventory loads past whatever YAML lies beside it.

        PyYAML is optional, and the tests which need it skip without it, so
        this pins the JSON-only path here rather than leaving it to a
        minimal install nothing in this file would exercise.
        """

        def no_yaml(name, **kwargs):
            raise MissingOptionalDependencyError(f"no {name}")

        monkeypatch.setattr(loader, "optional_import", no_yaml)
        root = write_inventory(
            tmp_path / "json_only",
            {
                "acquisitions/DAS.L001..RAW.json": '{"object_type": "Acquisition"}',
                # Field material the loader must step over without reading.
                "notes/log.yaml": "note: a deployment log\n",
            },
        )
        out = dc.inventory(root)
        assert out.networks[0].fiber_arrays[0].acquisitions[0].code == "RAW"

    def test_hidden_object_file_in_a_container(self, make_inventory):
        """A resource id may hold a dot, but a hidden file names no entity."""
        files = {**MINIMAL, "resources/.cable.yaml": "object_type: Cable\n"}
        with pytest.raises(InvalidInventoryError, match="hidden"):
            make_inventory(files)

    def test_epoch_finer_than_a_nanosecond(self, make_inventory):
        """A name stating more precision than is kept would load as another instant."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-05-12T103000.1234567899.yaml": (
                "object_type: Acquisition\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="finer than the nanosecond"):
            make_inventory(files)

    def test_unquoted_dates(self, make_inventory):
        """YAML reads an unquoted date as a date, which is how one is written."""
        out = make_inventory(
            {
                "fiber_arrays/DAS.L001.yaml": (
                    "object_type: FiberArray\nstart_time: 2024-01-01\n"
                    "end_time: 2024-07-01\n"
                ),
                "acquisitions/DAS.L001..RAW@2024-02-01.yaml": (
                    "object_type: Acquisition\nend_time: 2024-03-01\n"
                ),
            }
        )
        array = out.networks[0].fiber_arrays[0]
        assert array.end_time == np.datetime64("2024-07-01", "ns")
        assert array.acquisitions[0].end_time == np.datetime64("2024-03-01", "ns")

    def test_type_which_is_not_a_name(self, make_inventory):
        """A type which is not a name at all names no model either."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "object_type: [Acquisition]\n"}
        with pytest.raises(InvalidInventoryError, match="declares no object_type"):
            make_inventory(files)


class TestOneIdentityOneSpelling:
    """Identity is unique per container regardless of spelling."""

    def test_two_extensions(self, make_inventory):
        """The same name with two extensions is one identity spelled twice."""
        files = {
            "resources/cable_01.yaml": "object_type: Cable\n",
            "resources/cable_01.json": '{"object_type": "Cable"}',
        }
        with pytest.raises(InvalidInventoryError, match="two extensions"):
            make_inventory(files)

    # Where case is folded, writing the second name replaces the first
    # rather than joining it, so the collision this refuses cannot be built.
    # That is the rule holding rather than failing: the guard exists so an
    # inventory authored where both names fit does not quietly lose one when
    # it is copied somewhere they do not.
    @pytest.mark.skipif(FOLDS_CASE, reason="this filesystem holds one of the two")
    def test_case_only_difference(self, make_inventory):
        """A case-insensitive filesystem could not hold both."""
        files = {
            "fiber_arrays/DAS.L001.yaml": "object_type: FiberArray\n",
            "fiber_arrays/das.l001.yaml": "object_type: FiberArray\n",
        }
        with pytest.raises(InvalidInventoryError, match="differ only by case"):
            make_inventory(files)

    def test_case_only_difference_is_named_as_such(self, tmp_path):
        """The reason two entries collide, on every filesystem.

        The test above cannot run where case is folded, which is every
        Windows and most macOS checkouts, so the explaining half of the
        rule is pinned here instead.
        """
        first, second = tmp_path / "DAS.L001.yaml", tmp_path / "das.l001.yaml"
        assert "differ only by case" in loader._collide(first, second)

    def test_file_and_directory(self, make_inventory):
        """Both spellings of one identity at once raise."""
        files = {
            "fiber_arrays/DAS.L001.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
        }
        with pytest.raises(InvalidInventoryError, match="file and a directory"):
            make_inventory(files)

    def test_envelope_spelled_twice(self, make_inventory):
        """The envelope is one file however it is spelled."""
        files = {
            **MINIMAL,
            "inventory.yaml": "object_type: Inventory\n",
            "inventory.json": '{"object_type": "Inventory"}',
        }
        with pytest.raises(InvalidInventoryError, match="more than once"):
            make_inventory(files)

    def test_two_names_for_one_epoch(self, make_inventory):
        """Epoch-name uniqueness is temporal rather than textual."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-06-01.yaml": "object_type: Acquisition\n",
            "acquisitions/DAS.L001..RAW@2024-06-01T000000.yaml": (
                "object_type: Acquisition\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="overlap in time"):
            make_inventory(files)

    def test_overlapping_epochs_of_one_entity(self, make_inventory):
        """Two epochs of one entity may not overlap, not merely coincide."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-01-01.yaml": (
                "object_type: Acquisition\nend_time: '2024-08-01'\n"
            ),
            "acquisitions/DAS.L001..RAW@2024-06-01.yaml": "object_type: Acquisition\n",
        }
        # Both files are named, since the entity alone would not say which.
        with pytest.raises(InvalidInventoryError, match="RAW@2024-01-01"):
            make_inventory(files)


class TestEntityDirectories:
    """Tests for the directory spelling of one entity."""

    def test_missing_attrs_file(self, make_inventory):
        """A directory with no attrs file states nothing."""
        files = {**MINIMAL, "fiber_arrays/DAS.L001/notes.txt": "hi\n"}
        with pytest.raises(InvalidInventoryError, match="holds no attrs file"):
            make_inventory(files)

    def test_stray_object_file(self, make_inventory):
        """An entity directory's object file is named attrs."""
        files = {"fiber_arrays/DAS.L001/array.yaml": "object_type: FiberArray\n"}
        with pytest.raises(InvalidInventoryError, match="holds only its attrs"):
            make_inventory(files)

    def test_field_note_beside_an_attrs_file(self, make_inventory):
        """A note which happens to be YAML is not a misfiled object."""
        out = make_inventory(
            {
                **MINIMAL,
                "fiber_arrays/DAS.L001/attrs.yaml": (
                    "object_type: FiberArray\nname: from a directory\n"
                ),
                # Declares nothing, so it participates in nothing -- exactly
                # as it would one directory deeper.
                "fiber_arrays/DAS.L001/notes.yaml": "site: wellhead\n",
            }
        )
        assert out.networks[0].fiber_arrays[0].name == "from a directory"

    def test_attrs_spelled_twice(self, make_inventory):
        """One identity is spelled once inside the directory too."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/attrs.json": '{"object_type": "FiberArray"}',
        }
        with pytest.raises(InvalidInventoryError, match="more than once"):
            make_inventory(files)


class TestIgnoredEntries:
    """What a container and an entity directory step over."""

    def test_hidden_files_in_a_container(self, make_inventory, tmp_path):
        """A dotfile is the filesystem's business, not the inventory's."""
        root = tmp_path / "with_dotfiles"
        write_inventory(root, MINIMAL)
        (root / "acquisitions" / ".DS_Store").write_text("junk\n")
        assert dc.inventory(root).networks[0].code == "DAS"

    def test_hidden_and_foreign_entries_in_an_entity(self, tmp_path):
        """Field material lives happily inside an entity directory too."""
        root = tmp_path / "with_material"
        write_inventory(
            root,
            {
                **MINIMAL,
                "fiber_arrays/DAS.L001/attrs.yaml": (
                    "object_type: FiberArray\nname: from a directory\n"
                ),
                "fiber_arrays/DAS.L001/photos/wellhead.jpg": "not text",
                "fiber_arrays/DAS.L001/notes.txt": "a note\n",
            },
        )
        (root / "fiber_arrays" / "DAS.L001" / ".DS_Store").write_text("junk\n")
        out = dc.inventory(root)
        assert out.networks[0].fiber_arrays[0].name == "from a directory"

    def test_unreadable_file_names_itself(self, tmp_path):
        """A file which cannot be read at all says which one it was."""
        root = tmp_path / "unreadable"
        write_inventory(root, MINIMAL)
        (root / "acquisitions" / "DAS.L001..DEC.yaml").write_bytes(b"\xff\xfe\x00")
        with pytest.raises(InvalidInventoryError, match=r"DAS\.L001\.\.DEC\.yaml"):
            dc.inventory(root)


class TestSeams:
    """A table lives where the entity whose tracks it holds does."""

    def test_track_table_outside_an_entity_directory(self, make_inventory):
        """A table lives beside the attrs file of the entity it describes."""
        files = {**MINIMAL, "fiber_arrays/geometry.csv": "distance\n0\n"}
        with pytest.raises(InvalidInventoryError, match="outside an entity"):
            make_inventory(files)


class TestEpochTimestamps:
    """Timestamps in names are UTC and ISO 8601 basic."""

    @pytest.mark.parametrize(
        "stamp", ["2024-13-01", "2024-06", "not-a-time", "2024-06-01T1030"]
    )
    def test_malformed(self, make_inventory, stamp):
        """A name which claims an epoch and gets it wrong raises."""
        files = {
            f"acquisitions/DAS.L001..RAW@{stamp}.yaml": "object_type: Acquisition\n"
        }
        with pytest.raises(InvalidInventoryError, match="epoch timestamp"):
            make_inventory(files)

    @pytest.mark.parametrize(
        "stamp", ["2024-06-01Z", "2024-05-12T103000Z", "2024-05-12T103000+0100"]
    )
    def test_timezone_designator(self, make_inventory, stamp):
        """Naive means UTC, so a designator is refused rather than ignored."""
        files = {
            f"acquisitions/DAS.L001..RAW@{stamp}.yaml": "object_type: Acquisition\n"
        }
        with pytest.raises(InvalidInventoryError, match="timezone designator"):
            make_inventory(files)

    def test_negative_offset(self, make_inventory):
        """An offset in the time portion is a designator, not a malformed time."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-05-12T103000-0600.yaml": (
                "object_type: Acquisition\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="timezone designator"):
            make_inventory(files)

    # The last is on the final representable day, so the guard cannot be
    # passing by comparing dates alone.
    @pytest.mark.parametrize(
        "stamp", ["2500-01-01", "1000-01-01", "2262-04-12", "2262-04-11T235000"]
    )
    def test_outside_the_representable_range(self, make_inventory, stamp):
        """A nanosecond timestamp wraps silently, so the name is refused."""
        files = {
            f"acquisitions/DAS.L001..RAW@{stamp}.yaml": "object_type: Acquisition\n"
        }
        with pytest.raises(InvalidInventoryError, match="outside the range"):
            make_inventory(files)

    def test_the_last_representable_instant(self, make_inventory):
        """The check refuses what wraps without refusing what does not."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW@2262-04-11T234716.yaml": (
                    "object_type: Acquisition\n"
                )
            }
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert acquisition.start_time == np.datetime64("2262-04-11T23:47:16", "ns")

    def test_two_epoch_markers(self, make_inventory):
        """A name carries at most one epoch."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-06-01@2024-07-01.yaml": (
                "object_type: Acquisition\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="more than one"):
            make_inventory(files)

    def test_empty_epoch(self, make_inventory):
        """A trailing marker names no epoch."""
        files = {"acquisitions/DAS.L001..RAW@.yaml": "object_type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="names no epoch"):
            make_inventory(files)

    def test_resources_have_no_epochs(self, make_inventory):
        """A resource is not time-ranged, so its name states no epoch."""
        files = {
            **MINIMAL,
            "resources/int_01@2024-06-01.yaml": "object_type: Interrogator\n",
        }
        with pytest.raises(InvalidInventoryError, match="have none"):
            make_inventory(files)


class TestEpochPlacement:
    """A child is placed in the container epoch which held it."""

    def test_child_outside_every_epoch(self, make_inventory):
        """A child falling in no epoch of its container is misfiled."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "object_type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "acquisitions/DAS.L001..RAW@2024-07-01.yaml": "object_type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="0 epochs effective"):
            make_inventory(files)

    def test_ambiguous_child(self, make_inventory):
        """An unset start beside several container epochs is ambiguous."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "object_type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "fiber_arrays/DAS.L001@2024-06-01.yaml": "object_type: FiberArray\n",
            "acquisitions/DAS.L001..RAW.yaml": "object_type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="2 epochs effective at any"):
            make_inventory(files)

    def test_station_placed_in_a_network_epoch(self, make_inventory):
        """Networks epoch like everything else which is time-ranged."""
        out = make_inventory(
            {
                "networks/DAS.yaml": "object_type: Network\nend_time: '2024-06-01'\n",
                "networks/DAS@2024-06-01.yaml": "object_type: Network\nname: later\n",
                "stations/DAS.STA1@2024-07-01.yaml": "object_type: Station\n",
            }
        )
        by_name = {x.name: x for x in out.networks}
        assert not by_name[""].stations
        assert [x.code for x in by_name["later"].stations] == ["STA1"]


class TestEnvelope:
    """The envelope holds the document-level singletons and nothing else."""

    def test_wrong_type(self, make_inventory):
        """The envelope declares its type under the same rule as any file."""
        files = {**MINIMAL, "inventory.yaml": "object_type: Network\n"}
        with pytest.raises(InvalidInventoryError, match="envelope declares"):
            make_inventory(files)

    @pytest.mark.parametrize("field", ["networks", "resources"])
    def test_collections_are_refused(self, make_inventory, field):
        """The collections live in the directory structure."""
        files = {**MINIMAL, "inventory.yaml": f"object_type: Inventory\n{field}: []\n"}
        with pytest.raises(InvalidInventoryError, match="directory structure"):
            make_inventory(files)

    def test_unknown_field_names_the_file(self, make_inventory):
        """A typo in the envelope says which file states it."""
        files = {
            **MINIMAL,
            "inventory.yaml": "object_type: Inventory\nschema_verison: 1\n",
        }
        with pytest.raises(InvalidInventoryError, match=r"inventory\.yaml"):
            make_inventory(files)


class TestFactory:
    """The public factory routes a directory to this loader."""

    def test_directory(self, tmp_path):
        """A directory loads through the authoring format."""
        root = write_inventory(tmp_path / "as_directory", MINIMAL)
        assert isinstance(dc.inventory(root), inv.Inventory)

    def test_yaml_file_still_works(self, tmp_path):
        """A file keeps going to the single-file reader."""
        path = tmp_path / "inv.yaml"
        dc.inventory().to_yaml(path)
        assert isinstance(dc.inventory(path), inv.Inventory)

    def test_directory_which_holds_no_inventory(self, tmp_path):
        """A mistyped path must not read as an inventory which is empty."""
        root = tmp_path / "not_an_inventory"
        (root / "photos").mkdir(parents=True)
        with pytest.raises(InvalidInventoryError, match="holds no inventory"):
            dc.inventory(root)

    def test_empty_container_is_an_empty_inventory(self, tmp_path):
        """A container, even an empty one, says the directory is an inventory."""
        root = tmp_path / "empty"
        (root / "acquisitions").mkdir(parents=True)
        assert not dc.inventory(root).networks


class TestModelAssumptions:
    """Pin what this loader assumes about the models it builds."""

    def test_only_union_members_declare_the_tag_as_a_field(self):
        """The tag is a real field only where it discriminates a union."""
        checked = []
        for name, container in loader._CONTAINERS.items():
            for model in container.models:
                assert (TAG_FIELD in model.model_fields) == (name == "resources")
                checked.append(model)
        assert len(checked) == 9

    def test_every_model_reads_its_own_tag(self):
        """
        The loader leaves the tag in the data for the model to check.

        Before every model read its own, the loader had to pop the tag for
        the ones which do not declare it, since they forbid extra fields.
        """
        checked = []
        for container in loader._CONTAINERS.values():
            for model in container.models:
                # The addressed models are named by their file, so a code
                # is the one thing they will not default.
                fields = {"code": "X"} if "code" in model.model_fields else {}
                assert isinstance(model(**fields, **{TAG_FIELD: model.__name__}), model)
                with pytest.raises(ValidationError):
                    model(**fields, **{TAG_FIELD: "Inventory"})
                checked.append(model)
        assert len(checked) == 9

    def test_epoch_bearing_models_are_time_ranged(self):
        """Only a time-ranged model can hold the epoch a name states."""
        checked = []
        for name, container in loader._CONTAINERS.items():
            ranged = all(issubclass(x, TimeRangedModel) for x in container.models)
            # The property decides both who may be named with an @ and who
            # is checked for two names of one epoch, so pin it to the models.
            assert container.epochs == ranged == (name != "resources")
            checked.append(name)
        assert len(checked) == len(loader._CONTAINERS) == 5

    def test_inventory_models_refuse_unknown_input(self):
        """
        An inventory model refuses a key it does not declare.

        Which is why the tag has to be consumed rather than ignored, and is
        checked on the concrete models rather than on the base, since a
        subclass may override the config (PatchAttrs does, in the other
        hierarchy).
        """
        assert InventoryModel.model_config["extra"] == "forbid"
        checked = []
        for container in loader._CONTAINERS.values():
            for model in container.models:
                fields = {"code": "X"} if "code" in model.model_fields else {}
                with pytest.raises(ValidationError, match="not permitted"):
                    model(**fields, nonsense_key=1)
                checked.append(model)
        assert len(checked) == 9


# A fiber array whose optical path states every track shape at once. The
# path is 1000.1 m long, so every interval below sits inside it.
TRACKS = {
    "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\nname: array\n",
    "fiber_arrays/DAS.L001/path/attrs.yaml": "object_type: OpticalPath\nname: main\n",
    "fiber_arrays/DAS.L001/path/optical_components.csv": (
        "sequence,object_type,optical_length,name,fiber_number,fiber_color\n"
        "2,Splice,0.1,splice 1,,\n"
        "1,FiberSegment,1000.0,fiber 1,1,blue\n"
    ),
    "fiber_arrays/DAS.L001/path/coupling.csv": (
        "start_distance,end_distance,coupling_type,description\n"
        "0,340,conduit,\n"
        "340,355,trench,backfilled\n"
    ),
    "fiber_arrays/DAS.L001/path/annotations.csv": (
        "start_distance,end_distance,group,value\n"
        "0,340,rock_type,granite\n"
        "0,120,noisy,true\n"
        "120,340,frost_depth,1.2\n"
    ),
    "fiber_arrays/DAS.L001/path/geometry.csv": (
        "segment,distance,longitude,latitude,elevation\n"
        "S100,100.0,-117.0,40.0,687.0\n"
        "S100,102.0,-117.1,40.1,685.0\n"
    ),
}


def one_path(inventory):
    """Return the single optical path of a loaded inventory."""
    return inventory.networks[0].fiber_arrays[0].optical_paths[0]


class TestTrackTables:
    """Tests for the CSV half of the format."""

    def test_every_shape_at_once(self, make_inventory):
        """One path states all four tables, and each arrives whole."""
        path = one_path(make_inventory({**MINIMAL, **TRACKS}))
        assert [x.name for x in path.optical_components] == ["fiber 1", "splice 1"]
        assert [x.coupling_type for x in path.coupling] == ["conduit", "trench"]
        assert {x.group for x in path.annotations} == {
            "rock_type",
            "noisy",
            "frost_depth",
        }
        assert [x.name for x in path.geometry] == ["S100"]

    def test_rows_are_read_in_the_order_they_state(self, make_inventory):
        """Sequence decides the order, never row position."""
        path = one_path(make_inventory({**MINIMAL, **TRACKS}))
        # The file lists the splice first; sequence puts it second.
        assert [x.object_type for x in path.optical_components] == [
            "FiberSegment",
            "Splice",
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("granite", "granite"),
            ("true", True),
            ("false", False),
            ("1.2", 1.2),
            ("2", 2),
            # Integral, and how a spreadsheet writes a large one. Read from
            # the text this raises; read from the number it does not.
            ("1e3", 1000),
        ],
    )
    def test_a_value_is_read_as_its_text_states(self, make_inventory, text, expected):
        """A CSV has no types, so an annotation's value is read by content."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/annotations.csv": (
                f"start_distance,end_distance,group,value\n0,340,g,{text}\n"
            ),
        }
        value = one_path(make_inventory(files)).annotations[0].value
        assert value == expected and isinstance(value, type(expected))

    def test_a_group_holding_two_kinds(self, make_inventory):
        """A group's kind decides its shape, so one group holds one kind."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/annotations.csv": (
                "start_distance,end_distance,group,value\n"
                "0,120,zone,north\n"
                "120,340,zone,true\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="one group holds one kind"):
            make_inventory(files)

    def test_an_empty_cell_is_unset(self, make_inventory):
        """An empty cell means unset, never an empty string."""
        path = one_path(make_inventory({**MINIMAL, **TRACKS}))
        splice = path.optical_components[1]
        # The splice row leaves the fiber columns empty; they are a
        # FiberSegment's fields, and it is not one.
        assert splice.description == ""
        assert path.coupling[0].description == ""
        assert path.coupling[1].description == "backfilled"

    def test_a_geometry_names_its_axes_the_way_its_frame_does(self, make_inventory):
        """Coordinates are stored canonically, whatever the CRS calls them."""
        path = one_path(make_inventory({**MINIMAL, **TRACKS}))
        geometry = path.geometry[0]
        assert geometry.distance == (100.0, 102.0)
        # longitude, latitude, elevation are axes 0, 1, 2 of the default CRS.
        assert geometry.coordinates[0] == (-117.0, 40.0, 687.0)

    def test_a_declared_frame_renames_the_axes(self, make_inventory):
        """The envelope decides which headers a geometry table may state."""
        files = {
            **MINIMAL,
            **TRACKS,
            "inventory.yaml": (
                "object_type: Inventory\n"
                "coordinate_reference_system:\n"
                "  authority: EPSG\n"
                "  code: '32611'\n"
                "  coordinate_labels: [x, y, elevation]\n"
                "  units: [meter, meter, meter]\n"
            ),
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                "segment,distance,x,y,elevation\n"
                "S100,100.0,2562048.25,1137365.53,687.0\n"
                "S100,102.0,2562048.17,1137365.63,685.0\n"
            ),
        }
        geometry = one_path(make_inventory(files)).geometry[0]
        assert geometry.coordinates[0] == (2562048.25, 1137365.53, 687.0)

    def test_a_header_the_frame_does_not_declare(self, make_inventory):
        """A geometry header disagreeing with the frame raises."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                "segment,distance,x,y,z\nS100,100.0,1.0,2.0,3.0\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="its frame declares"):
            make_inventory(files)

    def test_a_point_missing_an_axis(self, make_inventory):
        """A coordinate states every axis its frame declares."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                "segment,distance,longitude,latitude,elevation\n"
                "S100,100.0,-117.0,40.0,687.0\n"
                "S100,102.0,-117.1,,685.0\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="leaves latitude empty"):
            make_inventory(files)

    def test_a_single_object_table(self, make_inventory):
        """A map has no grouping column: every point belongs to the one map."""
        files = {
            "acquisitions/DAS.L001.02.DEC/attrs.yaml": (
                "object_type: Acquisition\nspatial_interval: 1.0\n"
            ),
            "acquisitions/DAS.L001.02.DEC/distance_map.csv": (
                "channel,distance\n512,500.0\n1710,1698.0\n"
            ),
        }
        out = make_inventory(files)
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert acquisition.distance_map.channel == (512.0, 1710.0)
        assert acquisition.channel_to_distance([512])[0] == 500.0

    def test_a_stem_naming_no_attribute(self, make_inventory):
        """A table is matched to the model by name, so a typo is a typo."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/geometrys.csv": "segment,distance\nS,0\n",
        }
        with pytest.raises(InvalidInventoryError, match="names no attribute"):
            make_inventory(files)

    def test_a_stem_naming_a_field_which_is_not_row_shaped(self, make_inventory):
        """Only an attribute rows can build may be stated as a table."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/name.csv": "a,b\n1,2\n",
        }
        with pytest.raises(InvalidInventoryError, match="does not read as a table"):
            make_inventory(files)

    def test_a_row_shaped_attribute_with_no_table(self, make_inventory):
        """Station.channels is row-shaped and still has no table form.

        The message must not tell the author otherwise, which is what it
        did while it said "not a row-shaped attribute".
        """
        files = {
            "stations/DAS.STA1/attrs.yaml": "object_type: Station\n",
            "stations/DAS.STA1/channels.csv": "code,location_code\nHHZ,00\n",
        }
        assert "channels" in inv.Station.model_fields
        with pytest.raises(InvalidInventoryError, match="does not read as a table"):
            make_inventory(files)

    def test_stated_inline_and_as_a_table(self, make_inventory):
        """One fact is spelled once."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: main\n"
                "coupling:\n"
                "  - start_distance: 0.0\n"
                "    end_distance: 10.0\n"
                "    coupling_type: trench\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="spelled once"):
            make_inventory(files)

    def test_a_table_missing_the_column_it_is_read_by(self, make_inventory):
        """Components are placed by sequence, so a table without one raises."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "object_type,optical_length,name\nFiberSegment,1000.0,fiber 1\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="no sequence column"):
            make_inventory(files)

    def test_a_column_stated_twice(self, make_inventory):
        """A repeated header means two columns claim one field."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/coupling.csv": (
                "start_distance,end_distance,coupling_type,coupling_type\n"
                "0,340,conduit,trench\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="more than once"):
            make_inventory(files)


class TestPathEpochs:
    """`path` is the one reserved container stem, and it holds a lineage."""

    def test_a_bare_path_is_the_first_epoch(self, make_inventory):
        """The bare directory is the blank location, starting where the array does."""
        path = one_path(make_inventory({**MINIMAL, **TRACKS}))
        assert path.location_code == ""
        assert pd.isnull(path.start_time)

    def test_a_location_joins_the_stem_with_a_dot(self, make_inventory):
        """Each location code is its own lineage."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path.00/attrs.yaml": (
                "object_type: OpticalPath\nname: first\n"
            ),
            "fiber_arrays/DAS.L001/path.01/attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        paths = make_inventory(files).networks[0].fiber_arrays[0].optical_paths
        assert {x.location_code: x.name for x in paths} == {
            "00": "first",
            "01": "second",
        }

    def test_an_epoch_ends_where_the_next_begins(self, make_inventory):
        """Sorted by start, each epoch runs until its successor."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: first\n"
            ),
            "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        paths = {
            x.name: x
            for x in make_inventory(files).networks[0].fiber_arrays[0].optical_paths
        }
        boundary = np.datetime64("2024-05-12T10:30:00", "ns")
        # The first epoch was left open and the directory closed it.
        assert paths["first"].end_time == boundary
        assert paths["second"].start_time == boundary
        assert pd.isnull(paths["second"].end_time)

    def test_an_epoch_may_end_early(self, make_inventory):
        """A dark interval: an epoch states an end before its successor."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: first\nend_time: 2024-01-01\n"
            ),
            "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        paths = {
            x.name: x
            for x in make_inventory(files).networks[0].fiber_arrays[0].optical_paths
        }
        assert paths["first"].end_time == np.datetime64("2024-01-01", "ns")

    def test_an_epoch_may_not_end_late(self, make_inventory):
        """An epoch cannot claim time its successor already holds."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: first\nend_time: 2025-01-01\n"
            ),
            "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        with pytest.raises(
            InvalidInventoryError, match="after the epoch which follows"
        ):
            make_inventory(files)

    def test_two_names_for_one_epoch(self, make_inventory):
        """Epoch-name uniqueness is temporal rather than textual."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path@2024-06-01/attrs.yaml": (
                "object_type: OpticalPath\nname: first\n"
            ),
            "fiber_arrays/DAS.L001/path@2024-06-01T000000/attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="two spellings of one epoch"):
            make_inventory(files)

    def test_a_restated_location_must_agree(self, make_inventory):
        """The directory name is an address like any other."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path.00/attrs.yaml": (
                "object_type: OpticalPath\nlocation_code: '01'\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="must agree with the name"):
            make_inventory(files)

    def test_a_path_directory_declaring_something_else(self, make_inventory):
        """The declared type must agree with the container."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": "object_type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="holds an OpticalPath"):
            make_inventory(files)

    def test_a_path_where_nothing_holds_one(self, make_inventory):
        """An acquisition has no optical paths, so it cannot hold the epoch."""
        files = {
            "acquisitions/DAS.L001..RAW/attrs.yaml": "object_type: Acquisition\n",
            "acquisitions/DAS.L001..RAW/path/attrs.yaml": "object_type: OpticalPath\n",
        }
        with pytest.raises(InvalidInventoryError, match="which holds none"):
            make_inventory(files)

    def test_paths_stated_inline_and_as_directories(self, make_inventory):
        """One fact is spelled once."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": (
                "object_type: FiberArray\noptical_paths:\n  - name: inline\n"
            ),
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: from a directory\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="spelled once"):
            make_inventory(files)

    def test_an_object_misfiled_inside_a_path(self, make_inventory):
        """A path epoch holds its own attrs and tracks, like any entity."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": "object_type: OpticalPath\n",
            "fiber_arrays/DAS.L001/path/DAS.L002.yaml": "object_type: FiberArray\n",
        }
        with pytest.raises(InvalidInventoryError, match="holds only its attrs"):
            make_inventory(files)

    def test_a_path_resolves_through_the_inventory(self, make_inventory):
        """A directory-built path is reached the way any other is."""
        out = make_inventory({**MINIMAL, **TRACKS})
        resolved = out.resolve("DAS.L001..RAW")
        assert resolved.optical_path.name == "main"
        assert len(resolved.optical_path.coupling) == 2


class TestUnreadableTables:
    """A table which claims a place in the format must be readable."""

    def test_a_table_which_is_not_text(self, tmp_path):
        """A file which cannot be read at all says which one it was."""
        root = write_inventory(tmp_path / "binary", {**MINIMAL, **TRACKS})
        (root / "fiber_arrays/DAS.L001/path/coupling.csv").write_bytes(b"\xff\xfe\x00")
        with pytest.raises(InvalidInventoryError, match="Could not read"):
            dc.inventory(root)

    def test_a_table_with_no_columns(self, make_inventory):
        """An empty file states no track."""
        files = {**MINIMAL, **TRACKS, "fiber_arrays/DAS.L001/path/coupling.csv": ""}
        with pytest.raises(InvalidInventoryError, match="no columns"):
            make_inventory(files)

    def test_a_non_numeric_order_column(self, make_inventory):
        """Rows are placed by their order column, so it must be a number."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "sequence,object_type,optical_length,name\n"
                "first,FiberSegment,1000.0,fiber 1\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="non-numeric sequence"):
            make_inventory(files)

    def test_a_column_no_row_states(self, make_inventory):
        """A column every row leaves empty states nothing, and is not empty."""
        files = {
            "acquisitions/DAS.L001.02.DEC/attrs.yaml": (
                "object_type: Acquisition\nspatial_interval: 1.0\n"
            ),
            "acquisitions/DAS.L001.02.DEC/distance_map.csv": (
                "channel,instrument_distance,distance\n512,,500.0\n1710,,1698.0\n"
            ),
        }
        acquisition = make_inventory(files).networks[0].fiber_arrays[0].acquisitions[0]
        # Unset, rather than a tuple of nothing, which the model would refuse.
        assert acquisition.distance_map.instrument_distance is None

    def test_an_annotation_stating_no_value(self, make_inventory):
        """A membership group's value is its default, not a parsed cell."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/annotations.csv": (
                "start_distance,end_distance,group,value\n0,120,noisy,\n"
            ),
        }
        annotation = one_path(make_inventory(files)).annotations[0]
        assert annotation.value is True

    def test_a_path_restating_a_start_which_disagrees(self, make_inventory):
        """A path directory's name is a restated address like any other."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path@2024-06-01/attrs.yaml": (
                "object_type: OpticalPath\nstart_time: 2024-06-02\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="must agree with the name"):
            make_inventory(files)

    def test_an_envelope_declaring_an_unreadable_frame(self, make_inventory):
        """The frame builds once, with the envelope which states it.

        Which is why the loader rebuilds it without a guard of its own: a
        second check there could only report what this one already has.
        """
        files = {
            **MINIMAL,
            "inventory.yaml": (
                "object_type: Inventory\n"
                "coordinate_reference_system:\n"
                "  coordinate_labels: [x, x, z]\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="Could not read the envelope"):
            make_inventory(files)

    @pytest.mark.parametrize("row", ["0,340,conduit,extra", "0,340"])
    def test_a_row_which_is_not_its_header_wide(self, make_inventory, row):
        """A row states one cell per column or it is not a row.

        Pandas refuses neither: a surplus cell silently pushes the first
        column into the index, so `0,340,conduit,extra` would load as
        start_distance=340, end_distance=conduit, coupling_type=extra --
        every value one field left of what it says.
        """
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/coupling.csv": (
                f"start_distance,end_distance,coupling_type\n{row}\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="its header names 3 columns"):
            make_inventory(files)

    def test_a_sequence_which_places_two_rows_alike(self, make_inventory):
        """Components tile the path, so no two may claim one place."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "sequence,object_type,optical_length,name\n"
                "1,FiberSegment,100.0,a\n"
                "1,Splice,0.1,b\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="does not say which row"):
            make_inventory(files)

    def test_a_cell_which_does_not_pertain_to_its_row(self, make_inventory):
        """The model refuses a field its own type does not declare."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "sequence,object_type,optical_length,name,fiber_color\n"
                "1,Splice,0.1,b,blue\n"
            ),
        }
        # A splice has no fiber colour; only a segment does.
        with pytest.raises(InvalidInventoryError, match="Could not read OpticalPath"):
            make_inventory(files)


# Which model declares each table's attribute. Spelled out rather than
# searched for, so a table moving between models is a change here.
DECLARED_BY = {
    "optical_components": inv.OpticalPath,
    "coupling": inv.OpticalPath,
    "annotations": inv.OpticalPath,
    "geometry": inv.OpticalPath,
    "distance_map": inv.Acquisition,
}


class TestTableRegistry:
    """The table registry must stay pinned to the models."""

    def test_every_stem_is_a_field_of_its_model(self):
        """A stem which names no field could never be authored."""
        assert set(loader._TABLES) == set(DECLARED_BY)
        for stem, model in DECLARED_BY.items():
            assert stem in model.model_fields

    def test_a_grouping_or_ordering_column_is_not_a_stem(self):
        """The columns a table is read by are structural, not attributes."""
        for stem, table in loader._TABLES.items():
            for column in (table.group, table.order):
                assert column is None or column not in loader._TABLES
            assert stem  # every key names something

    def test_only_a_placing_table_drops_its_order_column(self):
        """A column dropped where it is not scaffolding would vanish unseen."""
        placing = {k for k, v in loader._TABLES.items() if v.places}
        assert placing == {"optical_components"}
        # And that column is the one the model has no field for.
        assert "sequence" not in inv.OpticalPath.model_fields


class TestSilentlyLostRows:
    """A row which states a place must keep it, or say it has none."""

    def test_a_blank_grouping_cell(self, make_inventory):
        """A null grouping key drops its row from every group, unasked."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                "segment,distance,longitude,latitude,elevation\n"
                "S100,100.0,-117.0,40.0,687.0\n"
                ",102.0,-117.1,40.1,685.0\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="state no place"):
            make_inventory(files)

    def test_every_row_blank_in_the_grouping_column(self, make_inventory):
        """The whole track would otherwise vanish and the path load empty."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                "segment,distance,longitude,latitude,elevation\n"
                ",100.0,-117.0,40.0,687.0\n"
                ",102.0,-117.1,40.1,685.0\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="state no place"):
            make_inventory(files)

    def test_a_single_blank_ordering_cell(self, make_inventory):
        """One blank is as unplaced as two, though only two ever collide."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "sequence,object_type,optical_length,name\n"
                "1,FiberSegment,100.0,a\n"
                ",Splice,0.1,b\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="state no place"):
            make_inventory(files)

    def test_a_column_some_points_state_and_others_do_not(self, make_inventory):
        """Columns become parallel arrays, so a gap would pair the unpaired."""
        files = {
            "acquisitions/DAS.L001.02.DEC/attrs.yaml": (
                "object_type: Acquisition\nspatial_interval: 1.0\n"
            ),
            # Channel 5 states no distance and 100 states no channel; read
            # column by column they would pair up as though they had.
            "acquisitions/DAS.L001.02.DEC/distance_map.csv": (
                "channel,instrument_distance,distance\n5,,10.0\n,20.0,100.0\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="stated by every point"):
            make_inventory(files)

    def test_a_stray_ordering_column_is_refused(self, make_inventory):
        """A sequence column is scaffolding only where the table says so."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/coupling.csv": (
                "sequence,start_distance,end_distance,coupling_type\n1,0,340,conduit\n"
            ),
        }
        # Coupling is not placed by sequence, so the model calls it unknown.
        with pytest.raises(InvalidInventoryError, match="Could not read OpticalPath"):
            make_inventory(files)


class TestPathEpochsBelongToTheirArray:
    """An epoch cannot start before the entity which holds it."""

    def test_a_bare_path_starts_where_its_array_does(self, make_inventory):
        """Left unset it would claim the unbounded past, before the array."""
        # No acquisition: an undated one could not live in a dated array,
        # which is (a)'s containment rule and not what this pins.
        files = {
            "fiber_arrays/DAS.L001@2024-01-01/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001@2024-01-01/path/attrs.yaml": (
                "object_type: OpticalPath\nname: main\n"
            ),
        }
        array = make_inventory(files).networks[0].fiber_arrays[0]
        assert array.start_time == np.datetime64("2024-01-01", "ns")
        assert array.optical_paths[0].start_time == array.start_time

    def test_an_undated_array_leaves_its_path_undated(self, make_inventory):
        """There is nothing to inherit, so the path keeps its own unset start."""
        path = one_path(make_inventory({**MINIMAL, **TRACKS}))
        assert pd.isnull(path.start_time)

    @pytest.mark.skipif(
        not KEEPS_TRAILING_DOT, reason="this filesystem holds one of the two"
    )
    def test_two_undated_epochs_of_one_lineage(self, make_inventory):
        """NaT equals nothing, so this collision has to be found by hand."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: first\n"
            ),
            # `path.` is the same blank location as `path`, and neither
            # names an instant.
            "fiber_arrays/DAS.L001/path./attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="two spellings of one epoch"):
            make_inventory(files)

    def test_a_lineage_error_names_a_file_which_exists(self, tmp_path):
        """An error which names a file nobody can open helps nobody."""
        root = write_inventory(
            tmp_path / "late_end",
            {
                **MINIMAL,
                "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
                "fiber_arrays/DAS.L001/path/attrs.yaml": (
                    "object_type: OpticalPath\nend_time: 2025-01-01\n"
                ),
                "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                    "object_type: OpticalPath\n"
                ),
            },
        )
        with pytest.raises(InvalidInventoryError) as info:
            dc.inventory(root)
        named = [x for x in str(info.value).split() if x.endswith(".yaml")]
        assert named, str(info.value)
        for name in named:
            assert (root / "fiber_arrays/DAS.L001" / name).exists()


class TestGapsMutationTestingFound:
    """Cases the suite asserted around rather than through."""

    def test_points_gather_into_the_segment_which_names_them(self, make_inventory):
        """Every point table elsewhere holds one object, so grouping is free.

        A loader ignoring the grouping column entirely would pass those,
        merging every point into a single segment.
        """
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                "segment,distance,longitude,latitude,elevation\n"
                "S100,100.0,-117.0,40.0,687.0\n"
                "S120,200.0,-117.2,40.2,690.0\n"
                "S100,102.0,-117.1,40.1,685.0\n"
                "S120,202.0,-117.3,40.3,688.0\n"
            ),
        }
        geometry = {x.name: x for x in one_path(make_inventory(files)).geometry}
        assert set(geometry) == {"S100", "S120"}
        # Interleaved in the file; gathered by name and ordered by distance.
        assert geometry["S100"].distance == (100.0, 102.0)
        assert geometry["S120"].distance == (200.0, 202.0)
        assert geometry["S100"].coordinates[1] == (-117.1, 40.1, 685.0)

    def test_a_wrong_cell_names_the_field_it_could_not_take(self, make_inventory):
        """Matching only 'Could not read' would pass for any path failure."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "sequence,object_type,optical_length,name,fiber_color\n"
                "1,Splice,0.1,b,blue\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="fiber_color"):
            make_inventory(files)

    @pytest.mark.parametrize("order", [("1", "1.5"), ("1.5", "1")])
    def test_an_int_and_a_float_are_one_kind(self, make_inventory, order):
        """The model reads them alike, so row order cannot decide this."""
        first, second = order
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/annotations.csv": (
                f"start_distance,end_distance,group,value\n"
                f"0,120,thickness,{first}\n120,340,thickness,{second}\n"
            ),
        }
        values = [x.value for x in one_path(make_inventory(files)).annotations]
        assert sorted(values) == [1, 1.5]

    @pytest.mark.parametrize("text", ["TRUE", "True", " true "])
    def test_a_boolean_however_a_spreadsheet_writes_it(self, make_inventory, text):
        """Excel writes TRUE; the cell is stripped and folded before reading."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/annotations.csv": (
                f"start_distance,end_distance,group,value\n0,120,noisy,{text}\n"
            ),
        }
        assert one_path(make_inventory(files)).annotations[0].value is True

    def test_a_decimal_point_keeps_a_value_a_float(self, make_inventory):
        """1.0 is written as a float and stays one, unlike 1."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/annotations.csv": (
                "start_distance,end_distance,group,value\n0,120,thickness,1.0\n"
            ),
        }
        value = one_path(make_inventory(files)).annotations[0].value
        assert isinstance(value, float) and value == 1.0

    def test_an_epoch_ending_exactly_where_the_next_begins(self, make_inventory):
        """The ordinary explicit close, which must not read as overlapping."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path/attrs.yaml": (
                "object_type: OpticalPath\nname: first\nend_time: 2024-05-12T10:30:00\n"
            ),
            "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                "object_type: OpticalPath\nname: second\n"
            ),
        }
        paths = {
            x.name: x
            for x in make_inventory(files).networks[0].fiber_arrays[0].optical_paths
        }
        assert paths["first"].end_time == paths["second"].start_time

    def test_lineages_of_two_locations_stay_apart(self, make_inventory):
        """Each location is its own lineage, so neither closes the other."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path.00/attrs.yaml": (
                "object_type: OpticalPath\nname: a\n"
            ),
            "fiber_arrays/DAS.L001/path.01@2024-06-01/attrs.yaml": (
                "object_type: OpticalPath\nname: b\n"
            ),
        }
        paths = {
            x.name: x
            for x in make_inventory(files).networks[0].fiber_arrays[0].optical_paths
        }
        # Pooled into one lineage, `a` would have been closed by `b`.
        assert pd.isnull(paths["a"].end_time)
        assert pd.isnull(paths["b"].end_time)

    def test_a_shouted_table_suffix_and_a_hidden_sidecar(self, tmp_path):
        """A table is found however its suffix is cased, and a dotfile is not."""
        root = write_inventory(tmp_path / "sidecars", {**MINIMAL, **TRACKS})
        entity = root / "fiber_arrays/DAS.L001/path"
        (entity / "coupling.csv").rename(entity / "coupling.CSV")
        (entity / "._geometry.csv").write_text("junk that is not a table\n")
        path = one_path(dc.inventory(root))
        assert len(path.coupling) == 2
        assert [x.name for x in path.geometry] == ["S100"]

    def test_a_blank_line_in_a_table_body(self, make_inventory):
        """A trailing or stray blank line states no row rather than a short one."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/coupling.csv": (
                "start_distance,end_distance,coupling_type\n0,340,conduit\n\n"
            ),
        }
        assert len(one_path(make_inventory(files)).coupling) == 1

    def test_a_cell_holding_the_word_na(self, make_inventory):
        """Only an empty cell is unset; NA is what the author wrote."""
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/coupling.csv": (
                "start_distance,end_distance,coupling_type,description\n"
                "0,340,conduit,NA\n"
            ),
        }
        assert one_path(make_inventory(files)).coupling[0].description == "NA"

    def test_two_undated_epochs_collide_on_every_filesystem(self, tmp_path):
        """The rule above, where the directories which state it cannot exist.

        Windows keeps no trailing dot, so `path.` and `path` are one
        directory there; the comparison is pinned here instead.
        """
        first = inv.OpticalPath(name="first")
        second = inv.OpticalPath(name="second")
        sources = {id(first): tmp_path / "a.yaml", id(second): tmp_path / "b.yaml"}
        assert pd.isnull(first.start_time) and pd.isnull(second.start_time)
        with pytest.raises(InvalidInventoryError, match="two spellings of one epoch"):
            loader._close_lineages([first, second], sources)


class TestPRReviewFindings:
    """Regressions for what the review bots found on the pull request."""

    def test_a_gap_names_the_line_it_is_on(self, make_inventory):
        """The frame is sorted and split by then, so positions lie."""
        files = {
            "acquisitions/DAS.L001.02.DEC/attrs.yaml": (
                "object_type: Acquisition\nspatial_interval: 1.0\n"
            ),
            # Sorted by distance these rows reverse, so a position within
            # the sorted frame is not the line the author would open.
            "acquisitions/DAS.L001.02.DEC/distance_map.csv": (
                "channel,instrument_distance,distance\n"
                "1,10,300\n"  # line 2
                "2,,100\n"  # line 3, the gap
                "3,30,200\n"  # line 4
            ),
        }
        with pytest.raises(InvalidInventoryError, match=r"row\(s\) 3") as info:
            make_inventory(files)
        # Counting within the sorted frame would have said row 2.
        assert "row(s) 2" not in str(info.value)

    def test_an_empty_single_object_table(self, make_inventory):
        """A header and no rows describes no map, rather than raising bare."""
        files = {
            "acquisitions/DAS.L001.02.DEC/attrs.yaml": (
                "object_type: Acquisition\nspatial_interval: 1.0\n"
            ),
            "acquisitions/DAS.L001.02.DEC/distance_map.csv": "channel,distance\n",
        }
        with pytest.raises(InvalidInventoryError, match="describes no distance_map"):
            make_inventory(files)

    @pytest.mark.skipif(FOLDS_CASE, reason="this filesystem holds one of the two")
    def test_two_path_directories_differing_only_by_case(self, make_inventory):
        """Two lineages here become one wherever the inventory is copied."""
        files = {
            **MINIMAL,
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path.aa/attrs.yaml": (
                "object_type: OpticalPath\nname: a\n"
            ),
            "fiber_arrays/DAS.L001/path.AA/attrs.yaml": (
                "object_type: OpticalPath\nname: b\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="differ only by case"):
            make_inventory(files)

    def test_a_symlinked_path_directory(self, tmp_path):
        """A symlink would read an epoch from outside the inventory."""
        outside = tmp_path / "elsewhere" / "path"
        outside.mkdir(parents=True)
        (outside / "attrs.yaml").write_text("object_type: OpticalPath\nname: away\n")
        root = write_inventory(
            tmp_path / "linked",
            {
                **MINIMAL,
                "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            },
        )
        os.symlink(outside, root / "fiber_arrays/DAS.L001/path")
        # Not part of the format, so the array simply has no optical path.
        assert not dc.inventory(root).networks[0].fiber_arrays[0].optical_paths

    def test_a_large_table_is_not_held_twice(self, make_inventory):
        """The width check streams, so only the frame stays resident."""
        rows = "\n".join(f"S100,{x}.0,-117.0,40.0,687.0" for x in range(2000))
        files = {
            **MINIMAL,
            **TRACKS,
            "fiber_arrays/DAS.L001/path/optical_components.csv": (
                "sequence,object_type,optical_length,name\n1,FiberSegment,3000.0,a\n"
            ),
            "fiber_arrays/DAS.L001/path/geometry.csv": (
                f"segment,distance,longitude,latitude,elevation\n{rows}\n"
            ),
        }
        assert len(one_path(make_inventory(files)).geometry[0].distance) == 2000


class TestFindInventory:
    """The name a data directory carries its own inventory under."""

    def test_nothing_there(self, tmp_path):
        """A directory carrying no inventory says so both ways."""
        assert loader.find_inventory(tmp_path) is None
        assert not loader.carries_inventory(tmp_path)

    def test_the_file_form(self, tmp_path):
        """A serialized inventory beside the data it describes."""
        found = tmp_path / f"{loader.BLESSED_NAME}.yaml"
        found.write_text(dc.inventory().to_yaml())
        assert loader.carries_inventory(tmp_path)
        assert loader.find_inventory(tmp_path) == found
        assert isinstance(dc.inventory(loader.find_inventory(tmp_path)), inv.Inventory)

    def test_the_directory_form(self, tmp_path):
        """An authoring directory beside the data it describes."""
        found = write_inventory(tmp_path / loader.BLESSED_NAME, MINIMAL)
        assert loader.carries_inventory(tmp_path)
        assert loader.find_inventory(tmp_path) == found
        assert len(dc.inventory(loader.find_inventory(tmp_path)).networks) == 1

    def test_the_visible_name_is_not_it(self, tmp_path):
        """`inventory.yaml` is the envelope of the authoring format."""
        (tmp_path / "inventory.yaml").write_text(dc.inventory().to_yaml())
        assert not loader.carries_inventory(tmp_path)
        assert loader.find_inventory(tmp_path) is None

    @pytest.mark.parametrize("suffix", loader._OBJECT_SUFFIXES)
    def test_every_object_suffix(self, tmp_path, suffix):
        """One data model stands behind each spelling here too."""
        found = tmp_path / f"{loader.BLESSED_NAME}{suffix}"
        found.write_text("{}")
        assert loader.find_inventory(tmp_path) == found

    def test_two_spellings_of_one_fact(self, tmp_path):
        """A directory states its inventory once."""
        (tmp_path / f"{loader.BLESSED_NAME}.yaml").write_text("{}")
        (tmp_path / loader.BLESSED_NAME).mkdir()
        with pytest.raises(InvalidInventoryError, match="more than one inventory"):
            loader.find_inventory(tmp_path)

    def test_two_files_of_one_fact(self, tmp_path):
        """Two suffixes are two spellings, as they are inside the format."""
        (tmp_path / f"{loader.BLESSED_NAME}.yaml").write_text("{}")
        (tmp_path / f"{loader.BLESSED_NAME}.json").write_text("{}")
        with pytest.raises(InvalidInventoryError, match="more than one inventory"):
            loader.find_inventory(tmp_path)

    def test_a_suffixless_file(self, tmp_path):
        """The bare name is the directory form; a file under it names none."""
        (tmp_path / loader.BLESSED_NAME).write_text("{}")
        with pytest.raises(InvalidInventoryError, match="is a file"):
            loader.find_inventory(tmp_path)

    def test_a_suffixed_directory(self, tmp_path):
        """The authoring directory takes no suffix."""
        (tmp_path / f"{loader.BLESSED_NAME}.yaml").mkdir()
        with pytest.raises(InvalidInventoryError, match="is a directory"):
            loader.find_inventory(tmp_path)

    def test_the_name_is_hidden(self):
        """Which is what keeps the file scanner from reading it as data."""
        assert loader.BLESSED_NAME.startswith(".")


class TestLoadSerializedFile:
    """A whole inventory read from one document, not a directory."""

    def test_json_needs_no_yaml(self, tmp_path, monkeypatch):
        """The suffix picks the parser, so JSON is not YAML's to read."""
        path = tmp_path / "whole.json"
        path.write_text('{"description": "a JSON inventory"}')

        def _refuse(name, **kwargs):
            raise MissingOptionalDependencyError(name)

        # Both bindings, because JSON is legal YAML: a fallback to the
        # YAML route would parse this file and pass a test which only
        # watched the loader's own import.
        monkeypatch.setattr(loader, "optional_import", _refuse)
        monkeypatch.setattr(inv, "optional_import", _refuse)
        assert dc.inventory(path).description == "a JSON inventory"

    def test_a_document_which_does_not_parse(self, tmp_path):
        """Which is an invalid inventory, not a parser's business."""
        path = tmp_path / "whole.yaml"
        path.write_text("this: [is not: yaml\n")
        with pytest.raises(InvalidInventoryError, match="Could not parse YAML"):
            dc.inventory(path)

    def test_fields_which_are_not_named(self, tmp_path):
        """`1: 2` is legal YAML, and names no field of anything."""
        path = tmp_path / "whole.yaml"
        path.write_text("1: 2\n")
        with pytest.raises(InvalidInventoryError, match="not named"):
            dc.inventory(path)

    def test_a_document_which_is_not_a_mapping(self, tmp_path):
        """A list of things defines no inventory."""
        path = tmp_path / "whole.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(InvalidInventoryError, match="holds no mapping"):
            dc.inventory(path)

    # 0x81 is undefined in cp1252 as well as invalid UTF-8, so the file is
    # undecodable wherever this runs rather than only where UTF-8 is the
    # default -- a Windows checkout would otherwise decode it and parse on.
    UNDECODABLE = b"description: \x81\n"

    def test_an_undecodable_file(self, tmp_path):
        """Bytes which are no text are no inventory either."""
        path = tmp_path / "whole.yaml"
        path.write_bytes(self.UNDECODABLE)
        with pytest.raises(InvalidInventoryError, match="Could not read"):
            dc.inventory(path)

    def test_an_undecodable_document_of_any_suffix(self, tmp_path):
        """The route which reads YAML by default decodes alike."""
        path = tmp_path / "whole.txt"
        path.write_bytes(self.UNDECODABLE)
        with pytest.raises(InvalidInventoryError, match="Could not read"):
            dc.inventory(path)

    def test_a_suffixless_document(self, tmp_path):
        """Read as YAML, and refused in the same words when it is not."""
        path = tmp_path / "whole.txt"
        path.write_text("this: [is not: yaml\n")
        with pytest.raises(InvalidInventoryError, match="Could not parse YAML"):
            dc.inventory(path)

    def test_a_suffixless_document_with_unnamed_fields(self, tmp_path):
        """The one route into an inventory says this the same way."""
        path = tmp_path / "whole.txt"
        path.write_text("1: 2\n")
        with pytest.raises(InvalidInventoryError, match="not named"):
            dc.inventory(path)

    def test_a_missing_file_is_still_named(self, tmp_path):
        """The message a caller who mistyped a path needs."""
        with pytest.raises(InvalidInventoryError, match="No such inventory file"):
            dc.inventory(tmp_path / "absent.yaml")

    def test_a_round_trip(self, tmp_path):
        """What `to_yaml` writes is what this reads."""
        original = dc.inventory(write_inventory(tmp_path / "authored", MINIMAL))
        path = tmp_path / "whole.yaml"
        original.to_yaml(path)
        assert dc.inventory(path) == original


class TestOneLoadingDoor:
    """`dc.inventory` decides what a source is; `from_yaml` reads text."""

    def test_text_and_file_agree(self, tmp_path):
        """The same document loads the same from either side of the door."""
        original = dc.inventory(write_inventory(tmp_path / "authored", MINIMAL))
        text = original.to_yaml()
        path = tmp_path / "whole.yaml"
        path.write_text(text)
        assert dc.inventory(text) == dc.inventory(path) == original

    def test_from_yaml_refuses_a_path(self, tmp_path):
        """The text reader says so rather than letting the parser fail."""
        path = tmp_path / "whole.yaml"
        path.write_text(dc.inventory().to_yaml())
        with pytest.raises(InvalidInventoryError, match="reads YAML text"):
            inv.Inventory.from_yaml(path)

    def test_a_mistyped_path_is_not_read_as_a_document(self, tmp_path):
        """A name spelled like a document's is a path which is not there."""
        with pytest.raises(InvalidInventoryError, match="No such inventory file"):
            dc.inventory(str(tmp_path / "absent.yaml"))

    def test_a_file_of_any_suffix_is_named_when_it_fails(self, tmp_path):
        """The path stays in the message, whichever parser read it."""
        path = tmp_path / "whole.txt"
        path.write_text("this: [is not: yaml\n")
        with pytest.raises(InvalidInventoryError, match=str(path.name)):
            dc.inventory(path)

    def test_one_line_document_is_text(self):
        """Existence decides, so a newline is not what makes text text."""
        assert dc.inventory("networks: []").networks == ()

    def test_a_document_whose_value_looks_like_a_file(self):
        """A key makes it a document, whatever its value resembles."""
        assert dc.inventory("description: read from archive.yaml").networks == ()

    def test_unparsable_text_fails_as_an_inventory(self):
        """The text reader refuses in the format's own words."""
        with pytest.raises(InvalidInventoryError, match="Could not parse YAML"):
            dc.inventory("this: [is not: yaml\nnor is this\n")
