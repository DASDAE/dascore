"""Tests for loading an inventory from an authoring directory."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core import inventory as inv
from dascore.core import inventory_loader as loader
from dascore.exceptions import InvalidInventoryError
from dascore.utils.models import InventoryModel, TimeRangedModel

pytest.importorskip("yaml")


# A minimal directory which loads: one acquisition names everything above it.
MINIMAL = {
    "acquisitions/DAS.L001..RAW.yaml": "type: Acquisition\ndata_category: DAS\n",
}


def write_inventory(root, files) -> object:
    """Write a mapping of relative path to text under root."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


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
                    "type: Acquisition\ndata_category: DAS\ngauge_length: 10.0\n"
                ),
            }
        )
        assert out.resolve("DAS.L001..RAW").acquisition.gauge_length == 10.0

    def test_full_directory(self, make_inventory):
        """Every container contributes to one inventory."""
        out = make_inventory(
            {
                "inventory.yaml": "type: Inventory\nschema_version: 1\n",
                "resources/int_01.yaml": (
                    "type: Interrogator\nmanufacturer: Fake\nmodel: FI-1\n"
                ),
                "networks/DAS.yaml": "type: Network\nname: test network\n",
                "fiber_arrays/DAS.L001.yaml": "type: FiberArray\nname: first\n",
                "stations/DAS.STA1.yaml": "type: Station\nname: a station\n",
                "acquisitions/DAS.L001..RAW.yaml": (
                    "type: Acquisition\ndata_category: DAS\ninterrogator: int_01\n"
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
                        "type: Acquisition\ndata_category: DAS\ngauge_length: 4.0\n"
                    )
                },
            )
        )
        as_json = dc.inventory(
            write_inventory(
                tmp_path / "json_form",
                {
                    "acquisitions/DAS.L001..RAW.json": (
                        '{"type": "Acquisition", "data_category": "DAS", '
                        '"gauge_length": 4.0}'
                    )
                },
            )
        )
        assert as_yaml.networks == as_json.networks

    def test_yml_suffix(self, make_inventory):
        """The short YAML suffix is the same spelling."""
        out = make_inventory({"acquisitions/DAS.L001..RAW.yml": "type: Acquisition\n"})
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
                    "type: Inventory\n"
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
                    "type: FiberArray\nname: from a directory\n"
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
                ".hidden/DAS.yaml": "type: Network\n",
                "acquisitions/README.txt": "a note beside the objects\n",
            }
        )
        assert [x.code for x in out.networks] == ["DAS"]

    def test_resource_id_may_hold_dots(self, make_inventory):
        """A single-token identity is the whole name, dots and all."""
        out = make_inventory(
            {
                **MINIMAL,
                "resources/cable.01.yaml": "type: Cable\nname: a cable\n",
            }
        )
        assert out.get_resource("cable.01").name == "a cable"

    def test_restated_address_may_agree(self, make_inventory):
        """A name may be restated inside the file when the two agree."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001.01.RAW.yaml": (
                    "type: Acquisition\ncode: RAW\nlocation_code: '01'\n"
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
            {"acquisitions/DAS.L001..RAW@2024-06-01.yaml": "type: Acquisition\n"}
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert acquisition.start_time == np.datetime64("2024-06-01T00:00:00", "ns")

    def test_basic_time_and_fractional_seconds(self, make_inventory):
        """The time portion is ISO basic, since ':' is not a legal filename."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW@2024-05-12T103000.12.yaml": (
                    "type: Acquisition\n"
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
                    "type: Acquisition\nstart_time: '2024-06-01'\n"
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
                    "type: Acquisition\nend_time: '2024-06-01'\ngauge_length: 10.0\n"
                ),
                "acquisitions/DAS.L001..RAW@2024-06-01.yaml": (
                    "type: Acquisition\ngauge_length: 5.0\n"
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
                    "type: FiberArray\nname: first\nend_time: '2024-06-01'\n"
                ),
                "fiber_arrays/DAS.L001@2024-06-01.yaml": (
                    "type: FiberArray\nname: second\n"
                ),
                "acquisitions/DAS.L001..RAW@2024-07-01.yaml": "type: Acquisition\n",
            }
        )
        arrays = {x.name: x for x in out.networks[0].fiber_arrays}
        assert not arrays["first"].acquisitions
        assert [x.code for x in arrays["second"].acquisitions] == ["RAW"]


class TestNearMisses:
    """Anything claiming to participate and getting it wrong raises."""

    def test_typo_in_container_name(self, make_inventory):
        """A typo must not quietly load an inventory with no acquisitions."""
        files = {"aquisitions/DAS.L001..RAW.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="nothing contains it"):
            make_inventory(files)

    def test_model_file_at_the_root(self, make_inventory):
        """An object at the root is outside every container."""
        files = {**MINIMAL, "DAS.L001.yaml": "type: FiberArray\n"}
        with pytest.raises(InvalidInventoryError, match="nothing contains it"):
            make_inventory(files)

    def test_missing_type(self, make_inventory):
        """A file which does not say what it is has not participated."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "data_category: DAS\n"}
        with pytest.raises(InvalidInventoryError, match="declares no type"):
            make_inventory(files)

    def test_wrong_container(self, make_inventory):
        """The container checks the declared type rather than supplying it."""
        files = {"fiber_arrays/DAS.L001.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="cannot hold"):
            make_inventory(files)

    def test_unknown_type(self, make_inventory):
        """A type which names no model is unknown rather than misfiled."""
        files = {"fiber_arrays/DAS.L001.yaml": "type: Telescope\n"}
        with pytest.raises(InvalidInventoryError, match="unknown"):
            make_inventory(files)

    def test_restated_address_disagrees(self, make_inventory):
        """There is never a precedence rule between two spellings."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "type: Acquisition\ncode: DEC\n"}
        with pytest.raises(InvalidInventoryError, match="must agree with the name"):
            make_inventory(files)

    def test_restated_start_time_disagrees(self, make_inventory):
        """The epoch suffix is a restated address, so it must agree."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-06-01.yaml": (
                "type: Acquisition\nstart_time: '2024-06-02'\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="must agree with the name"):
            make_inventory(files)

    def test_illegal_address_token_names_the_file(self, make_inventory):
        """The entity a token names is built from every address, not one file."""
        files = {"acquisitions/DAS.L0_01..RAW.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="names fiber_array"):
            make_inventory(files)

    def test_wrong_token_count(self, make_inventory):
        """An acquisition name is an address of four tokens."""
        files = {"acquisitions/DAS.L001.RAW.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="address of 4"):
            make_inventory(files)

    def test_schema_version_outside_the_envelope(self, make_inventory):
        """The envelope versions the document exactly once."""
        files = {
            "acquisitions/DAS.L001..RAW.yaml": (
                "type: Acquisition\nschema_version: 1\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="envelope versions"):
            make_inventory(files)

    def test_invalid_field_names_the_file(self, make_inventory):
        """A model error says which file could not be read."""
        files = {
            "acquisitions/DAS.L001..RAW.yaml": (
                "type: Acquisition\ngauge_length: not a number\n"
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


class TestReviewFindings:
    """Regressions for what the counterpart review found."""

    def test_object_filed_inside_an_entity_directory(self, make_inventory):
        """An object one level too deep must not be silently dropped."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "type: FiberArray\n",
            "fiber_arrays/DAS.L001/misplaced/DAS.L001..RAW.yaml": (
                "type: Acquisition\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="nothing contains it"):
            make_inventory(files)

    @pytest.mark.parametrize("suffix", ["YAML", "YML", "JSON"])
    def test_upper_case_suffixes(self, make_inventory, suffix):
        """A case-insensitive filesystem holds one file, not two spellings."""
        text = (
            '{"type": "Acquisition"}'
            if suffix == "JSON"
            else "type: Acquisition\ndata_category: DAS\n"
        )
        out = make_inventory({f"acquisitions/DAS.L001..RAW.{suffix}": text})
        assert out.networks[0].fiber_arrays[0].acquisitions[0].code == "RAW"

    def test_upper_case_envelope(self, make_inventory):
        """The envelope is found however its suffix is spelled."""
        out = make_inventory(
            {**MINIMAL, "inventory.YAML": "type: Inventory\nresource_id: shouted\n"}
        )
        assert out.resource_id == "shouted"

    def test_child_outliving_its_parent_epoch(self, make_inventory):
        """Starting inside an epoch is not enough to belong to it."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "fiber_arrays/DAS.L001@2024-06-01.yaml": "type: FiberArray\n",
            # Starts inside the first epoch and never ends, so resolution
            # after June would find the second array, which does not hold it.
            "acquisitions/DAS.L001..RAW@2024-05-01.yaml": "type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="runs past"):
            make_inventory(files)

    def test_child_ending_exactly_at_the_boundary_fits(self, make_inventory):
        """Half-open on both sides, so the shared instant belongs to neither."""
        out = make_inventory(
            {
                "fiber_arrays/DAS.L001.yaml": (
                    "type: FiberArray\nend_time: '2024-06-01'\n"
                ),
                "fiber_arrays/DAS.L001@2024-06-01.yaml": "type: FiberArray\n",
                "acquisitions/DAS.L001..RAW@2024-05-01.yaml": (
                    "type: Acquisition\nend_time: '2024-06-01'\n"
                ),
            }
        )
        first = out.networks[0].fiber_arrays[0]
        assert [x.code for x in first.acquisitions] == ["RAW"]

    def test_unreadable_envelope_value_names_the_file(self, make_inventory):
        """An envelope error reads like every other error this format raises."""
        files = {**MINIMAL, "inventory.yaml": "type: Inventory\nschema_version: nope\n"}
        with pytest.raises(InvalidInventoryError, match="Could not read the envelope"):
            make_inventory(files)

    def test_type_which_is_not_a_name(self, make_inventory):
        """A type which is not a name at all names no model either."""
        files = {"acquisitions/DAS.L001..RAW.yaml": "type: [Acquisition]\n"}
        with pytest.raises(InvalidInventoryError, match="declares no type"):
            make_inventory(files)


class TestOneIdentityOneSpelling:
    """Identity is unique per container regardless of spelling."""

    def test_two_extensions(self, make_inventory):
        """The same name with two extensions is one identity spelled twice."""
        files = {
            "resources/cable_01.yaml": "type: Cable\n",
            "resources/cable_01.json": '{"type": "Cable"}',
        }
        with pytest.raises(InvalidInventoryError, match="two extensions"):
            make_inventory(files)

    def test_case_only_difference(self, make_inventory):
        """A case-insensitive filesystem could not hold both."""
        files = {
            "fiber_arrays/DAS.L001.yaml": "type: FiberArray\n",
            "fiber_arrays/das.l001.yaml": "type: FiberArray\n",
        }
        with pytest.raises(InvalidInventoryError, match="differ only by case"):
            make_inventory(files)

    def test_file_and_directory(self, make_inventory):
        """Both spellings of one identity at once raise."""
        files = {
            "fiber_arrays/DAS.L001.yaml": "type: FiberArray\n",
            "fiber_arrays/DAS.L001/attrs.yaml": "type: FiberArray\n",
        }
        with pytest.raises(InvalidInventoryError, match="file and a directory"):
            make_inventory(files)

    def test_envelope_spelled_twice(self, make_inventory):
        """The envelope is one file however it is spelled."""
        files = {
            **MINIMAL,
            "inventory.yaml": "type: Inventory\n",
            "inventory.json": '{"type": "Inventory"}',
        }
        with pytest.raises(InvalidInventoryError, match="more than once"):
            make_inventory(files)

    def test_two_names_for_one_epoch(self, make_inventory):
        """Epoch-name uniqueness is temporal rather than textual."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-06-01.yaml": "type: Acquisition\n",
            "acquisitions/DAS.L001..RAW@2024-06-01T000000.yaml": "type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="one epoch"):
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
        files = {"fiber_arrays/DAS.L001/array.yaml": "type: FiberArray\n"}
        with pytest.raises(InvalidInventoryError, match="not part of an entity"):
            make_inventory(files)

    def test_attrs_spelled_twice(self, make_inventory):
        """One identity is spelled once inside the directory too."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "type: FiberArray\n",
            "fiber_arrays/DAS.L001/attrs.json": '{"type": "FiberArray"}',
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
                    "type: FiberArray\nname: from a directory\n"
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
        with pytest.raises(InvalidInventoryError, match="Could not read"):
            dc.inventory(root)


class TestSeams:
    """The parts of the format which cannot be read yet are refused."""

    def test_track_table_in_an_entity_directory(self, make_inventory):
        """A track table is refused by name rather than ignored."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "type: FiberArray\n",
            "fiber_arrays/DAS.L001/coupling.csv": "start_distance\n0\n",
        }
        with pytest.raises(InvalidInventoryError, match="track table"):
            make_inventory(files)

    def test_optical_path_epoch_directory(self, make_inventory):
        """An optical path epoch is refused by name rather than ignored."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "type: FiberArray\n",
            "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                "type: OpticalPath\n"
            ),
        }
        with pytest.raises(InvalidInventoryError, match="optical path epoch"):
            make_inventory(files)

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
        files = {f"acquisitions/DAS.L001..RAW@{stamp}.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="epoch timestamp"):
            make_inventory(files)

    @pytest.mark.parametrize(
        "stamp", ["2024-06-01Z", "2024-05-12T103000Z", "2024-05-12T103000+0100"]
    )
    def test_timezone_designator(self, make_inventory, stamp):
        """Naive means UTC, so a designator is refused rather than ignored."""
        files = {f"acquisitions/DAS.L001..RAW@{stamp}.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="timezone designator"):
            make_inventory(files)

    def test_negative_offset(self, make_inventory):
        """An offset in the time portion is a designator, not a malformed time."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-05-12T103000-0600.yaml": (
                "type: Acquisition\n"
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
        files = {f"acquisitions/DAS.L001..RAW@{stamp}.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="outside the range"):
            make_inventory(files)

    def test_the_last_representable_instant(self, make_inventory):
        """The check refuses what wraps without refusing what does not."""
        out = make_inventory(
            {
                "acquisitions/DAS.L001..RAW@2262-04-11T234716.yaml": (
                    "type: Acquisition\n"
                )
            }
        )
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]
        assert acquisition.start_time == np.datetime64("2262-04-11T23:47:16", "ns")

    def test_two_epoch_markers(self, make_inventory):
        """A name carries at most one epoch."""
        files = {
            "acquisitions/DAS.L001..RAW@2024-06-01@2024-07-01.yaml": (
                "type: Acquisition\n"
            )
        }
        with pytest.raises(InvalidInventoryError, match="more than one"):
            make_inventory(files)

    def test_empty_epoch(self, make_inventory):
        """A trailing marker names no epoch."""
        files = {"acquisitions/DAS.L001..RAW@.yaml": "type: Acquisition\n"}
        with pytest.raises(InvalidInventoryError, match="names no epoch"):
            make_inventory(files)

    def test_resources_have_no_epochs(self, make_inventory):
        """A resource is not time-ranged, so its name states no epoch."""
        files = {**MINIMAL, "resources/int_01@2024-06-01.yaml": "type: Interrogator\n"}
        with pytest.raises(InvalidInventoryError, match="have none"):
            make_inventory(files)


class TestEpochPlacement:
    """A child is placed in the container epoch which held it."""

    def test_child_outside_every_epoch(self, make_inventory):
        """A child falling in no epoch of its container is misfiled."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "acquisitions/DAS.L001..RAW@2024-07-01.yaml": "type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="0 epochs effective"):
            make_inventory(files)

    def test_ambiguous_child(self, make_inventory):
        """An unset start beside several container epochs is ambiguous."""
        files = {
            "fiber_arrays/DAS.L001.yaml": (
                "type: FiberArray\nend_time: '2024-06-01'\n"
            ),
            "fiber_arrays/DAS.L001@2024-06-01.yaml": "type: FiberArray\n",
            "acquisitions/DAS.L001..RAW.yaml": "type: Acquisition\n",
        }
        with pytest.raises(InvalidInventoryError, match="2 epochs effective at any"):
            make_inventory(files)

    def test_station_placed_in_a_network_epoch(self, make_inventory):
        """Networks epoch like everything else which is time-ranged."""
        out = make_inventory(
            {
                "networks/DAS.yaml": "type: Network\nend_time: '2024-06-01'\n",
                "networks/DAS@2024-06-01.yaml": "type: Network\nname: later\n",
                "stations/DAS.STA1@2024-07-01.yaml": "type: Station\n",
            }
        )
        by_name = {x.name: x for x in out.networks}
        assert not by_name[""].stations
        assert [x.code for x in by_name["later"].stations] == ["STA1"]


class TestEnvelope:
    """The envelope holds the document-level singletons and nothing else."""

    def test_wrong_type(self, make_inventory):
        """The envelope declares its type under the same rule as any file."""
        files = {**MINIMAL, "inventory.yaml": "type: Network\n"}
        with pytest.raises(InvalidInventoryError, match="envelope declares"):
            make_inventory(files)

    @pytest.mark.parametrize("field", ["networks", "resources"])
    def test_collections_are_refused(self, make_inventory, field):
        """The collections live in the directory structure."""
        files = {**MINIMAL, "inventory.yaml": f"type: Inventory\n{field}: []\n"}
        with pytest.raises(InvalidInventoryError, match="directory structure"):
            make_inventory(files)

    def test_unknown_field_names_the_file(self, make_inventory):
        """A typo in the envelope says which file states it."""
        files = {**MINIMAL, "inventory.yaml": "type: Inventory\nschema_verison: 1\n"}
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

    def test_empty_directory(self, tmp_path):
        """A directory with nothing in it is an empty inventory."""
        root = tmp_path / "empty"
        root.mkdir()
        assert not dc.inventory(root).networks


class TestModelAssumptions:
    """Pin what this loader assumes about the models it builds."""

    def test_only_union_members_carry_a_type_field(self):
        """The type tag is a real field only where it discriminates a union."""
        checked = []
        for name, container in loader._CONTAINERS.items():
            for model in container.models:
                assert ("type" in model.model_fields) == (name == "resources")
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

    def test_inventory_models_forbid_extra_fields(self):
        """Type must be popped: an inventory model refuses unknown input."""
        assert InventoryModel.model_config["extra"] == "forbid"
