"""Tests for loading an inventory from an authoring directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
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
    """The parts of the format which cannot be read yet are refused."""

    def test_track_table_in_an_entity_directory(self, make_inventory):
        """A track table is refused by name rather than ignored."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/coupling.csv": "start_distance\n0\n",
        }
        with pytest.raises(InvalidInventoryError, match="track table"):
            make_inventory(files)

    def test_optical_path_epoch_directory(self, make_inventory):
        """An optical path epoch is refused by name rather than ignored."""
        files = {
            "fiber_arrays/DAS.L001/attrs.yaml": "object_type: FiberArray\n",
            "fiber_arrays/DAS.L001/path@2024-05-12T103000/attrs.yaml": (
                "object_type: OpticalPath\n"
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
