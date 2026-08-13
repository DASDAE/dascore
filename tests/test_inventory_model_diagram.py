"""
Tests for the generated inventory data-model diagram.

The diagram on docs/tutorial/inventory.qmd is built from the pydantic models
by scripts/_inventory_model.py, which runs during the documentation build.
These tests pin the property that motivates generating it: what the diagram
shows is what the models say, so no edit to a model can leave the picture
describing an inventory DASCore does not have.

The generator lives outside the package, so it is loaded by path the way
tests/test_changelog.py loads the changelog checker.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
import typing
from pathlib import Path

import pytest

import dascore.core.inventory as inventory_module
from dascore.core.inventory import (
    Acquisition,
    Cable,
    FiberSegment,
    Geometry,
    Interrogator,
    Inventory,
    OpticalPath,
    _IntervalModel,
    _OpticalComponentBase,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PATH = _REPO_ROOT / "scripts" / "_inventory_model.py"
# What scripts/_render_api.py writes the API cross references to. Stated
# here so the two ends of the wiring are pinned independently.
_CROSS_REF_NAME = ".cross_ref.json"

# The generator is part of the documentation build, which an sdist does not
# ship; skip rather than fail where the scripts directory is absent.
pytestmark = pytest.mark.skipif(
    not _GENERATOR_PATH.is_file(),
    reason="the documentation scripts are not present",
)


def _load_generator():
    """Import the diagram generator, which lives outside the package."""
    name = "_inventory_model"
    spec = importlib.util.spec_from_file_location(name, _GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: dataclasses resolves a class's annotations
    # through sys.modules, and the generator defines dataclasses.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


@pytest.fixture(scope="module")
def generator():
    """The module which builds the diagram from the models."""
    pytest.importorskip("yaml", reason="the generator writes YAML")
    return _load_generator()


@pytest.fixture(scope="module")
def spec(generator):
    """The diagram as it is generated from the current models."""
    return generator.build_spec()


@pytest.fixture(scope="module")
def edges(spec):
    """Every containment edge, as (source, target) pairs."""
    return {(edge[0], edge[1]) for edge in spec["edges"]}


@pytest.fixture(scope="module")
def references(spec):
    """Every reference edge, as (source, target, field) triples."""
    return {tuple(edge) for edge in spec["references"]}


class TestNodesFollowTheModels:
    """Each drawn node must describe the model it names."""

    def test_every_public_model_is_drawn(self, generator, spec):
        """A model added to the inventory must appear in the diagram."""
        drawn = set(spec["nodes"])
        for name in generator._inventory_models():
            assert name in drawn

    def test_attributes_are_the_model_fields(self, generator, spec):
        """A node lists the fields its model has, and only those."""
        for node in generator.NODES:
            shown = [x["name"] for x in spec["nodes"][node.name]["attributes"]]
            expected = [
                name for name, info in node.model.model_fields.items() if info.repr
            ]
            assert sorted(shown) == sorted(expected)

    def test_the_serialization_tag_is_hidden(self, spec):
        """object_type is how a document names a class, not a field to set."""
        for node in spec["nodes"].values():
            assert "object_type" not in {x["name"] for x in node["attributes"]}

    def test_descriptions_come_from_the_models(self, generator, spec):
        """Every attribute's documentation is the field's own."""
        for node in generator.NODES:
            for attribute in spec["nodes"][node.name]["attributes"]:
                field = node.model.model_fields[attribute["name"]]
                assert attribute["description"] == (field.description or "")

    def test_types_are_the_written_annotations(self, spec):
        """A type is shown as the model spells it."""
        attributes = {
            x["name"]: x["type"] for x in spec["nodes"]["Acquisition"]["attributes"]
        }
        assert attributes["gauge_length"] == "FiniteFloat | None"
        assert attributes["interrogator"] == "Interrogator | str | None"
        assert attributes["distance_map"] == "DistanceMap | None"

    def test_private_aliases_are_expanded(self, spec):
        """A reader cannot look up an alias the models keep to themselves."""
        resources = {
            x["name"]: x["type"] for x in spec["nodes"]["Inventory"]["attributes"]
        }["resources"]
        assert "_Resource" not in resources
        for name in ("Interrogator", "Cable", "Enclosure", "OpticalMeasurement"):
            assert name in resources

    def test_no_private_name_reaches_the_page(self, spec):
        """Across every drawn field, not just the one alias in use today."""
        for node in spec["nodes"].values():
            for attribute in node["attributes"]:
                assert not re.search(r"(?<![\w.])_[A-Za-z]\w*", attribute["type"])

    def test_an_unexpanded_private_name_fails_the_build(self, generator):
        """
        Rather than shipping a symbol the reader cannot look up.

        One private alias expanding into text holding another's name is how
        this would happen, so the check is on the finished text.
        """
        with pytest.raises(ValueError, match="private name"):
            generator._attributes(Acquisition, {"Interrogator": "_Secret"})

    def test_markdown_in_documentation_fails_the_build(self, generator):
        """
        The filter reads the spec through a markdown parser.

        A backtick or asterisk in a field's documentation would be taken as
        formatting and dropped on the way to the tooltip, so the text is
        refused at the point it is generated. The repo's docstrings use
        double backticks freely, which is how this would arrive.
        """
        for text in ("holds a ``value``", "an *emphatic* claim"):
            with pytest.raises(ValueError, match="read as markdown"):
                generator._check_plain(text, "somewhere")

    def test_summaries_come_from_the_docstrings(self, generator, spec):
        """Every node says what it is, in the model's own words."""
        for node in generator.NODES:
            summary = spec["nodes"][node.name]["summary"]
            first_paragraph = inspect.getdoc(node.model).partition("\n\n")[0]
            assert summary == " ".join(first_paragraph.split())

    def test_every_node_has_a_summary(self, spec):
        """A model with no docstring would leave a node saying nothing."""
        for node in spec["nodes"].values():
            assert node["summary"]

    def test_a_summary_is_the_purpose_not_the_rules(self, generator, spec):
        """The rest of a docstring is the class's rules; they stay behind."""
        # OpticalPath's docstring runs to several paragraphs of track rules.
        doc = inspect.getdoc(OpticalPath)
        summary = spec["nodes"]["OpticalPath"]["summary"]
        assert len(doc.split("\n\n")) > 1
        assert len(summary) < len(doc) / 2

    def test_inherited_fields_are_listed_too(self, spec):
        """A component shows what it gets from its base as well as its own."""
        shown = {x["name"] for x in spec["nodes"]["FiberSegment"]["attributes"]}
        assert {"fiber_number", "loss_db", "optical_length"} <= shown

    def test_own_fields_come_first(self, spec):
        """
        The fields which distinguish a class lead its attribute list.

        Pinned in full, not by a single pair: the order a reader compares
        against the model source is the whole point of sorting, and any
        rule which merely puts one own-field before one inherited field
        would satisfy a spot check while scrambling everything else.
        """
        shown = [x["name"] for x in spec["nodes"]["Acquisition"]["attributes"]]
        assert shown == [
            "code",
            "location_code",
            "data_type",
            "data_category",
            "data_units",
            "interrogator",
            "interrogator_port",
            "firmware_version",
            "software_version",
            "gauge_length",
            "pulse_rate",
            "pulse_width",
            "sample_rate",
            "spatial_interval",
            "distance_map",
            "closed_fiber_loop",
            "start_time",
            "end_time",
            "description",
            "extra_fields",
        ]


class TestEdgesFollowTheModels:
    """Which class points at which must be read out of the annotations."""

    def test_containment_matches_the_fields(self, edges):
        """A field holding another drawn model is an edge to it."""
        assert ("Inventory", "Network") in edges
        assert ("Network", "FiberArray") in edges
        assert ("FiberArray", "Acquisition") in edges
        assert ("FiberArray", "OpticalPath") in edges
        assert ("Acquisition", "DistanceMap") in edges
        assert ("Station", "Channel") in edges

    def test_no_edge_leaves_the_diagram(self, spec, edges, references):
        """Every edge joins two drawn nodes."""
        drawn = set(spec["nodes"])
        for source, target in edges:
            assert source in drawn and target in drawn
        for source, target, _ in references:
            assert source in drawn and target in drawn

    def test_a_union_points_at_the_base_it_shares(self, edges):
        """A path holds components, stated once rather than four times."""
        assert ("OpticalPath", "OpticalComponent") in edges
        for name in ("FiberSegment", "Connector", "Splice", "Terminator"):
            assert ("OpticalComponent", name) in edges
            assert ("OpticalPath", name) not in edges

    def test_a_resource_id_alternative_is_a_reference(self, references):
        """A field taking a str beside a model refers rather than contains."""
        assert ("Acquisition", "Interrogator", "interrogator") in references
        assert ("FiberSegment", "Cable", "container") in references
        assert ("Cable", "ExternalResource", "specification") in references

    def test_a_reference_is_not_also_containment(self, edges):
        """The two kinds of edge are exclusive."""
        assert ("Acquisition", "Interrogator") not in edges
        assert ("FiberSegment", "Cable") not in edges

    def test_a_mapping_key_is_not_a_reference(self, edges, references):
        """The str keying Inventory.resources says nothing about the values."""
        for name in ("Interrogator", "Cable", "Enclosure", "ExternalResource"):
            assert ("Inventory", name) in edges
            assert not [x for x in references if x[:2] == ("Inventory", name)]

    def test_an_inherited_field_is_drawn_once(self, references):
        """The base draws what it declares; subclasses draw their own."""
        measurements = {x for x in references if x[1] == "OpticalMeasurement"}
        assert ("OpticalComponent", "OpticalMeasurement", "loss_measurement") in (
            measurements
        )
        assert ("FiberSegment", "OpticalMeasurement", "loss_measurement") not in (
            measurements
        )

    def test_a_self_reference_survives(self, references):
        """A cable inside a cable is a real thing to say."""
        assert ("Cable", "Cable", "container") in references


class TestNodeListIsValidated:
    """The presentation half must be checked against the models."""

    def _nodes(self, generator, **kwargs):
        """A one-entry node list with the given overrides."""
        return (generator.Node(Inventory, "metadata", **kwargs),)

    def test_a_missing_model_is_refused(self, generator):
        """Leaving a public model out fails the documentation build."""
        with pytest.raises(ValueError, match="leaves out"):
            generator._check_nodes(self._nodes(generator))

    def test_a_public_class_may_not_be_relabelled(self, generator):
        """A drawn class answers to its own name."""
        nodes = tuple(
            generator.Node(node.model, node.style_class, label="Nonsense")
            if node.model is Acquisition
            else node
            for node in generator.NODES
        )
        with pytest.raises(ValueError, match="drawn under its own name"):
            generator._check_nodes(nodes)

    def test_a_private_base_may_be_relabelled(self, generator):
        """Which is how OpticalComponent gets a name a reader knows."""
        generator._check_nodes(generator.NODES)
        assert any(x.label == "OpticalComponent" for x in generator.NODES)

    def test_an_undefined_style_is_refused(self, generator):
        """A node cannot ask for a color the legend does not define."""
        nodes = tuple(
            generator.Node(node.model, "nonexistent") if node.model is Cable else node
            for node in generator.NODES
        )
        with pytest.raises(ValueError, match="undefined style"):
            generator._check_nodes(nodes)

    def test_a_repeated_model_is_refused(self, generator):
        """One class, one box."""
        nodes = (*generator.NODES, generator.NODES[0])
        with pytest.raises(ValueError, match="more than once"):
            generator._check_nodes(nodes)

    def test_a_repeated_drawn_name_is_refused(self, generator):
        """
        Two classes drawn under one name would silently become one box.

        Nodes and edges are both keyed by the drawn name, so the survivor
        would inherit edges pointing at the class which vanished, and
        cytoscape would be handed an edge with a dangling endpoint --
        which throws at page load rather than failing the build.
        """
        nodes = (*generator.NODES, generator.Node(_IntervalModel, "metadata"))
        renamed = tuple(
            generator.Node(node.model, node.style_class, label="OpticalComponent")
            if node.model is _IntervalModel
            else node
            for node in nodes
        )
        with pytest.raises(ValueError, match="both drawn as"):
            generator._check_nodes(renamed)

    def test_a_non_model_is_refused(self, generator):
        """Only pydantic models can be introspected into a node."""
        nodes = (*generator.NODES, generator.Node(int, "metadata"))
        with pytest.raises(ValueError, match="not a pydantic model"):
            generator._check_nodes(nodes)

    def test_every_style_is_used(self, generator):
        """A legend entry no node wears would be explaining nothing."""
        worn = {node.style_class for node in generator.NODES}
        assert worn == {style["id"] for style in generator.STYLES}


class TestAnnotationsMustStayText:
    """The type column is source text, which needs the future import."""

    def test_the_models_keep_the_future_import(self):
        """
        The type column is source text only while the models defer them.

        This is the requirement itself, stated where it can be checked on
        any Python: without the import a class stores evaluated objects,
        and the diagram would print reprs where it promises source.
        """
        source = Path(inventory_module.__file__).read_text(encoding="utf-8")
        assert "from __future__ import annotations" in source

    def test_an_evaluated_annotation_is_refused(self, generator):
        """Losing the future import fails loudly rather than printing reprs."""
        # Built with the annotations in the class namespace rather than
        # assigned afterwards: since PEP 649 (3.14) an assignment does not
        # land in vars(), so the class would carry no annotation at all and
        # the test would pass for the wrong reason.
        evaluated = type("Evaluated", (), {"__annotations__": {"resolved": int}})
        with pytest.raises(TypeError, match="from __future__ import annotations"):
            generator._raw_annotation(evaluated, "resolved")

    def test_an_unknown_field_is_refused(self, generator):
        """Asking for a field no class in the mro declares is a mistake."""
        with pytest.raises(AttributeError, match="no annotation"):
            generator._raw_annotation(Acquisition, "not_a_field")


class TestWrittenSpec:
    """What lands on disk for the documentation build to read."""

    def test_written_where_the_page_looks(self, generator, tmp_path):
        """The page names _generated/inventory_model.yml beside itself."""
        path = generator.write_model_spec(tmp_path)
        assert path == tmp_path / "_generated" / generator.SPEC_NAME
        assert path.is_file()

    def test_says_it_is_generated(self, generator, tmp_path):
        """Nobody should edit the file by hand."""
        text = generator.write_model_spec(tmp_path).read_text(encoding="utf-8")
        assert text.lstrip().startswith("#")
        assert "Do not edit" in text

    def test_round_trips_through_yaml(self, generator, spec, tmp_path):
        """What the filter parses is what the generator built."""
        yaml = pytest.importorskip("yaml")
        text = generator.write_model_spec(tmp_path).read_text(encoding="utf-8")
        assert yaml.safe_load(text) == spec

    def test_api_links_are_resolved_when_known(self, generator):
        """A node links to its class's API page where there is one."""
        cross_refs = {"dascore.core.inventory.Acquisition": "/api/acquisition.qmd"}
        built = generator.build_spec(cross_refs=cross_refs)
        assert built["nodes"]["Acquisition"]["reference_href"] == "/api/acquisition.qmd"
        assert built["nodes"]["Cable"]["reference_href"] == ""

    def test_the_cross_reference_file_is_found(self, generator, tmp_path):
        """
        The written spec picks up the API map the build step leaves.

        Without this the wiring is untested end to end: renaming the file,
        moving the spec a directory deeper, or generating before the API
        docs are rendered each silently strips every link out of the
        diagram, and a spec with no links at all still validates.
        """
        yaml = pytest.importorskip("yaml")
        cross_refs = {"dascore.core.inventory.Cable": "/api/cable.qmd"}
        # The literal name scripts/_render_api.py writes, deliberately not
        # the generator's own constant: reading that back would make a
        # renamed constant agree with itself and prove nothing.
        (tmp_path / _CROSS_REF_NAME).write_text(
            json.dumps(cross_refs), encoding="utf-8"
        )
        written = generator.write_model_spec(tmp_path)
        spec = yaml.safe_load(written.read_text(encoding="utf-8"))
        assert spec["nodes"]["Cable"]["reference_href"] == "/api/cable.qmd"

    def test_the_real_build_resolves_its_links(self, generator):
        """
        Every drawn public class must have an API page to link to.

        The names the generator looks up are the ones scripts/_render_api.py
        writes, so a change to either side's key format shows up here.
        """
        docs = Path(generator.SPEC_PATH).parent
        if not (docs / _CROSS_REF_NAME).is_file():
            pytest.skip("the API docs have not been built in this checkout")
        cross_refs = generator._load_cross_refs(docs)
        assert cross_refs, "the API cross references were written but not found"
        built = generator.build_spec(cross_refs=cross_refs)
        for node in generator.NODES:
            has_link = bool(built["nodes"][node.name]["reference_href"])
            # A private base has no API page; every public class has one.
            assert has_link is not node.model.__name__.startswith("_")


class TestWalkTypes:
    """The rule which separates containing something from referring to it."""

    def test_a_str_beside_a_model_marks_a_reference(self, generator):
        """Which is how a resource_id may stand in for the object."""
        hints = typing.get_type_hints(Acquisition)
        found = dict(generator._walk_types(hints["interrogator"]))
        assert found[Interrogator] is True

    def test_a_model_alone_is_containment(self, generator):
        """Nothing may stand in for a contained object."""
        hints = typing.get_type_hints(OpticalPath)
        found = dict(generator._walk_types(hints["geometry"]))
        # Asserted before the all(), which an empty result would satisfy.
        assert Geometry in found
        assert all(value is False for value in found.values())

    def test_a_container_keeps_the_resource_id_offer(self, generator):
        """A tuple of models beside a str still refers rather than contains."""
        annotation = tuple[Interrogator, ...] | str | None
        found = dict(generator._walk_types(annotation))
        assert found[Interrogator] is True

    def test_a_str_elsewhere_does_not_carry(self, generator):
        """A mapping key is a str about which nothing follows."""
        hints = typing.get_type_hints(Inventory)
        found = dict(generator._walk_types(hints["resources"]))
        assert found[Interrogator] is False

    def test_an_indirect_base_still_hangs_the_subclass(self, generator):
        """
        A drawn subclass finds its nearest drawn ancestor up the whole mro.

        The models already put private classes between a model and the one
        it is drawn under (_IntervalModel), so a subclass whose direct base
        went undrawn would float free: unreachable by expanding anything,
        and drawn as a second root beside Inventory.
        """

        class _Undrawn(_OpticalComponentBase):
            """A private intermediate, of the kind the models already use."""

        class _Leaf(_Undrawn):
            """Drawn, but two steps from the class it is drawn beneath."""

        nodes = (*generator.NODES, generator.Node(_Leaf, "path_detail"))
        containment, _ = generator._edges(nodes)
        assert ["OpticalComponent", "_Leaf"] in containment

    def test_the_drawn_ancestor_wins(self, generator):
        """A class is drawn as the most basal node it descends from."""
        nodes = {node.model: node for node in generator.NODES}
        assert generator._node_of(FiberSegment, nodes).name == "OpticalComponent"
        assert generator._node_of(_OpticalComponentBase, nodes).name == (
            "OpticalComponent"
        )
        assert generator._node_of(Cable, nodes).name == "Cable"
        assert generator._node_of(int, nodes) is None
