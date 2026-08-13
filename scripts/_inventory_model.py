"""
Generate the inventory data-model diagram from DASCore's own models.

`docs/tutorial/inventory.qmd` draws the inventory as an interactive class
diagram. Everything in that picture -- which classes appear, what each one
holds, the type and documentation of every field, and which class points at
which -- is read out of the pydantic models here, so the diagram cannot
describe a model DASCore does not have.

The one thing the models cannot state is how the picture should look. That
half lives in `NODES` below: which classes are drawn, what color group each
belongs to, and which start collapsed. It is validated against the models
every time it runs, so a class renamed or removed fails the documentation
build rather than quietly dropping out of the diagram.
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union, get_args, get_origin

import yaml

import dascore.core.inventory as inventory_module
from dascore.core.inventory import (
    Acquisition,
    Cable,
    Channel,
    Connector,
    CoordinateReferenceSystem,
    CouplingCondition,
    CreationInfo,
    DistanceMap,
    Enclosure,
    ExternalResource,
    FiberArray,
    FiberSegment,
    Geometry,
    Interrogator,
    Inventory,
    Network,
    OpticalMeasurement,
    OpticalPath,
    OpticalPathAnnotation,
    Response,
    Splice,
    Station,
    Terminator,
    _OpticalComponentBase,
)
from dascore.models import InventoryModel

# Where the generated spec lands. Under an underscore directory so quarto
# leaves it alone, and a build artifact rather than a source file: it is
# rewritten from the models on every documentation build.
SPEC_PATH = Path(__file__).resolve().parent.parent / "docs" / "_generated"
SPEC_NAME = "inventory_model.yml"
# Written by scripts/_render_api.py; maps a dotted name to its API doc page.
CROSS_REF_NAME = ".cross_ref.json"


@dataclass(frozen=True)
class Node:
    """One class in the diagram, and how it should be drawn."""

    model: type
    # Groups the node in the legend; see STYLES for the palette.
    style_class: str
    # A shorter name to print inside the node when the class name is long
    # enough to stretch the box. The full name still titles the tooltip.
    display: str = ""
    # Nodes whose children only matter once a reader goes looking.
    collapsed: bool = False
    # Only a private base class may be renamed, so a public class can never
    # be drawn under a name it does not answer to. See _check_nodes.
    label: str = ""

    @property
    def name(self) -> str:
        """The name this class is drawn under."""
        return self.label or self.model.__name__


# The diagram, in the order the nodes are declared. Adding a model to the
# inventory means adding it here, which _check_nodes enforces; what the
# whole file must hold to is tests/test_inventory_model_diagram.py.
NODES: tuple[Node, ...] = (
    Node(Inventory, "metadata"),
    Node(CreationInfo, "metadata", display="Creation info", collapsed=True),
    Node(CoordinateReferenceSystem, "metadata", display="CRS"),
    Node(Network, "container"),
    Node(Station, "monitoring"),
    Node(Channel, "instrument", collapsed=True),
    Node(Response, "instrument", collapsed=True),
    Node(FiberArray, "monitoring"),
    Node(Acquisition, "instrument"),
    Node(DistanceMap, "instrument", display="Distance map"),
    Node(OpticalPath, "monitoring", collapsed=True),
    Node(
        _OpticalComponentBase,
        "path_detail",
        display="Component",
        collapsed=True,
        label="OpticalComponent",
    ),
    Node(FiberSegment, "path_detail", display="Fiber segment"),
    Node(Connector, "path_detail"),
    Node(Splice, "path_detail"),
    Node(Terminator, "path_detail"),
    Node(Geometry, "path_detail"),
    Node(CouplingCondition, "path_detail", display="Coupling"),
    Node(OpticalPathAnnotation, "path_detail", display="Annotation"),
    Node(Interrogator, "shareable_resource"),
    Node(Cable, "shareable_resource"),
    Node(Enclosure, "shareable_resource"),
    Node(ExternalResource, "shareable_resource", display="External resource"),
    Node(OpticalMeasurement, "shareable_resource", display="Measurement"),
)

# The legend, and the fill/stroke/text colors each group draws with.
STYLES: tuple[dict[str, str], ...] = (
    {
        "id": "metadata",
        "label": "Document metadata",
        "description": "The document itself and the frame it is read in.",
        "fill": "#e7f5ff",
        "stroke": "#0b7285",
        "color": "#102a43",
    },
    {
        "id": "container",
        "label": "Container",
        "description": "Organizational grouping for observing identities.",
        "fill": "#ebfbee",
        "stroke": "#2f9e44",
        "color": "#1b4332",
    },
    {
        "id": "monitoring",
        "label": "Observing identity",
        "description": "A durable thing that is monitored or does monitoring.",
        "fill": "#f3f0ff",
        "stroke": "#6741d9",
        "color": "#2f1b78",
    },
    {
        "id": "instrument",
        "label": "Instrument configuration",
        "description": "How an instrument was set up to record.",
        "fill": "#fff4e6",
        "stroke": "#f08c00",
        "color": "#5c3d00",
    },
    {
        "id": "path_detail",
        "label": "Path detail",
        "description": "A component or a described interval along an optical path.",
        "fill": "#fff0f6",
        "stroke": "#c2255c",
        "color": "#5c1230",
    },
    {
        "id": "shareable_resource",
        "label": "Shareable resource",
        "description": "A resource_id object reused across the inventory.",
        "fill": "#eef2ff",
        "stroke": "#5c7cfa",
        "color": "#25316d",
    },
)


@dataclass(frozen=True)
class _Reference:
    """A field naming another class, and how the field reaches it."""

    target: type
    by_resource_id: bool
    field_name: str
    # Where the field is declared, which is not always where it is drawn:
    # an inherited field belongs to the base class's node.
    declared_by: type


def _inventory_models(module=inventory_module) -> dict[str, type]:
    """Return the public models the inventory module defines."""
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and inspect.isclass(value)
        and issubclass(value, InventoryModel)
        and value.__module__ == module.__name__
    }


def _check_nodes(nodes: tuple[Node, ...] = NODES) -> None:
    """Refuse a node list the models cannot back, or which leaves one out."""
    seen: set[type] = set()
    drawn_as: set[str] = set()
    for node in nodes:
        model = node.model
        if not (inspect.isclass(model) and hasattr(model, "model_fields")):
            msg = f"{model!r} is not a pydantic model."
            raise ValueError(msg)
        if model in seen:
            msg = f"{model.__name__} is drawn more than once."
            raise ValueError(msg)
        seen.add(model)
        # Nodes and edges are both keyed by the drawn name, so two classes
        # sharing one would silently collapse into a single box and leave
        # the other's edges pointing at nothing.
        if node.name in drawn_as:
            msg = f"Two classes are both drawn as {node.name!r}."
            raise ValueError(msg)
        drawn_as.add(node.name)
        if node.label and not model.__name__.startswith("_"):
            msg = (
                f"{model.__name__} is public, so it must be drawn under its own "
                f"name rather than {node.label!r}."
            )
            raise ValueError(msg)
        if node.style_class not in {style["id"] for style in STYLES}:
            msg = f"{node.name} uses undefined style {node.style_class!r}."
            raise ValueError(msg)
    # A model added to the inventory and not to the diagram would simply be
    # missing from it, which is the one kind of drift a reader cannot see.
    # Private models are drawn only where they earn a box of their own.
    if missing := sorted(set(_inventory_models()) - {x.__name__ for x in seen}):
        msg = (
            f"The inventory model diagram leaves out {', '.join(missing)}. Add "
            "each to NODES in scripts/_inventory_model.py."
        )
        raise ValueError(msg)


def _declaring_class(model: type, name: str) -> type:
    """Return the class in the mro which declares a field."""
    for base in model.__mro__:
        if name in vars(base).get("__annotations__", {}):
            return base
    msg = f"{model.__name__} has no annotation for field {name!r}."
    raise AttributeError(msg)


def _raw_annotation(model: type, name: str) -> str:
    """
    Return the annotation of a field as its source text.

    The models are written under ``from __future__ import annotations``, so
    what a class records is the text the author wrote -- ``tuple[Acquisition,
    ...]`` rather than a resolved and re-printed type. That text is what the
    diagram shows, so a reader of the diagram and a reader of the model see
    the same thing.
    """
    declared_by = _declaring_class(model, name)
    found = vars(declared_by)["__annotations__"][name]
    if not isinstance(found, str):
        msg = (
            f"{model.__name__}.{name} is annotated with an object rather "
            "than text; dascore.core.inventory must keep its "
            "'from __future__ import annotations'."
        )
        raise TypeError(msg)
    return found


def _render_type(annotation) -> str:
    """Render a resolved annotation the way it would have been written."""
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return " | ".join(_render_type(x) for x in get_args(annotation))
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _alias_expansions(module=inventory_module) -> dict[str, str]:
    """
    Return the private type aliases, and the text to show in their place.

    ``_Resource`` says nothing to a reader of the documentation, while the
    five classes it stands for say everything.
    """
    out = {}
    for name, value in vars(module).items():
        if not name.startswith("_") or name.startswith("__"):
            continue
        # Written with the discriminator wrapper or without it: a bare
        # `_Component: TypeAlias = FiberSegment | Connector` must expand too,
        # or its private name is printed into the type column of every field
        # which uses it, naming a symbol the reader cannot look up.
        origin = get_origin(value)
        if origin is Annotated:
            out[name] = _render_type(get_args(value)[0])
        elif origin in (Union, types.UnionType):
            out[name] = _render_type(value)
    return out


def _shown_fields(model: type) -> list[str]:
    """
    Return the fields to draw, the class's own first.

    A field hidden from the model's repr is hidden here too: that is how the
    serialization tag every union member carries declares itself an internal
    detail rather than something a user writes.
    """
    names = [name for name, info in model.model_fields.items() if info.repr]
    depth = model.__mro__.index
    return sorted(names, key=lambda name: depth(_declaring_class(model, name)))


def _attributes(model: type, expansions: dict[str, str]) -> list[dict[str, str]]:
    """Return one entry per drawn field of a model."""
    out = []
    for name in _shown_fields(model):
        text = _raw_annotation(model, name)
        for alias, expansion in expansions.items():
            text = re.sub(rf"\b{re.escape(alias)}\b", expansion, text)
        # A private name reaching the page names a symbol the reader cannot
        # look up, so it fails the build rather than shipping. One alias
        # expanding into another's name is the way it would happen.
        if leaked := re.search(r"(?<![\w.])_[A-Za-z]\w*", text):
            msg = (
                f"{model.__name__}.{name} would show the private name "
                f"{leaked.group()!r} in {text!r}. Expand it in "
                "_alias_expansions, or draw what it stands for."
            )
            raise ValueError(msg)
        out.append(
            {
                "name": name,
                "type": text,
                "description": model.model_fields[name].description or "",
            }
        )
    return out


def _summary(model: type) -> str:
    """
    Return the first paragraph of a model's docstring.

    The rest of the docstring is the class's rules, which belong on its API
    page; a tooltip wants the sentence that says what the thing is.
    """
    doc = inspect.getdoc(model) or ""
    paragraph = doc.split("\n\n")[0]
    return " ".join(paragraph.split())


def _walk_types(annotation, resource_id_accepted: bool = False):
    """
    Yield every class in an annotation, and how the annotation reaches it.

    A field which accepts a ``str`` beside a model accepts a resource_id in
    place of the object, which is what makes it a reference to a shared
    resource rather than something the model contains. The ``str`` has to
    sit in the *same* union as the model to mean that: the key type of a
    mapping is a str about which nothing follows.
    """
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = get_args(annotation)
        accepted = str in args
        for arg in args:
            yield from _walk_types(arg, accepted)
    elif origin is not None:
        # The flag carries into a container: `tuple[Interrogator, ...] | str`
        # offers a resource_id for the things in the tuple.
        for arg in get_args(annotation):
            yield from _walk_types(arg, resource_id_accepted)
    elif isinstance(annotation, type):
        yield annotation, resource_id_accepted


def _references(model: type) -> list[_Reference]:
    """Return every other class a model's drawn fields point at."""
    hints = typing.get_type_hints(model)
    out = []
    for name in _shown_fields(model):
        declared_by = _declaring_class(model, name)
        for target, by_resource_id in _walk_types(hints[name]):
            out.append(_Reference(target, by_resource_id, name, declared_by))
    return out


def _node_of(model: type, nodes: dict[type, Node]) -> Node | None:
    """
    Return the node a class is drawn as, which may be an ancestor of it.

    The most basal drawn ancestor wins, so a field holding the union of the
    four optical components points at the base class they share: the diagram
    states once that a path holds components, and separately which
    components there are.
    """
    for base in reversed(model.__mro__):
        if (node := nodes.get(base)) is not None:
            return node
    return None


def _edges(nodes: tuple[Node, ...] = NODES) -> tuple[list, list]:
    """Return the containment and reference edges between the drawn nodes."""
    by_model = {node.model: node for node in nodes}
    containment: list[list[str]] = []
    references: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(into: list, source: str, target: str, label: str) -> None:
        key = (source, target, label)
        if key in seen:
            return
        seen.add(key)
        into.append([source, target, label] if label else [source, target])

    for node in nodes:
        # A drawn subclass hangs off its nearest drawn ancestor, which need
        # not be a direct base: this module already puts private classes
        # (_IntervalModel) between a model and the one it is drawn under,
        # and a subclass whose parent went undrawn would otherwise become an
        # island no amount of expanding could reach.
        for base in node.model.__mro__[1:]:
            if (parent := by_model.get(base)) is not None:
                _add(containment, parent.name, node.name, "")
                break
        for reference in _references(node.model):
            # A field inherited from another drawn class is that class's to
            # draw, so four component classes do not each redraw the base's
            # edges. A field from an undrawn base belongs to every node
            # which has it, there being nowhere else to put it.
            declared_by = reference.declared_by
            if declared_by is not node.model and declared_by in by_model:
                continue
            target = _node_of(reference.target, by_model)
            if target is None or (target is node and not reference.by_resource_id):
                continue
            if reference.by_resource_id:
                _add(references, node.name, target.name, reference.field_name)
            else:
                _add(containment, node.name, target.name, "")
    return containment, references


def _api_paths(cross_refs: dict[str, str], nodes=NODES) -> dict[str, str]:
    """Return each node's API documentation page, where it has one."""
    out = {}
    for node in nodes:
        key = f"{node.model.__module__}.{node.model.__qualname__}"
        if (found := cross_refs.get(key)) is not None:
            out[node.name] = found
    return out


def _load_cross_refs(docs_path: Path) -> dict[str, str]:
    """Load the API cross-reference map, which an earlier step writes."""
    path = docs_path / CROSS_REF_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_spec(nodes: tuple[Node, ...] = NODES, cross_refs=None) -> dict:
    """Return the diagram spec described by the inventory models."""
    _check_nodes(nodes)
    expansions = _alias_expansions()
    api_paths = _api_paths(cross_refs or {}, nodes)
    containment, references = _edges(nodes)
    return {
        "id": "dasdae-inventory",
        "title": "DASDAE Inventory",
        "direction": "TB",
        "styles": [dict(style) for style in STYLES],
        "nodes": {
            node.name: {
                "label": node.name,
                "display": node.display or node.name,
                "style_class": node.style_class,
                "children": "collapsed" if node.collapsed else "open",
                "summary": _summary(node.model),
                "reference_href": api_paths.get(node.name, ""),
                "attributes": _attributes(node.model, expansions),
            }
            for node in nodes
        },
        "edges": containment,
        "references": references,
    }


def write_model_spec(docs_path: Path | None = None) -> Path:
    """Write the diagram spec and return where it landed."""
    out_path = SPEC_PATH if docs_path is None else docs_path / "_generated"
    docs = out_path.parent
    spec = build_spec(cross_refs=_load_cross_refs(docs))
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / SPEC_NAME
    header = (
        "# Generated by scripts/_inventory_model.py from DASCore's inventory\n"
        "# models. Do not edit; edit the models, or the diagram's presentation\n"
        "# in that script, and rebuild the docs.\n"
    )
    text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=10_000)
    path.write_text(header + text, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Wrote {write_model_spec()}")  # noqa: T201
