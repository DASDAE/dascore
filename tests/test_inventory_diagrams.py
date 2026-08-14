"""
Tests which keep the inventory tutorial's mermaid diagrams honest.

The diagrams are hand-written, so nothing stops them describing a model that no
longer looks like that. These tests read the diagrams back out of the page and
check each edge against the models: that the source is a model, that the field
labelling the edge exists on it, that the target is a type that field can
actually hold, and that a dashed edge is drawn exactly where the field accepts a
resource_id string in place of the object.
"""

from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

import pytest

import dascore.core.inventory as inventory_module

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_PATH = _REPO_ROOT / "docs"
_PAGE_PATH = _DOC_PATH / "tutorial" / "inventory.qmd"

# Run wherever the docs are, skip where they are not, on the same terms as
# test_changelog.py: the sdist grafts tests but ships only docs/LICENSE, so the
# directory existing does not mean the real docs tree is there. Deliberately not
# keyed on the tutorial itself, or deleting the page would skip rather than fail.
_DOCS_PRESENT = (_DOC_PATH / "index.qmd").is_file()

pytestmark = pytest.mark.skipif(
    not _DOCS_PRESENT, reason="the documentation tree is not installed"
)

_BLOCK = re.compile(r"^```\{mermaid\}\n(.*?)^```", re.MULTILINE | re.DOTALL)
# `Source -->|field| Target` or its dashed form, where a target may carry a
# label: `Components["FiberSegment · Splice"]`.
_EDGE = re.compile(
    r"^\s*(\w+)\s*(-->|-\.->)\s*\|(\w+)\|\s*(\w+)(?:\[\"([^\"]+)\"\])?\s*$"
)


def _diagram_edges():
    """Yield (source, dashed, field, targets) for every edge in the page."""
    for block in _BLOCK.findall(_PAGE_PATH.read_text()):
        for line in block.splitlines():
            if (match := _EDGE.match(line)) is None:
                continue
            source, arrow, field, node, label = match.groups()
            # A labelled node stands for the several types its label names.
            targets = tuple(label.split(" · ")) if label else (node,)
            yield source, arrow == "-.->", field, targets


def _accepted_types(model, field):
    """Return the types the field holds, and whether it accepts a reference.

    A reference is a `str` in the same union as a model, which is how the
    inventory spells "this may be a resource_id instead of the object".
    """
    annotation = get_type_hints(model)[field]
    found, referenced = set(), False

    def _walk(node, in_reference_union):
        nonlocal referenced
        origin = get_origin(node)
        if origin in (Union, types.UnionType):
            args = get_args(node)
            in_reference_union = str in args
            for arg in args:
                _walk(arg, in_reference_union)
        elif origin is not None:
            for arg in get_args(node):
                _walk(arg, in_reference_union)
        elif isinstance(node, type):
            found.add(node)
            referenced = referenced or (in_reference_union and node is not str)

    _walk(annotation, False)
    return found, referenced


_EDGES = tuple(_diagram_edges())


class TestDiagramEdges:
    """Every edge drawn in the tutorial has to be a field the models have."""

    def test_the_page_draws_edges(self):
        """A regex which quietly matched nothing would pass every test below."""
        assert len(_EDGES) >= 12

    @pytest.mark.parametrize(("source", "dashed", "field", "targets"), _EDGES)
    def test_an_edge_matches_the_models(self, source, dashed, field, targets):
        """The source, the field, the targets, and the arrow all have to agree."""
        model = getattr(inventory_module, source, None)
        assert model is not None, f"{source} is not an inventory model."
        assert field in model.model_fields, f"{source} has no field {field!r}."

        accepted, referenced = _accepted_types(model, field)
        names = {x.__name__ for x in accepted}
        for target in targets:
            assert target in names, f"{source}.{field} cannot hold a {target}."

        arrow = "dashed" if dashed else "solid"
        assert dashed == referenced, (
            f"{source}.{field} is drawn {arrow}, which says the wrong thing "
            "about whether it accepts a resource_id in place of the object."
        )
