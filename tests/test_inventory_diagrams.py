"""
Tests which keep the inventory tutorial's mermaid diagrams honest.

The diagrams are hand-written, so nothing stops them describing a model that no
longer looks like that. These tests read the diagrams back out of the page and
check each edge against the models: that the source is a model, that the field
labelling the edge exists on it, that the target is a model that field can
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
from dascore.core.inventory import InventoryModel

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
_ARROW = re.compile(r"-\.?->")


def _read_diagrams():
    """Return the page's edges, and the edge lines which could not be read.

    The unread lines matter as much as the edges: an edge this module cannot
    parse is an edge it cannot check, and dropping it silently is how the whole
    file goes vacuous one arrow at a time.
    """
    if not _DOCS_PRESENT:  # nothing to read; every test here is skipped
        return (), ()
    edges, unread = [], []
    for block in _BLOCK.findall(_PAGE_PATH.read_text()):
        for line in block.splitlines():
            if (match := _EDGE.match(line)) is None:
                if _ARROW.search(line):
                    unread.append(line.strip())
                continue
            source, arrow, field, node, label = match.groups()
            # A labelled node stands for the several types its label names.
            targets = tuple(label.split(" · ")) if label else (node,)
            edges.append((source, arrow == "-.->", field, targets))
    return tuple(edges), tuple(unread)


def _accepted_models(model, field):
    """Return the models the field holds, and whether it accepts a reference.

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
        elif isinstance(node, type) and issubclass(node, InventoryModel):
            found.add(node)
            referenced = referenced or in_reference_union

    _walk(annotation, False)
    return found, referenced


_EDGES, _UNREAD = _read_diagrams()


class TestDiagramEdges:
    """Every edge drawn in the tutorial has to be a field the models have."""

    def test_the_page_draws_edges(self):
        """A regex which quietly matched nothing would pass every test below."""
        assert len(_EDGES) >= 12

    def test_every_edge_line_is_read(self):
        """An arrow this module cannot parse is an arrow it cannot check."""
        assert not _UNREAD, f"Unparsed mermaid edges: {_UNREAD}"

    @pytest.mark.parametrize(("source", "dashed", "field", "targets"), _EDGES)
    def test_an_edge_matches_the_models(self, source, dashed, field, targets):
        """The source, the field, the targets, and the arrow all have to agree."""
        model = getattr(inventory_module, source, None)
        assert isinstance(model, type) and issubclass(model, InventoryModel), (
            f"{source} is not an inventory model."
        )
        assert field in model.model_fields, f"{source} has no field {field!r}."

        accepted, referenced = _accepted_models(model, field)
        names = {x.__name__ for x in accepted}
        for target in targets:
            assert target in names, f"{source}.{field} cannot hold a {target}."

        arrow = "dashed" if dashed else "solid"
        assert dashed == referenced, (
            f"{source}.{field} is drawn {arrow}, which says the wrong thing "
            "about whether it accepts a resource_id in place of the object."
        )
