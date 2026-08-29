"""Tests for the inventory tutorial's explanatory diagrams."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

import pytest

from dascore.core.inventory import Acquisition, FiberArray, Inventory, OpticalPath

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_PATH = _REPO_ROOT / "docs"
_PAGE_PATH = _DOC_PATH / "tutorial" / "inventory.qmd"
_STATIC_PATH = _DOC_PATH / "_static"
_DIAGRAMS = {
    "inventory_hierarchy.svg": {"station", "channel", "fiber", "acquisition"},
    "optical_path_concept.svg": {"components", "geometry", "labels", "coupling"},
}
_GRAPHIC_ELEMENTS = {
    "circle",
    "ellipse",
    "image",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "text",
    "use",
}
_MODEL_FIELDS = {
    "inventory_hierarchy.svg": {
        Inventory: {"resources"},
        FiberArray: {"optical_paths", "acquisitions"},
        OpticalPath: {"optical_components", "geometry", "labels", "coupling"},
        Acquisition: {"interrogator", "distance_map"},
    },
    "optical_path_concept.svg": {
        OpticalPath: {"optical_components", "geometry", "labels", "coupling"},
    },
}

# The sdist grafts tests but ships only docs/LICENSE. Key this on a page the
# full documentation tree always contains, not on the files under test.
_DOCS_PRESENT = (_DOC_PATH / "index.qmd").is_file()

pytestmark = pytest.mark.skipif(
    not _DOCS_PRESENT, reason="the documentation tree is not installed"
)


def _page_text() -> str:
    """Return the inventory tutorial source."""
    return _PAGE_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", _DIAGRAMS)
def test_page_references_each_diagram(name):
    """Each shipped diagram is used exactly once by the inventory page."""
    pattern = rf"^!\[[^]]+\]\(\.\./_static/{re.escape(name)}\)"
    assert len(re.findall(pattern, _page_text(), re.MULTILINE)) == 1


@pytest.mark.parametrize("name", _DIAGRAMS)
def test_diagram_is_valid_svg(name):
    """A missing, empty, or malformed image must fail before the docs build."""
    root = ElementTree.parse(_STATIC_PATH / name).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.get("viewBox")
    assert all(float(value) > 0 for value in root.get("viewBox").split()[2:])
    namespace = "{http://www.w3.org/2000/svg}"
    definitions = {
        element
        for defs in root.findall(f".//{namespace}defs")
        for element in defs.iter()
    }
    tags = {
        element.tag.rsplit("}", 1)[-1]
        for element in root.iter()
        if element not in definitions
    }
    assert tags & _GRAPHIC_ELEMENTS


@pytest.mark.parametrize(("name", "model_fields"), _MODEL_FIELDS.items())
def test_diagram_model_fields(name, model_fields):
    """Field names shown in the diagrams stay aligned with the models."""
    root = ElementTree.parse(_STATIC_PATH / name).getroot()
    text = " ".join(root.itertext()).lower()
    for model, fields in model_fields.items():
        assert fields <= model.model_fields.keys()
        assert all(field in text for field in fields)


@pytest.mark.parametrize(("name", "required_terms"), _DIAGRAMS.items())
def test_diagram_has_alt_text(name, required_terms):
    """The page gives each visual a useful text alternative."""
    pattern = rf"^!\[[^]]+\]\(\.\./_static/{re.escape(name)}\)\{{(?P<attrs>[^}}]+)\}}"
    match = re.search(pattern, _page_text(), re.MULTILINE)
    assert match is not None
    alt_match = re.search(r'fig-alt="(?P<alt>[^"]+)"', match["attrs"])
    assert alt_match is not None
    alt = alt_match["alt"].lower()
    assert len(alt.split()) >= 20
    assert all(term in alt for term in required_terms)
