"""Verify authored-page aliases through the website cross-reference filter."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import _render_api

ROOT = Path(__file__).resolve().parents[1]


class TestAuthoredLinks:
    """New and legacy source paths resolve to the same published page."""

    def test_prefixed_alias(self, tmp_path, monkeypatch):
        """An extensionless reference keeps its published destination and anchor."""
        docs = tmp_path / "dascore" / "docs"
        page = docs / "tutorial" / "spool.qmd"
        page.parent.mkdir(parents=True)
        page.write_text("A tutorial.")
        monkeypatch.setattr(_render_api, "DOC_PATH", docs)
        mapping = _render_api._map_other_qmd_files(docs, docs / "api")
        spec = spec_from_file_location(
            "doc_link_filter", ROOT / "dascore/docs/filters/fill_links.py"
        )
        assert spec is not None and spec.loader is not None
        filter_module = module_from_spec(spec)
        spec.loader.exec_module(filter_module)
        monkeypatch.setattr(filter_module, "get_cross_ref_dict", lambda: mapping)
        for key in (
            "docs/tutorial/spool",
            "docs/tutorial/spool.qmd",
            "dascore/docs/tutorial/spool",
            "dascore/docs/tutorial/spool.qmd",
        ):
            link = {
                "t": "Link",
                "c": [
                    ["", [], []],
                    [{"t": "Str", "c": "spool"}],
                    [f"%60{key}%60#concatenate", ""],
                ],
            }
            raw = json.dumps(link, separators=(",", ":"))
            result = json.loads(filter_module.replace_links(link, raw))
            assert result["c"][-1][0] == "/tutorial/spool.qmd#concatenate"
