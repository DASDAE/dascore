"""One authored recipe serves the skills CLI and ordinary documentation."""

from __future__ import annotations

import pytest
import yaml

import dascore as dc
from dascore.cli import main
from dascore.utils import doc_skills
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import DocumentationError


class TestSkills:
    """Exercise the real packaged catalog and command-line surface."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Keep generated Markdown isolated from the user's cache."""
        with dc.config_context(docs_cache_dir=tmp_path):
            yield

    def test_catalog(self):
        """The first release exposes the routing and inventory procedures."""
        skills = doc_skills.get_skills()
        assert set(skills) == {"dascore", "make-inventory"}
        assert all(x["title"] and x["description"] for x in skills.values())

    @pytest.mark.parametrize("name", ["dascore", "make-inventory"])
    def test_same_document(self, cache, name):
        """Skill retrieval returns the exact canonical cached documentation page."""
        target = doc_skills.get_skills()[name]["target"]
        text = doc_skills.read_skill(name)
        with documentation_cache() as (root, manifest):
            assert text == read_document(root, manifest, target)

    def test_cli(self, cache, capsys, hide_module):
        """Discovery and recipe retrieval work without the optional search engine."""
        hide_module("tantivy")
        assert main(["skills"]) == 0
        listed = capsys.readouterr().out
        for name, info in doc_skills.get_skills().items():
            assert f"{name}: {info['description']}" in listed
        assert main(["skill", "make-inventory"]) == 0
        assert capsys.readouterr().out == doc_skills.read_skill("make-inventory")

    @pytest.mark.parametrize(
        "args", [["skill"], ["skill", "unknown"], ["skill", "dascore", "--bootstrap"]]
    )
    def test_errors(self, args, capsys):
        """Missing, unknown, and contradictory skill requests fail usefully."""
        assert main(args) == 1
        captured = capsys.readouterr()
        assert not captured.out and "dascore:" in captured.err
        if "unknown" in args:
            assert "Available skills: dascore, make-inventory" in captured.err

    def test_bootstrap(self, capsys, tmp_path):
        """Export is verbatim and does not require building the corpus."""
        with dc.config_context(docs_cache_dir=tmp_path / "cache"):
            assert main(["skill", "--bootstrap"]) == 0
        text = capsys.readouterr().out
        assert text == doc_skills.read_bootstrap()
        front = yaml.safe_load(text.split("---", 2)[1])
        assert front["name"] == "dascore" and front["description"]
        assert not (tmp_path / "cache").exists()

    def test_inventory_round_trip(self):
        """The worked inventory survives serialization and representative attachment."""
        inventory = dc.get_example_inventory("tunnel")
        restored = dc.inventory(inventory.io.to_yaml())
        assert restored == inventory
        patch = dc.get_example_patch(
            "random_das",
            acquisition_key="XT.TUN1.00.DAS",
            time_min="2024-06-01",
            shape=(1776, 10),
        )
        spool = dc.spool(patch).attach_inventory(restored)
        assert len(spool.select(section="borehole")) == 3
        enriched = spool.enrich()[0]
        assert enriched.get_coord("z").values[1590] == -10.0


class TestCatalogValidation:
    """Catch catalog mistakes without executing any recipe."""

    @pytest.fixture
    def sources(self, tmp_path, monkeypatch):
        """Use a tiny real YAML catalog and QMD document."""
        recipe = tmp_path / "recipe.qmd"
        recipe.write_text(
            "---\ntitle: Recipe\ndescription: First description\n---\nProcedure."
        )
        catalog = tmp_path / "skills.yml"
        catalog.write_text("example: recipe\n")
        monkeypatch.setattr(doc_skills, "DOC_PATH", tmp_path)
        monkeypatch.setattr(doc_skills, "source_paths", lambda: [recipe])
        return catalog, recipe

    def test_source_metadata(self, sources):
        """Edits to the canonical recipe immediately update discovery metadata."""
        _, recipe = sources
        assert doc_skills.get_skills()["example"]["description"] == "First description"
        recipe.write_text(
            recipe.read_text().replace("First description", "Updated description")
        )
        assert (
            doc_skills.get_skills()["example"]["description"] == "Updated description"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "[]",
            "Bad_Name: recipe",
            "example: missing",
            "example: ../private",
            "example: 123",
        ],
    )
    def test_bad_catalog(self, sources, text):
        """Catalog entries must name known documentation pages with valid aliases."""
        catalog, _ = sources
        catalog.write_text(text)
        with pytest.raises(DocumentationError):
            doc_skills.get_skills()

    @pytest.mark.parametrize("field", ["title", "description"])
    def test_missing_metadata(self, sources, field):
        """Each skill requires human-readable discovery metadata in its recipe."""
        _, recipe = sources
        recipe.write_text(
            "\n".join(
                x
                for x in recipe.read_text().splitlines()
                if not x.startswith(field + ":")
            )
        )
        with pytest.raises(DocumentationError, match="needs a title and description"):
            doc_skills.get_skills()
