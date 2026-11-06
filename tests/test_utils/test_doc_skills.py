"""One authored page serves the skills CLI and ordinary documentation."""

from __future__ import annotations

import pytest
import yaml

import dascore as dc
from dascore.cli import main
from dascore.utils import doc_skills
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import DOC_PATH, SKILL_PATH, DocumentationError

# The shipped catalog, in listing order.
_CATALOG = {
    "dascore": "recipes/dascore",
    "make-inventory": "recipes/tunnel_inventory",
    "make-annotations": "tutorial/annotations",
    "make-namespace": "contributing/extending_dascore",
    "migrate-to-v1": "migrations/dascore_1.0_migration",
    "test-beta": "migrations/test_latest_beta",
    "add-format": "contributing/new_format",
    "start-task": "contributing/start_task",
    "draft-release": "contributing/draft_release",
}


class TestSkills:
    """Exercise the real packaged catalog and command-line surface."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Keep generated Markdown isolated from the user's cache."""
        with dc.config_context(docs_cache_dir=tmp_path):
            yield

    def test_catalog(self):
        """The catalog exposes each catalogued procedure with its metadata."""
        skills = doc_skills.get_skills()
        assert {name: info["target"] for name, info in skills.items()} == _CATALOG
        assert all(x["title"] and x["description"] for x in skills.values())

    def test_skill_files(self):
        """Each shipped skill is a named file, and catalogued pages include theirs."""
        shipped = {}
        for path in sorted(SKILL_PATH.glob("*/SKILL.md")):
            front = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
            assert front["name"] == path.parent.name and front["description"]
            shipped[front["name"]] = front["description"]
        assert set(_CATALOG) <= set(shipped)
        for name, info in doc_skills.get_skills().items():
            # The entry point's page is its routing recipe, not the bootstrap file.
            if name == "dascore":
                continue
            assert info["description"] == shipped[name]
            # A blank line before the include makes Quarto print the skill's
            # front matter on pages whose code cells it executes.
            page = (DOC_PATH / f"{info['target']}.qmd").read_text(encoding="utf-8")
            include = f"{{{{< include ../../.agents/skills/{name}/SKILL.md >}}}}"
            assert page.endswith(f"\n---\n{include}\n")

    @pytest.mark.parametrize("name", list(_CATALOG))
    def test_same_document(self, cache, name):
        """Skill retrieval returns the exact canonical cached documentation page."""
        target = doc_skills.get_skills()[name]["target"]
        text = doc_skills.read_skill(name)
        with documentation_cache() as (root, manifest):
            assert text == read_document(root, manifest, target)

    def test_cli(self, cache, capsys):
        """Discovery lists canonical pages and retrieval serves their contents."""
        assert main(["skills"]) == 0
        listed = capsys.readouterr().out
        for name, info in doc_skills.get_skills().items():
            assert f"{name}: {info['description']}" in listed
        assert main(["skill", "make-inventory"]) == 0
        assert capsys.readouterr().out == doc_skills.read_skill("make-inventory")

    def test_install(self, tmp_path, capsys):
        """One command copies every shipped skill and replaces an older copy."""
        target = tmp_path / "agent" / "skills"
        stale = target / "dascore" / "SKILL.md"
        stale.parent.mkdir(parents=True)
        stale.write_text("An older copy.")
        assert main(["skills", "--install", str(target)]) == 0
        installed = capsys.readouterr().out.split()
        sources = sorted(SKILL_PATH.glob("*/SKILL.md"))
        assert installed == [str(target / x.parent.name / x.name) for x in sources]
        assert {x.parent.name for x in sources} == set(_CATALOG)
        for source in sources:
            copy = target / source.parent.name / source.name
            assert copy.read_text(encoding="utf-8") == source.read_text(
                encoding="utf-8"
            )

    @pytest.mark.parametrize(
        "args, code, message",
        [
            (["skill"], 2, "Provide a skill name"),
            (
                ["skill", "unknown"],
                1,
                f"Available skills: {', '.join(_CATALOG)}",
            ),
            (["skill", "dascore", "--bootstrap"], 2, "Use a skill name"),
        ],
    )
    def test_errors(self, args, code, message, capsys):
        """Usage errors and failed lookups report their distinct causes."""
        assert main(args) == code
        captured = capsys.readouterr()
        assert not captured.out and message in captured.err
        if code == 2:
            assert "usage:" in captured.err

    def test_bootstrap(self, capsys, tmp_path):
        """Export is verbatim and does not require building the corpus."""
        with dc.config_context(docs_cache_dir=tmp_path / "cache"):
            assert main(["skill", "--bootstrap"]) == 0
        text = capsys.readouterr().out
        assert text == doc_skills.read_bootstrap()
        front = yaml.safe_load(text.split("---", 2)[1])
        assert front["name"] == "dascore" and front["description"]
        assert not (tmp_path / "cache").exists()


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
