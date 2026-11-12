"""One authored page serves the skills CLI and ordinary documentation."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

import dascore as dc
from dascore.cli import main
from dascore.utils import doc_skills
from dascore.utils.doc_corpus import DOC_PATH, SKILL_PATH, DocumentationError

_SCRIPTS = Path(__file__).parents[2] / "scripts"


def _load_script(name):
    """Import a documentation build script, which lives outside the package."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations look their module up by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestSkills:
    """Exercise the real packaged catalog and command-line surface."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Keep generated Markdown isolated from the user's cache."""
        with dc.config_context(docs_cache_dir=tmp_path):
            yield

    def test_skill_files(self):
        """Each shipped skill is a named file, served by the page including it."""
        shipped = {}
        for path in sorted(SKILL_PATH.glob("*/SKILL.md")):
            front = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
            assert front["name"] == path.parent.name and front["description"]
            shipped[front["name"]] = front["description"]
        skills = doc_skills.get_skills()
        assert list(skills) == sorted(shipped)
        assert skills["dascore"]["target"] == "recipes/dascore"
        for name, info in skills.items():
            assert info["description"] == shipped[name]
            # The entry point's page is its routing recipe, not the bootstrap file.
            if name == "dascore" or info["target"] is None:
                continue
            page = (DOC_PATH / f"{info['target']}.qmd").read_text(encoding="utf-8")
            include = f"{{{{< include ../../.agents/skills/{name}/SKILL.md >}}}}"
            assert page.count(include) == 1
            # A blank line before the include makes Quarto print the skill's
            # front matter on pages whose code cells it executes.
            if "engine: jupyter" in page:
                assert f"\n---\n{include}\n" in page

    @pytest.mark.skipif(not _SCRIPTS.exists(), reason="needs a repository checkout")
    def test_build_tools_read_skill_files(self):
        """Code cells and links that moved into skill files are still checked."""
        generator = _load_script("generate_doc_code_tests")
        page = DOC_PATH / "recipes" / "tunnel_inventory.qmd"
        assert generator.extract_qmd_file(page).chunks
        validator = _load_script("_validate_links")
        skill = SKILL_PATH / "dascore-make-inventory" / "SKILL.md"
        assert skill in set(validator.get_qmd_files())

    def test_cli(self, cache, capsys):
        """Discovery lists canonical pages and retrieval serves their contents."""
        assert main(["skills"]) == 0
        listed = capsys.readouterr().out
        for name, info in doc_skills.get_skills().items():
            assert f"{name}: {info['description']}" in listed
        for name, info in doc_skills.get_skills().items():
            assert main(["skill", name]) == 0
            printed = capsys.readouterr().out
            assert printed == doc_skills.read_skill(name)
            assert re.search(rf"^# {re.escape(info['title'])}$", printed, re.M)
            # The skill file's own front matter stays out of the printed page.
            assert f"name: {name}" not in printed

    def test_install(self, tmp_path, capsys, monkeypatch):
        """One command installs an entry for every skill, replacing older ones."""
        target = tmp_path / "agent" / "skills"
        for name in ("dascore", "dascore-make-inventory"):
            stale = target / name / "SKILL.md"
            stale.parent.mkdir(parents=True)
            stale.write_text("An older copy.")
        # Agents often quote the path, which leaves `~` for the command to expand.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert main(["skills", "--install", "~/agent/skills"]) == 0
        installed = capsys.readouterr().out.split()
        sources = sorted(SKILL_PATH.glob("*/SKILL.md"))
        assert installed == [str(target / x.parent.name / x.name) for x in sources]
        # Each of these serves a project whose DASCore may have no command line.
        assert doc_skills._STANDALONE == {
            "dascore",
            "dascore-migrate-to-v1",
            "dascore-start-task",
            "dascore-test-beta",
        }
        descriptions = {k: v["description"] for k, v in doc_skills.get_skills().items()}
        for source in sources:
            name = source.parent.name
            shipped = source.read_text(encoding="utf-8")
            entry = (target / name / source.name).read_text(encoding="utf-8")
            if name in doc_skills._STANDALONE:
                # These must work where the project's DASCore has no CLI.
                assert entry == shipped
                continue
            # Other entries keep the metadata and defer the procedure to the CLI.
            front, body = entry.removeprefix("---\n").split("\n---\n", 1)
            assert yaml.safe_load(front) == {
                "name": name,
                "description": descriptions[name],
            }
            assert f"-m dascore skill {name}`" in body
            procedure = shipped.split("\n---\n", 1)[1].splitlines()
            assert not {x for x in procedure if x.strip()} & set(body.splitlines())

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
    def test_install_over_links(self, tmp_path):
        """A linked or dangling skill folder is replaced, not written through."""
        other = tmp_path / "other-environment" / "dascore-make-inventory"
        other.mkdir(parents=True)
        procedure = other / "SKILL.md"
        procedure.write_text("Another installation's procedure.")
        target = tmp_path / "skills"
        target.mkdir()
        (target / "dascore-make-inventory").symlink_to(other, target_is_directory=True)
        (target / "dascore-add-format").symlink_to(
            tmp_path / "deleted", target_is_directory=True
        )
        doc_skills.install_skills(target)
        assert procedure.read_text() == "Another installation's procedure."
        for name in ("dascore-make-inventory", "dascore-add-format"):
            entry = target / name / "SKILL.md"
            assert not entry.parent.is_symlink()
            assert f"-m dascore skill {name}`" in entry.read_text(encoding="utf-8")

    def test_install_into_package(self, tmp_path, monkeypatch):
        """Installing into a directory that is the shipped skills changes nothing."""
        shipped = tmp_path / "skills"
        for name in ("dascore", "dascore-make-inventory"):
            source = SKILL_PATH / name / "SKILL.md"
            copy = shipped / name / "SKILL.md"
            copy.parent.mkdir(parents=True)
            copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(doc_skills, "SKILL_PATH", shipped)
        before = (shipped / "dascore-make-inventory" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        doc_skills.install_skills(shipped)
        after = (shipped / "dascore-make-inventory" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert before == after

    @pytest.mark.parametrize(
        "args, code, message",
        [
            (["skill"], 2, "Provide a skill name"),
            (
                ["skill", "unknown"],
                1,
                "Available skills: dascore, dascore-add-format",
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
    """Catch mistakes in how pages and skill files pair, without running recipes."""

    @pytest.fixture
    def sources(self, tmp_path, monkeypatch):
        """Use one tiny skill file and the page including it."""
        skill = tmp_path / ".agents" / "skills" / "example" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\nname: example\ndescription: First description\n---\n\n"
            "# Example title\n\nProcedure.\n"
        )
        docs = tmp_path / "docs"
        docs.mkdir()
        recipe = docs / "recipe.qmd"
        recipe.write_text(
            "---\ntitle: Recipe\n---\n"
            "{{< include ../.agents/skills/example/SKILL.md >}}\n"
        )
        pages = [recipe]
        monkeypatch.setattr(doc_skills, "DOC_PATH", docs)
        monkeypatch.setattr(doc_skills, "SKILL_PATH", skill.parent.parent)
        monkeypatch.setattr(doc_skills, "source_paths", lambda: list(pages))
        return skill, recipe, pages

    def test_including_page(self, sources):
        """The including page gives the title; the file gives the description."""
        skill, _, _ = sources
        info = doc_skills.get_skills()["example"]
        assert info == {
            "target": "recipe",
            "title": "Recipe",
            "description": "First description",
        }
        skill.write_text(skill.read_text().replace("First", "Updated"))
        assert (
            doc_skills.get_skills()["example"]["description"] == "Updated description"
        )

    def test_named_in_front_matter(self, sources):
        """A page holding a skill's procedure itself names it in its front matter."""
        _, recipe, _ = sources
        recipe.write_text("---\ntitle: Own text\nskill: example\n---\nProcedure.\n")
        assert doc_skills.get_skills()["example"]["target"] == "recipe"

    def test_page_less_skill(self, sources):
        """A skill no page includes serves its file, titled by its heading."""
        _, _, pages = sources
        pages.clear()
        info = doc_skills.get_skills()["example"]
        assert info == {
            "target": None,
            "title": "Example title",
            "description": "First description",
        }
        assert doc_skills.read_skill("example") == "# Example title\n\nProcedure.\n"

    @pytest.mark.parametrize(
        "text",
        [
            "---\nname: example\ndescription: No heading\n---\nProcedure.\n",
            "---\nname: example\n---\n# Title\n",
        ],
    )
    def test_bad_page_less_skill(self, sources, text):
        """A page-less skill needs a file with a heading and a description."""
        skill, _, pages = sources
        pages.clear()
        skill.write_text(text)
        with pytest.raises(DocumentationError, match="needs a title and description"):
            doc_skills.get_skills()

    def test_untitled_page(self, sources):
        """The page serving a skill needs a title."""
        _, recipe, _ = sources
        recipe.write_text(recipe.read_text().replace("title: Recipe\n", ""))
        with pytest.raises(DocumentationError, match="needs a title and description"):
            doc_skills.get_skills()

    def test_two_pages(self, sources):
        """A skill is served by one page."""
        _, recipe, pages = sources
        copy = recipe.with_name("copy.qmd")
        copy.write_text(recipe.read_text())
        pages.append(copy)
        with pytest.raises(DocumentationError, match="served by both"):
            doc_skills.get_skills()

    def test_unknown_skill(self, sources):
        """A page cannot serve a skill that does not ship."""
        _, recipe, _ = sources
        recipe.write_text("---\ntitle: Recipe\nskill: missing\n---\n")
        with pytest.raises(DocumentationError, match="unknown skills"):
            doc_skills.get_skills()

    def test_bad_name(self, sources):
        """Skill folders need lowercase, hyphenated names."""
        skill, _, pages = sources
        pages.clear()
        bad = skill.parent.with_name("Bad_Name") / "SKILL.md"
        bad.parent.mkdir()
        bad.write_text(skill.read_text())
        with pytest.raises(DocumentationError, match="Invalid skill name"):
            doc_skills.get_skills()
