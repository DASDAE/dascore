"""
Test notebook links and images, the marimo setup cell, mirrored data paths, and
that each tutorial name has one defining block.

These can break in the browser despite a successful build. Build failures are
caught by quarto and marimo during the doc build; these tests need neither.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from dascore.utils import downloader

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
_SCRIPT_PATH = _SCRIPTS / "build_notebooks.py"

# These checks exercise checkout-only build tools; the sdist omits scripts/.
pytestmark = pytest.mark.skipif(
    not _SCRIPT_PATH.is_file(), reason="scripts/ is not installed"
)


def _load_module(name):
    """Import a script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up while the class is created.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_notebooks():
    """The loaded build_notebooks module."""
    return _load_module("build_notebooks")


@pytest.fixture()
def tutorial_page(build_notebooks):
    """A path standing in for a page in the tutorial directory."""
    return build_notebooks.DOCS / "tutorial" / "patch.qmd"


@pytest.fixture()
def site_url(build_notebooks):
    """The default base url notebooks link back to."""
    return build_notebooks.DEFAULT_SITE_URL


class TestQmdLinkToUrl:
    """Tests for turning .qmd link targets into published urls."""

    def test_site_absolute_link(self, build_notebooks, tutorial_page, site_url):
        """Links the cross-reference filter emits are already site rooted."""
        out = build_notebooks.qmd_link_to_url(
            "/api/dascore/core/patch/Patch.qmd", "", tutorial_page, site_url
        )
        assert out == "https://dascore.org/api/dascore/core/patch/Patch.html"

    def test_sibling_link(self, build_notebooks, tutorial_page, site_url):
        """A bare page name resolves against the page's own directory."""
        out = build_notebooks.qmd_link_to_url("coords.qmd", "", tutorial_page, site_url)
        assert out == "https://dascore.org/tutorial/coords.html"

    def test_parent_relative_link(self, build_notebooks, tutorial_page, site_url):
        """A link out of the tutorial directory keeps its real destination."""
        out = build_notebooks.qmd_link_to_url(
            "../notes/patch_attrs.qmd", "", tutorial_page, site_url
        )
        assert out == "https://dascore.org/notes/patch_attrs.html"

    def test_anchor_is_kept(self, build_notebooks, tutorial_page, site_url):
        """An anchor should survive the rewrite."""
        out = build_notebooks.qmd_link_to_url(
            "concepts.qmd", "#units", tutorial_page, site_url
        )
        assert out == "https://dascore.org/tutorial/concepts.html#units"

    def test_rewrite_links_leaves_other_links_alone(
        self, build_notebooks, tutorial_page, site_url
    ):
        """Only .qmd targets are rewritten."""
        source = ["see [x](https://example.com/a.html) and [y](coords.qmd)\n"]
        out = "".join(build_notebooks.rewrite_links(source, tutorial_page, site_url))
        assert "https://example.com/a.html" in out
        assert "https://dascore.org/tutorial/coords.html" in out
        assert ".qmd" not in out


def _notebook(cells):
    """Build a minimal notebook around some cells."""
    return {"cells": cells, "metadata": {"kernelspec": {"name": "local"}}}


def _code(source):
    """Build a code cell."""
    return {"cell_type": "code", "source": [source]}


class TestPostProcess:
    """Tests for preparing a rendered notebook for marimo."""

    wheel = Path("/dist/dascore-1.0-py3-none-any.whl")

    @pytest.fixture()
    def processed(self, build_notebooks, tmp_path, tutorial_page, site_url):
        """A notebook run through post_process."""
        cells = [
            {"cell_type": "markdown", "source": ["[a](coords.qmd)\n"]},
            _code("import dascore"),
        ]
        path = tmp_path / "nb.ipynb"
        path.write_text(json.dumps(_notebook(cells)))
        return build_notebooks.post_process(path, tutorial_page, site_url, self.wheel)

    def test_setup_cell_is_first(self, processed):
        """Marimo convert makes the first cell named setup the setup cell."""
        first = processed["cells"][0]
        assert first["metadata"]["marimo"]["name"] == "setup"
        assert "get_fetcher" in first["source"]
        # Every cell sees the setup cell's names, and the markdown needs mo.
        assert first["source"].startswith("import marimo as mo\n")

    def test_header_installs_bundled_wheel(self, build_notebooks, processed):
        """The released dascore does not install in Pyodide; the wheel must."""
        header = processed["metadata"]["marimo"]["header"]
        assert header.startswith("# /// script")
        assert f'"dascore @ {self.wheel}"' in header
        for package in build_notebooks.EXTRA_PACKAGES:
            assert f'"{package}"' in header

    def test_links_point_at_site(self, processed):
        """A relative .qmd link means nothing inside the notebook."""
        markdown = "".join(processed["cells"][1]["source"])
        assert markdown == "[a](https://dascore.org/tutorial/coords.html)\n"

    def test_page_with_only_empty_code_is_skipped(
        self, build_notebooks, tmp_path, tutorial_page, site_url
    ):
        """A page whose only code cells are empty has nothing to run.

        Such a page would otherwise yield a notebook holding just the setup
        cell, which the tutorial page would then advertise as runnable.
        """
        cells = [{"cell_type": "markdown", "source": ["hi\n"]}, _code("  \n")]
        path = tmp_path / "diagrams.ipynb"
        path.write_text(json.dumps(_notebook(cells)))
        assert (
            build_notebooks.post_process(path, tutorial_page, site_url, self.wheel)
            is None
        )


class TestAttachments:
    """Quarto stores local images as attachments, which marimo convert drops."""

    def test_attachment_becomes_data_url(self, build_notebooks):
        """The image travels inside the markdown, the only part marimo keeps."""
        cell = {
            "cell_type": "markdown",
            "source": ['<img src="attachment:../_static/a.png" />\n'],
            "attachments": {"../_static/a.png": {"image/png": "iVBORw0K"}},
        }
        out = build_notebooks.inline_attachments(cell)
        assert out == ['<img src="data:image/png;base64,iVBORw0K" />\n']
        notebook = _notebook([{**cell, "source": out}])
        assert not build_notebooks.find_unrewritten_links(notebook)

    def test_missing_attachment_is_reported(self, build_notebooks):
        """An image without its payload would render broken, so the build stops."""
        cell = {"cell_type": "markdown", "source": ['<img src="attachment:b.svg" />']}
        cell["source"] = build_notebooks.inline_attachments(cell)
        out = build_notebooks.find_unrewritten_links(_notebook([cell]))
        assert out == ["attachment:b.svg"]


class TestMirroredData:
    """Example data ships with the site so tutorials do not hit GitHub."""

    @pytest.fixture()
    def fetcher(self, tmp_path, monkeypatch):
        """A fresh pooch fetcher standing in for dascore's shared one."""
        fetcher = downloader._get_fetcher.__wrapped__(str(tmp_path / "cache"))
        monkeypatch.setattr(downloader, "get_fetcher", lambda: fetcher)
        return fetcher

    @pytest.fixture()
    def browser(self, monkeypatch):
        """Fake the browser: emscripten, with the notebook served from a site."""
        site = SimpleNamespace(notebook_location=lambda: Path("/site/marimo"))
        monkeypatch.setitem(sys.modules, "marimo", site)
        monkeypatch.setattr(sys, "platform", "emscripten")

    def test_browser_downloads_from_site(self, build_notebooks, fetcher, browser):
        """In the browser, pooch fetches each mirrored file from the site."""
        exec(build_notebooks.SETUP_SOURCE, {})

        for name in build_notebooks.MIRRORED_DATA:
            assert fetcher.get_url(name) == f"/site/marimo/public/data/{name}"

    def test_unmirrored_files_still_use_registry(
        self, build_notebooks, fetcher, browser
    ):
        """Only the mirrored files move; anything else downloads as before."""
        other = next(
            name
            for name in fetcher.registry
            if name not in build_notebooks.MIRRORED_DATA
        )
        before = fetcher.get_url(other)

        exec(build_notebooks.SETUP_SOURCE, {})

        assert fetcher.get_url(other) == before

    def test_setup_is_inert_outside_browser(
        self, build_notebooks, fetcher, monkeypatch
    ):
        """Running a notebook locally keeps downloading from the registry."""
        # The test suite also runs under Pyodide, which is emscripten.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setitem(sys.modules, "marimo", SimpleNamespace())
        before = dict(fetcher.urls)

        exec(build_notebooks.SETUP_SOURCE, {})

        assert fetcher.urls == before

    def test_mirror_layout_matches_setup_cell(
        self, build_notebooks, tmp_path, monkeypatch
    ):
        """mirror_data writes each file where the setup cell points pooch."""
        stub = tmp_path / "stub"
        stub.mkdir()
        for name in build_notebooks.MIRRORED_DATA:
            (stub / name).write_bytes(name.encode())
        monkeypatch.setattr(build_notebooks, "fetch", lambda name: stub / name)

        build_notebooks.mirror_data(tmp_path / "public")

        for name in build_notebooks.MIRRORED_DATA:
            assert (tmp_path / "public" / "data" / name).read_bytes() == name.encode()


def _defined_names(code: str) -> set[str]:
    """Return the module-scope names bound by assignment, import, def or class."""
    names = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(child.name)
                continue
            if isinstance(
                child,
                ast.Lambda
                | ast.ListComp
                | ast.SetComp
                | ast.DictComp
                | ast.GeneratorExp,
            ):
                continue
            if isinstance(child, ast.Import | ast.ImportFrom):
                names.update((a.asname or a.name).split(".")[0] for a in child.names)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            visit(child)

    visit(ast.parse(code))
    return {name for name in names if not name.startswith("_")}


class TestTutorialNames:
    """Each tutorial page defines a top-level name in one block only."""

    def test_no_name_defined_twice(self):
        """Marimo convert renames a redefinition, so the notebook drifts.

        A name assigned in two blocks becomes `name_1` in the second notebook
        cell, and a repeated import is dropped from it, so the notebook no
        longer matches the page the reader came from.
        """
        extract = _load_module("generate_doc_code_tests").extract_qmd_file
        repeated = {}
        for page in sorted(
            (_REPO_ROOT / "dascore" / "docs" / "tutorial").glob("*.qmd")
        ):
            blocks = defaultdict(list)
            for chunk in extract(page).chunks:
                for name in _defined_names(chunk.source):
                    blocks[name].append(chunk.start_line)
            if dupes := {k: v for k, v in blocks.items() if len(v) > 1}:
                repeated[page.name] = dupes
        assert not repeated


class TestSiteUrl:
    """The notebooks must link to the deployment they were built for."""

    def test_environment_sets_the_link_target(
        self, build_notebooks, tutorial_page, monkeypatch
    ):
        """Dev docs publish to netlify, releases to dascore.org.

        Built from dev but pointed at the released site, every link to a page
        added since the last release 404s, and nothing about the notebook
        looks wrong.
        """
        monkeypatch.setenv("DASCORE_DOC_SITE_URL", "https://dascore.netlify.app/")
        out = build_notebooks.qmd_link_to_url(
            "coords.qmd", "", tutorial_page, build_notebooks.get_site_url()
        )
        assert out == "https://dascore.netlify.app/tutorial/coords.html"
