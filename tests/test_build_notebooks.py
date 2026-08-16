"""
Tests for scripts/build_notebooks.py, which renders the tutorial pages into
the notebooks the JupyterLite site serves.

The rendering itself needs quarto, so these cover the parts that decide what
the reader ends up with: where a cross-page link points, and what the notebook
looks like once it has been adjusted for a browser kernel. Both fail quietly
when they are wrong -- a bad link 404s and a missed output ships stale data --
so they are worth pinning.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "build_notebooks.py"

# The sdist ships tests but not scripts/, on the same terms as
# test_inventory_diagrams.py skipping when the docs tree is absent.
pytestmark = pytest.mark.skipif(
    not _SCRIPT_PATH.is_file(), reason="scripts/ is not installed"
)


def _load_module():
    """Import the build script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("build_notebooks", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_notebooks():
    """The loaded build_notebooks module."""
    return _load_module()


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


class TestMermaidRegex:
    """Executable mermaid blocks need chrome, so they become plain fences."""

    def test_executable_block_becomes_fence(self, build_notebooks):
        """```{mermaid} should lose its braces."""
        text = "intro\n```{mermaid}\ngraph TD;\n```\n"
        out = build_notebooks.MERMAID_REGEX.sub("```mermaid", text)
        assert "```mermaid\ngraph TD;" in out
        assert "{mermaid}" not in out

    def test_indented_python_block_untouched(self, build_notebooks):
        """Only fences at the start of a line are converted."""
        text = "```{python}\nx = 1\n```\n"
        assert build_notebooks.MERMAID_REGEX.sub("```mermaid", text) == text


def _notebook(cells):
    """Build a minimal notebook around some cells."""
    return {"cells": cells, "metadata": {"kernelspec": {"name": "local"}}}


def _code(source, outputs=None):
    """Build a code cell."""
    return {
        "cell_type": "code",
        "source": [source],
        "outputs": outputs if outputs is not None else [],
        "execution_count": 3,
    }


class TestPostProcess:
    """Tests for adjusting a rendered notebook for the browser."""

    @pytest.fixture()
    def processed(self, build_notebooks, tmp_path, tutorial_page, site_url):
        """A notebook run through post_process."""
        cells = [
            {"cell_type": "markdown", "source": ["[a](coords.qmd)\n"]},
            _code("import dascore", outputs=[{"text": "stale"}]),
            _code("   \n"),
        ]
        path = tmp_path / "nb.ipynb"
        path.write_text(json.dumps(_notebook(cells)))
        return build_notebooks.post_process(path, tutorial_page, site_url)

    def test_setup_cell_is_first(self, processed):
        """The reader has to install dascore before anything else runs."""
        first = "".join(processed["cells"][0]["source"])
        assert "%pip install" in first
        assert "dascore" in first

    def test_kernel_points_at_pyodide(self, build_notebooks, processed):
        """A local interpreter path would not exist in the browser."""
        assert processed["metadata"]["kernelspec"] == build_notebooks.KERNELSPEC

    def test_outputs_are_dropped(self, processed):
        """Shipping stale output would misrepresent what the code does."""
        assert all(not cell.get("outputs") for cell in processed["cells"])
        code = [c for c in processed["cells"] if c["cell_type"] == "code"]
        assert all(c["execution_count"] is None for c in code)

    def test_links_are_rewritten(self, processed):
        """Markdown links should point at the published site."""
        markdown = [c for c in processed["cells"] if c["cell_type"] == "markdown"]
        assert "https://dascore.org/tutorial/coords.html" in "".join(
            markdown[0]["source"]
        )

    def test_blank_code_cells_are_removed(self, processed):
        """Non-python blocks render as empty cells, which help nobody."""
        code = [c for c in processed["cells"] if c["cell_type"] == "code"]
        # Only the setup cell and the real one survive.
        assert len(code) == 2

    def test_page_without_code_is_skipped(
        self, build_notebooks, tmp_path, tutorial_page, site_url
    ):
        """A prose-only page should not become a notebook."""
        path = tmp_path / "prose.ipynb"
        path.write_text(
            json.dumps(_notebook([{"cell_type": "markdown", "source": ["hi\n"]}]))
        )
        assert build_notebooks.post_process(path, tutorial_page, site_url) is None

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
        assert build_notebooks.post_process(path, tutorial_page, site_url) is None


class TestExtraPackages:
    """The browser kernel needs a few packages dascore does not require."""

    def test_setup_cell_installs_extras(self, build_notebooks):
        """Each extra the doc examples import must be in the setup cell."""
        for package in build_notebooks.EXTRA_PACKAGES:
            assert package in build_notebooks.SETUP_SOURCE


class TestSiteUrl:
    """The notebooks must link to the deployment they were built for."""

    def test_defaults_to_stable_site(self, build_notebooks, monkeypatch):
        """Without configuration the released docs are the right target."""
        monkeypatch.delenv("DASCORE_DOC_SITE_URL", raising=False)
        assert build_notebooks.get_site_url() == build_notebooks.DEFAULT_SITE_URL

    def test_environment_overrides(self, build_notebooks, monkeypatch):
        """Dev docs publish elsewhere, so the base url has to be settable."""
        monkeypatch.setenv("DASCORE_DOC_SITE_URL", "https://dascore.netlify.app/")
        assert build_notebooks.get_site_url() == "https://dascore.netlify.app"

    def test_links_follow_the_configured_site(self, build_notebooks, tutorial_page):
        """A page link should sit under whichever site was requested."""
        out = build_notebooks.qmd_link_to_url(
            "coords.qmd", "", tutorial_page, "https://dascore.netlify.app"
        )
        assert out == "https://dascore.netlify.app/tutorial/coords.html"
