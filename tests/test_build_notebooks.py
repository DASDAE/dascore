"""
Tests for scripts/build_notebooks.py, which renders the tutorial pages into
the notebooks the JupyterLite site serves.

Scoped to the ways this build breaks *silently* -- a notebook that renders,
deploys and passes CI, then fails in a reader's browser. A mis-rewritten link
404s, the wrong kernelspec leaves the notebook unable to start, and mirroring
the data to a path pooch does not look in sends everyone back to downloading
it. None of those turn anything red on their own.

Failures that already announce themselves are deliberately not covered here:
a broken mermaid substitution makes quarto exit with "Chrome not found", an
unknown data file makes pooch raise on the registry lookup, and a tutorial
importing something the browser lacks fails the doc examples that
test_wasm.yml runs under Pyodide.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dascore.constants import DATA_VERSION

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

    def test_blank_code_cells_are_removed(self, processed):
        """Non-python blocks render as empty cells, which help nobody."""
        code = [c for c in processed["cells"] if c["cell_type"] == "code"]
        # Only the setup cell and the real one survive.
        assert len(code) == 2

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


class TestMirroredData:
    """Example data ships with the site so tutorials do not hit GitHub."""

    def test_setup_cell_points_pooch_at_the_mirror(self, build_notebooks):
        """Without the env var the reader downloads from GitHub instead."""
        assert "DFS_DATA_DIR" in build_notebooks.SETUP_SOURCE
        assert build_notebooks.BROWSER_DATA_DIR in build_notebooks.SETUP_SOURCE

    def test_layout_matches_what_pooch_expects(
        self, build_notebooks, tmp_path, monkeypatch
    ):
        """Pooch resolves DFS_DATA_DIR as `<dir>/<DATA_VERSION>/<name>`.

        Mirroring to any other shape leaves the files unreachable and the
        tutorials quietly downloading again, so the version subdirectory is
        the part worth pinning.
        """
        stub = tmp_path / "stub"
        stub.mkdir()
        for name in build_notebooks.MIRRORED_DATA:
            (stub / name).write_bytes(b"x")
        monkeypatch.setattr(
            "dascore.utils.downloader.fetch", lambda name, **kw: stub / name
        )

        contents = tmp_path / "contents"
        build_notebooks.mirror_data(contents)

        target = contents / build_notebooks.DATA_DIR_NAME / DATA_VERSION
        assert sorted(p.name for p in target.iterdir()) == sorted(
            build_notebooks.MIRRORED_DATA
        )
        # The browser path the setup cell exports must resolve to that dir.
        assert build_notebooks.BROWSER_DATA_DIR.endswith(build_notebooks.DATA_DIR_NAME)

    def test_rebuild_drops_stale_files(self, build_notebooks, tmp_path, monkeypatch):
        """A file dropped from the list must not linger in the site."""
        stub = tmp_path / "stub"
        stub.mkdir()
        for name in build_notebooks.MIRRORED_DATA:
            (stub / name).write_bytes(b"x")
        monkeypatch.setattr(
            "dascore.utils.downloader.fetch", lambda name, **kw: stub / name
        )
        contents = tmp_path / "contents"
        target = contents / build_notebooks.DATA_DIR_NAME / DATA_VERSION
        target.mkdir(parents=True)
        (target / "obsolete.h5").write_bytes(b"old")

        build_notebooks.mirror_data(contents)

        assert not (target / "obsolete.h5").exists()


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
