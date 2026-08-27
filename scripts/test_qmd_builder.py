"""Tests for building Quarto config values."""

from __future__ import annotations

import pytest

_qmd_builder = pytest.importorskip("_qmd_builder")


class TestGetDascoreTitle:
    """Tests for DASCore title version formatting."""

    def test_release_version(self, monkeypatch):
        """Release versions are shown as-is."""
        monkeypatch.delenv("DASCORE_DOC_VERSION", raising=False)
        monkeypatch.setattr(_qmd_builder.dc, "__version__", "0.1.16")

        assert _qmd_builder._get_dascore_title() == "DASCore (0.1.16)"

    def test_dev_version(self, monkeypatch):
        """Dev versions strip local version metadata."""
        monkeypatch.delenv("DASCORE_DOC_VERSION", raising=False)
        monkeypatch.setattr(_qmd_builder.dc, "__version__", "0.1.16.dev19+gabc123")

        assert _qmd_builder._get_dascore_title() == "DASCore (0.1.16.dev19)"

    def test_doc_version_override(self, monkeypatch):
        """The docs version override controls the rendered site title."""
        monkeypatch.setenv("DASCORE_DOC_VERSION", "0.1.16")
        monkeypatch.setattr(_qmd_builder.dc, "__version__", "0.1.16.dev19+gabc123")

        assert _qmd_builder._get_dascore_title() == "DASCore (0.1.16)"


class TestGetRepoBranch:
    """Tests for the branch used by the docs' edit links."""

    def test_default_branch(self, monkeypatch):
        """Builds which don't set the branch point at master."""
        monkeypatch.delenv("DASCORE_DOC_BRANCH", raising=False)

        assert _qmd_builder._get_repo_branch() == "master"

    def test_branch_override(self, monkeypatch):
        """The dev doc build points edit links at dev."""
        monkeypatch.setenv("DASCORE_DOC_BRANCH", "dev")

        assert _qmd_builder._get_repo_branch() == "dev"


class TestCompactTocTree:
    """Tests for the section level API sidebar."""

    def test_one_entry_per_section(self, tmp_path):
        """A compact tree names sections, not the objects inside them."""
        api_path = tmp_path / "api"
        (api_path / "dascore" / "core").mkdir(parents=True)
        (api_path / "dascore" / "core.qmd").write_text("")
        (api_path / "dascore" / "io.qmd").write_text("")
        (api_path / "dascore" / "core" / "Patch.qmd").write_text("")

        out = _qmd_builder.build_compact_toc_tree(api_path)

        assert out == [
            "- text: core",
            "  href: api/dascore/core.qmd",
            "- text: io",
            "  href: api/dascore/io.qmd",
        ]

    def test_off_by_default(self, monkeypatch):
        """The exhaustive tree is what a build gets unless it asks."""
        monkeypatch.delenv("DASCORE_DOC_COMPACT_SIDEBAR", raising=False)

        assert not _qmd_builder._use_compact_sidebar()

    def test_environment_switch(self, monkeypatch, tmp_path):
        """The benchmark switches trees without a code change."""
        monkeypatch.setenv("DASCORE_DOC_COMPACT_SIDEBAR", "1")
        api_path = tmp_path / "api"
        (api_path / "dascore").mkdir(parents=True)
        (api_path / "dascore" / "core.qmd").write_text("")

        assert _qmd_builder.build_amp_toc_tree(api_path) == [
            "- text: core",
            "  href: api/dascore/core.qmd",
        ]
