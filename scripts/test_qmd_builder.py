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
