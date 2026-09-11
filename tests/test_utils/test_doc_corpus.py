"""Boundary checks for installed documentation sources and API inspection."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from types import ModuleType, SimpleNamespace

import pytest

from dascore.utils import doc_corpus


class TestAuthoredDocuments:
    """Packaged prose becomes readable Markdown without executing examples."""

    @pytest.fixture
    def sources(self, tmp_path, monkeypatch):
        """Supply a small authored corpus without importing the full API."""
        source = tmp_path / "docs"
        source.mkdir()
        monkeypatch.setattr(doc_corpus, "DOC_PATH", source)
        monkeypatch.setattr(doc_corpus, "_api_documents", lambda: ({}, []))
        return source, tmp_path / "output"

    def test_include_and_keywords(self, sources):
        """Includes, scalar keywords, links, and unevaluated code survive the build."""
        source, output = sources
        (source / "fragment.md").write_text("Included instructions.")
        (source / "page.qmd").write_text(
            "---\ntitle: A procedure\nkeywords: filtering\n---\n"
            "{{< include fragment.md >}}\n"
            "[relative](fragment.qmd?mode=one#details)\n"
            "[absolute](/fragment.qmd)\n"
            "[unavailable](missing.qmd)\n"
            "[API](`dascore.missing.Plugin`)\n"
            "```{python}\nraise AssertionError('must not run')\n```\n"
        )
        for directory in ("api", "_site", ".quarto", "page_files"):
            path = source / directory / "stale.qmd"
            path.parent.mkdir()
            path.write_text("Do not index build output.")
        index = doc_corpus.write_corpus(output)
        assert {item["id"] for item in index["documents"]} == {"fragment", "page"}
        record = next(item for item in index["documents"] if item["id"] == "page")
        assert record["keywords"] == ["filtering"]
        text = (output / record["path"]).read_text()
        assert "Included instructions." in text
        assert "(fragment.md?mode=one#details)" in text
        assert "(fragment.md)" in text
        assert "https://dascore.org/missing.html" in text
        assert "https://dascore.org/api/dascore.html" in text
        assert "```python\nraise AssertionError" in text

    @pytest.mark.parametrize(
        "body, message",
        [
            ("---\ntitle: no closing delimiter", "Unterminated"),
            ("{{< include ../outside.md >}}", "escapes"),
            ("{{< include page.qmd >}}", "Circular"),
        ],
    )
    def test_invalid_source(self, sources, body, message):
        """Broken authored sources fail with specific errors."""
        source, output = sources
        (source / "page.qmd").write_text(body)
        with pytest.raises(doc_corpus.DocumentationError, match=message):
            doc_corpus.write_corpus(output)

    def test_readme(self, tmp_path, monkeypatch):
        """Installed metadata wins over an unrelated sibling README."""
        package = tmp_path / "dascore"
        package.mkdir()
        readme = tmp_path / "readme.md"
        readme.write_text("Unrelated package README")
        monkeypatch.setattr(doc_corpus, "PACKAGE_PATH", package)
        distribution = SimpleNamespace(
            read_text=lambda name: "Name: dascore\n\nPackaged README"
        )
        monkeypatch.setattr(
            doc_corpus.metadata, "distribution", lambda name: distribution
        )
        assert doc_corpus._read_readme() == "Packaged README"
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "dascore"\n')
        readme.write_text("Editable source README")
        assert doc_corpus._read_readme() == "Editable source README"


class TestApiInspection:
    """Only supported owned objects and modules are inspected."""

    @pytest.fixture
    def isolated_api(self, monkeypatch):
        """Use controlled modules and avoid loading file-format entry points."""
        monkeypatch.setattr(doc_corpus, "_API_PACKAGES", ())
        monkeypatch.setattr(doc_corpus, "_API_MODULES", ("dascore.doc_probe",))
        distribution = SimpleNamespace(
            entry_points=[
                EntryPoint(
                    name="unused",
                    value="not_installed:Format",
                    group="dascore.fiber_io",
                )
            ]
        )
        monkeypatch.setattr(
            doc_corpus.metadata, "distribution", lambda name: distribution
        )

    def test_uninspectable(self, isolated_api, monkeypatch):
        """Missing signatures and wrapped callable objects do not crash inspection."""
        module = ModuleType("dascore.doc_probe")

        class NoSignature:
            __signature__ = "not a signature"

        class Callable:
            def __call__(self):
                raise AssertionError("must not run")

        NoSignature.__module__ = Callable.__module__ = module.__name__
        NoSignature.__qualname__ = "NoSignature"

        def wrapped():
            raise AssertionError("must not run")

        wrapped.__wrapped__ = Callable()
        module.NoSignature = NoSignature
        module.wrapped = wrapped
        monkeypatch.setattr(doc_corpus, "import_module", lambda name: module)
        records, omitted = doc_corpus._api_documents()
        assert "dascore.doc_probe.NoSignature" in records
        assert "signature" not in records["dascore.doc_probe.NoSignature"]["body"]
        assert "dascore.doc_probe.wrapped" not in records
        assert not omitted

    @pytest.mark.parametrize("missing", ["optional_driver", "dascore.internal"])
    def test_missing_module(self, isolated_api, monkeypatch, missing):
        """Missing optional dependencies are reported; broken package imports fail."""

        def unavailable(name):
            raise ModuleNotFoundError(name=missing)

        monkeypatch.setattr(doc_corpus, "import_module", unavailable)
        if missing.startswith("dascore"):
            with pytest.raises(ModuleNotFoundError):
                doc_corpus._api_documents()
        else:
            records, omitted = doc_corpus._api_documents()
            assert not records
            assert omitted == ["dascore.doc_probe: missing optional_driver"]

    def test_private_modules(self, isolated_api, tmp_path, monkeypatch):
        """Private package subtrees and helper modules are excluded before import."""
        for name in ("core/_private/helper.py", "core/_helper.py", "core/public.py"):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("raise AssertionError('must not run')")
        monkeypatch.setattr(doc_corpus, "PACKAGE_PATH", tmp_path)
        monkeypatch.setattr(doc_corpus, "_API_MODULES", ())
        monkeypatch.setattr(doc_corpus, "_API_PACKAGES", ("core",))
        imported = []

        def import_owned(name):
            imported.append(name)
            return ModuleType(name)

        monkeypatch.setattr(doc_corpus, "import_module", import_owned)
        records, _ = doc_corpus._api_documents()
        assert imported == ["dascore.core.public"]
        assert list(records) == [f"module:{name}" for name in imported]
