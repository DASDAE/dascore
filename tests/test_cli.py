"""The documentation CLI serves the selected installation's public API."""

from __future__ import annotations

import inspect
import io
import json
import runpy
import subprocess
import sys
import sysconfig
from importlib.metadata import EntryPoint
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt

import dascore as dc
from dascore.cli import main
from dascore.utils import doc_cache, doc_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Prepare the real installed corpus once for lookup and rendering checks."""
    root = tmp_path_factory.mktemp("documentation")
    return root, doc_corpus.write_corpus(root)


class TestDocuments:
    """Real API aliases and authored documents share one corpus."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Patch.select", inspect.getdoc(dc.Patch.select).splitlines()[0]),
            ("Spool.chunk", "overlap"),
            ("Inventory", inspect.getdoc(dc.Inventory).splitlines()[0]),
            ("Patch.viz.waterfall", "waterfall"),
            ("dascore.core.patch.Patch.viz.waterfall", "waterfall"),
            ("PatchAttrs.tag", "A custom string field."),
            ("recipes/tunnel_inventory", "inventory"),
        ],
    )
    def test_lookup(self, corpus, name, expected):
        """Friendly names resolve to installed API and recipe content."""
        text = doc_cache.read_document(*corpus, name)
        assert expected.lower() in text.lower()
        assert dc.__version__ in text
        if name == "PatchAttrs.tag":
            assert "Type: `<class 'str'>`" in text
            assert "Defined in: `dascore.core.attrs`" in text

    @pytest.mark.parametrize(
        "name",
        ["get_unit", "utils.downloader.get_registry_df", "Patch.summary", "Patch.log"],
    )
    def test_wrapped_lookup(self, corpus, name):
        """Cached functions, cached properties, and ufuncs remain discoverable."""
        text = doc_cache.read_document(*corpus, name)
        assert "No docstring" not in text
        if name == "Patch.summary":
            assert "Property:" in text
            assert "summary(" not in text
        if name == "Patch.log":
            assert "log(patch, *args, **kwargs)" in text
            assert "wraps the NumPy ufunc documented below" in text

    def test_module_collision(self, corpus):
        """Exported functions and their same-named modules are both reachable."""
        root, manifest = corpus
        function = doc_cache.read_document(root, manifest, "dascore.viz.waterfall")
        module = doc_cache.read_document(root, manifest, "module:dascore.viz.waterfall")
        assert "id: dascore.viz.waterfall.waterfall" in function
        assert "id: module:dascore.viz.waterfall" in module
        assert function != module
        tutorial = doc_cache.read_document(root, manifest, "tutorial/visualization")
        assert "api/dascore/viz/waterfall/waterfall.md)" in tutorial

    def test_keywords(self, corpus):
        """Docstring keywords are preserved in metadata and the readable body."""
        root, manifest = corpus
        record = next(
            x
            for x in manifest["documents"]
            if x["id"] == "dascore.proc.filter.pass_filter"
        )
        assert "low pass" in record["keywords"]
        text = doc_cache.read_document(root, manifest, "Patch.pass_filter")
        assert "Keywords\n--------\nfiltering, bandpass, low pass" in text

    def test_aliases(self, corpus):
        """Export aliases resolve to one canonical document."""
        root, index = corpus
        assert index["aliases"]["BaseSpool.chunk"] == index["aliases"]["Spool.chunk"]
        assert doc_cache.read_document(
            root, index, "Patch.select"
        ) == doc_cache.read_document(root, index, "dascore.proc.coords.select")

    def test_case_distinct(self, corpus):
        """Class/factory names retain separate pages on Windows and macOS."""
        root, index = corpus
        paths = [record["path"].casefold() for record in index["documents"]]
        assert len(paths) == len(set(paths))
        assert index["aliases"]["Spool"] != index["aliases"]["spool"]
        assert "id: dascore.core.spool.Spool" in doc_cache.read_document(
            root, index, "Spool"
        )
        assert "id: dascore.core.spool.spool" in doc_cache.read_document(
            root, index, "spool"
        )

    @pytest.mark.parametrize("name", ["Inventory.from_yaml", "Spool.from_directory"])
    def test_classmethod(self, corpus, name):
        """Classmethod documentation identifies the automatically supplied class."""
        text = doc_cache.read_document(*corpus, name)
        assert "Python supplies the first class parameter." in text
        assert "the instance supplies" not in text

    def test_staticmethod(self, corpus):
        """Processor patch functions require the caller to pass the patch."""
        text = doc_cache.read_document(*corpus, "proc.basic.Abs.patch_function")
        assert "Direct and static calls require all shown parameters." in text
        assert "Python supplies the first class parameter." not in text

    def test_signature(self, corpus):
        """Signatures are inspected without evaluating string annotations."""
        signature = str(inspect.signature(dc.Patch.select, eval_str=False))
        assert signature in doc_cache.read_document(*corpus, "Patch.select")

    def test_no_operations(self, tmp_path, monkeypatch):
        """Building docs must not instantiate a Patch or invoke an operation."""

        def fail(*args, **kwargs):
            raise AssertionError("The documentation builder executed a data operation")

        monkeypatch.setattr(dc.Patch, "__init__", fail)
        monkeypatch.setattr(doc_corpus.PatchUFunc, "__get__", fail)
        index = doc_corpus.write_corpus(tmp_path)
        assert index["aliases"]["Patch.select"]

    def test_descriptors(self, tmp_path, monkeypatch):
        """Inspecting members does not evaluate arbitrary descriptors."""

        class Trap:
            def __get__(self, *args):
                raise AssertionError("descriptor evaluated")

            def __getattr__(self, name):
                raise AssertionError("descriptor attribute evaluated")

        monkeypatch.setattr(dc.Patch, "doc_trap", Trap(), raising=False)
        index = doc_corpus.write_corpus(tmp_path)
        assert "Patch.doc_trap" not in index["aliases"]

    def test_property(self, corpus):
        """Property docs describe attribute access without a call signature."""
        text = doc_cache.read_document(*corpus, "Patch.shape")
        assert "Property: access this as an attribute" in text
        assert "shape(self)" not in text

    def test_namespace_collision(self, corpus, monkeypatch):
        """Duplicate namespace providers are reported without loading their code."""
        entries = [
            EntryPoint(name="viz", value="first:Viz", group="dascore.patch_namespace"),
            EntryPoint(name="viz", value="second:Viz", group="dascore.patch_namespace"),
        ]
        monkeypatch.setattr(
            doc_cache.metadata, "entry_points", lambda **kwargs: entries
        )
        with pytest.raises(doc_corpus.DocumentationError, match="Ambiguous namespace"):
            doc_cache.read_document(*corpus, "Patch.viz.waterfall")

    def test_local_links(self, corpus):
        """Every local page and image link in the generated corpus exists."""
        root, index = corpus
        broken = []
        for record in index["documents"]:
            page = root / record["path"]
            for block in MarkdownIt().parse(page.read_text(encoding="utf-8")):
                for token in block.children or ():
                    if token.type not in {"link_open", "image"}:
                        continue
                    target = token.attrGet(
                        "href" if token.type == "link_open" else "src"
                    )
                    assert target is not None
                    url = urlsplit(target)
                    if url.scheme or url.netloc or not url.path:
                        continue
                    path = (page.parent / unquote(url.path)).resolve()
                    if not path.exists():
                        broken.append((record["id"], target))
        assert not broken

    def test_anchored_reference(self, corpus):
        """An existing anchored API reference stays local and retains its anchor."""
        text = doc_cache.read_document(*corpus, "Patch.whiten")
        assert "tutorial/processing.md#whiten)" in text

    def test_markdown(self, corpus):
        """Quarto examples remain readable source, with working page links."""
        text = doc_cache.read_document(*corpus, "tutorial/patch")
        assert "```python" in text
        assert "```{python}" not in text
        assert "(spool.md)" in text
        assert "{{< include" not in doc_cache.read_document(*corpus, "index")

    def test_missing(self, corpus):
        """Unknown names, expressions, and private traversal are never evaluated."""
        for name in ("Patch.no_such_method", "Patch.__dict__", "__import__('os')"):
            with pytest.raises(
                doc_corpus.DocumentationError, match="No local documentation"
            ):
                doc_cache.read_document(*corpus, name)

    def test_ambiguous(self, corpus):
        """An ambiguous alias explains which exact identifiers are available."""
        root, original = corpus
        index = {
            **original,
            "aliases": {
                **original["aliases"],
                "ambiguous": [
                    original["aliases"]["Patch.select"][0],
                    original["aliases"]["Spool.chunk"][0],
                ],
            },
        }
        with pytest.raises(doc_corpus.DocumentationError, match="Ambiguous"):
            doc_cache.read_document(root, index, "ambiguous")


class TestCache:
    """Cached documentation survives rebuild failures and source changes."""

    @pytest.fixture
    def cache(self, tmp_path, monkeypatch):
        """Isolate the cache with a tiny authored source and real filesystem writes."""
        source = tmp_path / "source"
        source.mkdir()
        document = source / "example.py"
        document.write_text("first")
        monkeypatch.setattr(doc_cache, "PACKAGE_PATH", source)
        monkeypatch.setattr(doc_cache, "source_paths", lambda: [])

        def build(root):
            (root / "example.md").write_text(
                document.read_text(encoding="utf-8"), encoding="utf-8"
            )
            return {
                "documents": [{"id": "example", "path": "example.md"}],
                "aliases": {"example": ["example.md"]},
                "assets": [],
                "omitted": [],
            }

        monkeypatch.setattr(doc_cache, "write_corpus", build)
        with dc.config_context(docs_cache_dir=tmp_path / "cache"):
            yield document

    def test_reuse(self, cache, monkeypatch):
        """A warm lookup reads the completed corpus without regenerating it."""
        with doc_cache.documentation_cache() as (root, index):
            stamp = (root / "manifest.json").stat().st_mtime_ns

        def fail(root):
            raise AssertionError("Warm cache rebuilt")

        monkeypatch.setattr(doc_cache, "write_corpus", fail)
        with doc_cache.documentation_cache() as (root, index):
            assert doc_cache.read_document(root, index, "example") == "first"
            assert (root / "manifest.json").stat().st_mtime_ns == stamp

    def test_editable(self, cache):
        """An editable source change invalidates a cache under the same version."""
        with doc_cache.documentation_cache() as (root, first):
            pass
        cache.write_text("second")
        with doc_cache.documentation_cache() as (root, second):
            assert first["identity"]["source"] != second["identity"]["source"]
            assert doc_cache.read_document(root, second, "example") == "second"

    def test_nested_sources(self, cache):
        """Nested API edits invalidate the corpus; generated website files do not."""
        nested = cache.parent / "core" / "operation.py"
        nested.parent.mkdir()
        nested.write_text("before")
        with doc_cache.documentation_cache() as (_, before):
            pass
        nested.write_text("after")
        with doc_cache.documentation_cache() as (_, after):
            assert before["identity"] != after["identity"]
        generated = cache.parent / "docs" / "api" / "example.py"
        generated.parent.mkdir(parents=True)
        generated.write_text("website build output")
        with doc_cache.documentation_cache() as (_, unchanged):
            assert after["identity"] == unchanged["identity"]

    def test_failed_rebuild(self, cache, monkeypatch):
        """Failure while generating a replacement preserves the complete old corpus."""
        with doc_cache.documentation_cache() as (root, original):
            pass

        def fail(staging):
            (staging / "partial.md").write_text("partial")
            raise OSError("interrupted build")

        monkeypatch.setattr(doc_cache, "write_corpus", fail)
        with pytest.raises(OSError, match="interrupted"):
            with doc_cache.documentation_cache(rebuild=True):
                pass
        assert doc_cache.read_document(root, original, "example") == "first"
        assert (
            json.loads((root / "manifest.json").read_text(encoding="utf-8")) == original
        )
        assert not (root / "partial.md").exists()

    def test_failed_publish(self, cache, monkeypatch):
        """A failed directory replacement rolls back to the preceding corpus."""
        with doc_cache.documentation_cache() as (root, original):
            pass
        replace = doc_cache.os.replace

        def fail_new(source, target):
            if Path(source).name.startswith(f".{dc.__version__}-"):
                raise OSError("replacement failed")
            return replace(source, target)

        monkeypatch.setattr(doc_cache.os, "replace", fail_new)
        with pytest.raises(OSError, match="replacement failed"):
            with doc_cache.documentation_cache(rebuild=True):
                pass
        assert doc_cache.read_document(root, original, "example") == "first"
        assert (
            json.loads((root / "manifest.json").read_text(encoding="utf-8")) == original
        )

    def test_source_change_during_build(self, cache, monkeypatch):
        """A source edit during generation cannot publish mixed documentation."""
        with doc_cache.documentation_cache() as (root, original):
            pass
        build = doc_cache.write_corpus

        def change_source(staging):
            result = build(staging)
            cache.write_text("edited during build")
            return result

        monkeypatch.setattr(doc_cache, "write_corpus", change_source)
        with pytest.raises(doc_corpus.DocumentationError, match="sources changed"):
            with doc_cache.documentation_cache(rebuild=True):
                pass
        assert doc_cache.read_document(root, original, "example") == "first"

    def test_old_backup(self, cache):
        """A backup left by an interrupted cleanup does not prevent rebuilding."""
        with doc_cache.documentation_cache() as (root, _original):
            pass
        backup = root.with_name(root.name + ".previous")
        backup.mkdir()
        (backup / "old.md").write_text("older generation")
        with doc_cache.documentation_cache(rebuild=True) as (root, current):
            assert doc_cache.read_document(root, current, "example") == "first"
        assert not backup.exists()

    def test_omitted(self, cache, monkeypatch, capsys):
        """The CLI identifies missing optional API modules in its summary."""
        build = doc_cache.write_corpus

        def omit(staging):
            result = build(staging)
            result["omitted"] = ["dascore.example: missing optional_driver"]
            return result

        monkeypatch.setattr(doc_cache, "write_corpus", omit)
        assert main(["doc"]) == 0
        assert (
            "Unavailable: dascore.example: missing optional_driver"
            in capsys.readouterr().out
        )

    def test_rebuild(self, cache):
        """Explicit rebuilding repairs an altered generated page."""
        with doc_cache.documentation_cache() as (root, index):
            (root / "example.md").write_text("altered")
        with doc_cache.documentation_cache(rebuild=True) as (root, index):
            assert doc_cache.read_document(root, index, "example") == "first"

    @pytest.mark.parametrize("damage", ["missing", "manifest", "interrupted"])
    def test_recovery(self, cache, damage):
        """Missing pages, malformed manifests, and interrupted swaps are recoverable."""
        with doc_cache.documentation_cache() as (root, index):
            pass
        if damage == "missing":
            (root / "example.md").unlink()
        elif damage == "manifest":
            (root / "manifest.json").write_text("{")
        else:
            root.rename(root.with_name(root.name + ".previous"))
        with doc_cache.documentation_cache() as (root, index):
            assert doc_cache.read_document(root, index, "example") == "first"

    def test_redirected_unicode(self, cache, monkeypatch):
        """Redirected documentation is UTF-8 even with a legacy Windows encoding."""
        cache.write_text("Supported: ✅", encoding="utf-8")
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        with monkeypatch.context() as context:
            context.setattr(sys, "stdout", stream)
            assert main(["doc", "example"]) == 0
            stream.flush()
        assert buffer.getvalue().decode("utf-8") == "Supported: ✅"

    def test_module_entry(self, cache, monkeypatch, capsys):
        """The module entry point dispatches a real document request and exits zero."""
        monkeypatch.setattr(sys, "argv", ["dascore", "doc", "example"])
        with pytest.raises(SystemExit) as result:
            runpy.run_module("dascore", run_name="__main__")
        assert result.value.code == 0
        assert capsys.readouterr().out == "first"

    def test_abandoned_staging(self, cache):
        """A later request removes staging abandoned by a killed builder."""
        with doc_cache.documentation_cache() as (root, _):
            pass
        abandoned = root.parent / f".{root.name}-abandoned"
        abandoned.mkdir()
        (abandoned / "partial.md").write_text("unfinished")
        unrelated = root.parent / ".another-version-staging"
        unrelated.mkdir()
        with doc_cache.documentation_cache() as (root, manifest):
            assert doc_cache.read_document(root, manifest, "example") == "first"
        assert not abandoned.exists()
        assert unrelated.exists()

    def test_status(self, cache, capsys):
        """CLI provenance belongs to the invoking process."""
        assert main(["doc"]) == 0
        text = capsys.readouterr().out
        assert sys.executable in text
        assert dc.__file__ in text
        assert dc.__version__ in text
        assert main(["doc", "example"]) == 0
        assert capsys.readouterr().out == "first"
        assert main(["doc", "missing"]) == 1
        captured = capsys.readouterr()
        assert not captured.out
        assert "No local documentation" in captured.err

    def test_help(self, cache):
        """Help and version succeed without building documentation."""
        for args in (["--help"], ["--version"], ["doc", "--help"]):
            assert main(args) == 0
        assert not dc.get_config().docs_cache_dir.exists()


@pytest.mark.concurrency
class TestProcesses:
    """Real command-line processes can build and reuse one shared cache."""

    def test_concurrent(self, tmp_path):
        """Simultaneous first users see complete documents through both entry points."""
        code = (
            "import dascore as dc; from dascore.cli import main; "
            f"dc.set_config(docs_cache_dir={str(tmp_path)!r}); "
            "raise SystemExit(main(['doc', 'Patch.select']))"
        )
        commands = [[sys.executable, "-c", code] for _ in range(2)]
        processes = [
            subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            for command in commands
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            assert process.returncode == 0, stderr
            assert "select(" in stdout
        root = tmp_path / dc.__version__
        assert (root / "manifest.json").is_file()
        assert not root.with_name(root.name + ".previous").exists()

    @pytest.mark.parametrize(
        "command",
        [
            [sys.executable, "-m", "dascore", "--version"],
            [
                str(
                    Path(sysconfig.get_path("scripts"))
                    / ("dascore.exe" if sys.platform == "win32" else "dascore")
                ),
                "--version",
            ],
        ],
    )
    def test_entry_points(self, command, tmp_path):
        """Both entry points work outside the repository working directory."""
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
        assert dc.__version__ in result.stdout


class TestOptionalCLI:
    """CLI commands require Typer; Python library use does not."""

    def test_missing_typer(self, hide_module, capsys):
        """A missing Typer reports the standard optional-dependency guidance."""
        hide_module("typer")
        assert main(["doc"]) == 1
        captured = capsys.readouterr()
        assert not captured.out
        assert "typer is not installed" in captured.err
        assert "pip install typer" in captured.err
        assert "uv pip install typer" in captured.err
        assert "Traceback" not in captured.err

    @pytest.mark.parametrize("args", [[], ["unknown"], ["doc", "--unknown"]])
    def test_usage_errors(self, args, capsys):
        """Missing commands and invalid arguments return a nonzero usage error."""
        assert main(args) == 2
        captured = capsys.readouterr()
        assert "Usage:" in captured.err
        assert "Error" in captured.err

    @pytest.mark.concurrency
    def test_import_is_lazy(self):
        """Importing the library and entry point does not import Typer."""
        code = (
            "import sys; import dascore; import dascore.cli; "
            "assert 'typer' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
