"""Tests for generating qmd-backed pytest modules."""

from __future__ import annotations

from pathlib import Path

from generate_doc_code_tests import (
    DOCS_PATH,
    REPO_ROOT,
    TEXT_ENCODING,
    Chunk,
    QmdFile,
    extract_qmd_file,
    get_output_path,
    render_test_module,
    write_test_tree,
)


def _write(path: Path, text: str) -> Path:
    # Tests create tiny synthetic qmd files instead of touching real docs.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=TEXT_ENCODING)
    return path


class TestExtractQmdFile:
    """Tests for extracting executable qmd chunks."""

    def test_skips_plain_python_fences(self, tmp_path):
        """Only executable quarto python fences should be included."""
        # `{python}` should execute, while `{.python}` is just a syntax fence.
        path = _write(
            tmp_path / "example.qmd",
            """---
title: Example
---

```{python}
import dascore as dc
```

```{.python}
print("not executed")
```
""",
        )
        out = extract_qmd_file(path)
        assert out.chunks == (Chunk(start_line=5, source="import dascore as dc\n"),)

    def test_skips_doc_with_eval_false(self, tmp_path):
        """Document-level eval false should skip all chunks."""
        # Front-matter execution settings should disable every chunk below.
        path = _write(
            tmp_path / "example.qmd",
            """---
execute:
  eval: false
---

```{python}
print("nope")
```
""",
        )
        assert extract_qmd_file(path).chunks == ()

    def test_skips_doc_with_inline_execute_false(self, tmp_path):
        """Inline document-level execute false should skip all chunks."""
        path = _write(
            tmp_path / "example.qmd",
            """---
execute: false
---

```{python}
print("nope")
```
""",
        )
        assert extract_qmd_file(path).chunks == ()

    def test_extracts_python_fence_with_space_delimited_options(self, tmp_path):
        """Quarto python fences can include filename options after a space."""
        # Quarto allows options after a space, not just after commas.
        path = _write(
            tmp_path / "example.qmd",
            """```{python filename="example.py"}
print("run")
```
""",
        )
        out = extract_qmd_file(path)
        assert out.chunks == (Chunk(start_line=2, source='print("run")\n'),)

    def test_skips_python_fence_with_inline_eval_false(self, tmp_path):
        """Inline fence options should be able to disable execution."""
        path = _write(
            tmp_path / "example.qmd",
            """```{python, eval=false}
print("skip")
```

```{python}
print("run")
```
""",
        )
        out = extract_qmd_file(path)
        assert out.chunks == (Chunk(start_line=6, source='print("run")\n'),)

    def test_skips_python_fence_with_inline_execute_zero(self, tmp_path):
        """Space-delimited inline execute flags should also be respected."""
        path = _write(
            tmp_path / "example.qmd",
            """```{python execute=0}
print("skip")
```

```{python}
print("run")
```
""",
        )
        out = extract_qmd_file(path)
        assert out.chunks == (Chunk(start_line=6, source='print("run")\n'),)

    def test_reads_qmd_with_utf8_encoding(self, tmp_path, monkeypatch):
        """Extraction should force UTF-8 instead of platform default encodings."""
        path = _write(tmp_path / "example.qmd", "placeholder")
        called: dict[str, str | None] = {}
        original = Path.read_text

        def _read_text(self, *args, **kwargs):
            # Intercept the read call so we can assert the requested encoding.
            if self == path:
                called["encoding"] = kwargs.get("encoding")
                return """```{python}
print("run")
```"""
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _read_text)

        out = extract_qmd_file(path)

        assert called["encoding"] == TEXT_ENCODING
        assert out.chunks == (Chunk(start_line=2, source='print("run")\n'),)

    def test_dedents_indented_python_fence(self, tmp_path):
        """Indented fenced code should be normalized before storage."""
        path = _write(
            tmp_path / "example.qmd",
            """```{python}
    if True:
        print("run")
```
""",
        )
        out = extract_qmd_file(path)
        assert out.chunks == (
            Chunk(start_line=2, source='if True:\n    print("run")\n'),
        )

    def test_skips_chunk_with_eval_or_execute_false(self, tmp_path):
        """Chunk-level execution flags should be honored."""
        # Chunk-local options should override the document default.
        path = _write(
            tmp_path / "example.qmd",
            """```{python}
#| eval: false
print("skip")
```

```{python}
#| execute: false
print("skip")
```

```{python}
print("run")
```
""",
        )
        out = extract_qmd_file(path)
        assert out.chunks == (Chunk(start_line=12, source='print("run")\n'),)


class TestOutputPaths:
    """Tests for source-to-test path mapping."""

    def test_root_doc_maps_to_root_test(self):
        """Top-level docs should stay at the top of generated tests."""
        # docs/index.qmd becomes tests/test_autogenerated_doccode/test_index.py
        source = DOCS_PATH / "index.qmd"
        tests = Path("/repo/tests/test_autogenerated_doccode")
        assert get_output_path(source, tests_path=tests) == tests / "test_index.py"

    def test_nested_doc_maps_to_nested_test(self):
        """Nested docs should keep their relative directory structure."""
        # Nested docs should preserve their package-like layout.
        source = DOCS_PATH / "tutorial" / "file_io.qmd"
        tests = Path("/repo/tests/test_autogenerated_doccode")
        expected = tests / "tutorial" / "test_file_io.py"
        assert get_output_path(source, tests_path=tests) == expected

    def test_custom_docs_root_can_be_supplied(self):
        """Mapping should support callers with a non-default docs root."""
        docs_root = Path("/repo/custom_docs")
        source = docs_root / "guide" / "example.qmd"
        tests = Path("/repo/tests/test_autogenerated_doccode")
        expected = tests / "guide" / "test_example.py"
        assert (
            get_output_path(source, tests_path=tests, docs_path=docs_root) == expected
        )


class TestRenderAndWrite:
    """Tests for generated module output."""

    def test_render_includes_source_and_chunk_payload(self):
        """Generated modules should inline source code with source comments."""
        # The generated file should contain enough literal data to run by itself.
        source = REPO_ROOT / "docs" / "tutorial" / "example.qmd"
        module = render_test_module(source, (Chunk(start_line=12, source="x = 1\n"),))
        assert "Autogenerated from docs/tutorial/example.qmd" in module
        assert "@pytest.mark.docs_examples" in module
        assert "def test_main()" in module
        assert "source_qmd = 'docs/tutorial/example.qmd'" in module
        assert "with qmd_test_context(source_qmd):" in module
        assert "### docs/tutorial/example.qmd:12" in module
        assert "x = 1" in module
        assert "CHUNKS =" not in module
        assert "SOURCE_QMD =" not in module
        assert "_runtime" not in module

    def test_render_hoists_future_imports(self):
        """Future imports should move to module scope."""
        source = REPO_ROOT / "docs" / "tutorial" / "example.qmd"
        module = render_test_module(
            source,
            (
                Chunk(
                    start_line=12,
                    source="from __future__ import annotations\nx = 1\n",
                ),
            ),
        )
        assert "from __future__ import annotations\n\nimport pytest" in module
        assert "# docs/tutorial/example.qmd:12" in module
        assert "        x = 1" in module
        assert "        from __future__ import annotations" not in module

    def test_write_tree_removes_stale_files(self, tmp_path):
        """Regeneration should replace the whole generated test tree."""
        # Full regeneration should remove old outputs before writing new ones.
        tests_path = tmp_path / "tests" / "test_autogenerated_doccode"
        stale = tests_path / "obsolete.py"
        stale.parent.mkdir(parents=True)
        stale.write_text("old")

        source = DOCS_PATH / "tutorial" / "file_io.qmd"
        chunks = (Chunk(start_line=1, source="print(1)\n"),)
        qmd_file = QmdFile(path=source, chunks=chunks)

        written = write_test_tree([qmd_file], tests_path=tests_path)
        output = tests_path / "tutorial" / "test_file_io.py"

        assert output in written
        assert output.exists()
        assert not stale.exists()
        assert (tests_path / "conftest.py").exists()

    def test_write_tree_skips_docs_without_executable_chunks(self, tmp_path):
        """Docs with no executable chunks should not emit per-doc test modules."""
        tests_path = tmp_path / "tests" / "test_autogenerated_doccode"
        source = DOCS_PATH / "tutorial" / "no_code.qmd"
        qmd_file = QmdFile(path=source, chunks=())

        written = write_test_tree([qmd_file], tests_path=tests_path)
        output = tests_path / "tutorial" / "test_no_code.py"

        assert (tests_path / "__init__.py") in written
        assert (tests_path / "conftest.py") in written
        assert output not in written
        assert not output.exists()
