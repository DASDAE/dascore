"""Tests for measuring the documentation build."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

_doc_report = pytest.importorskip("_doc_report")


@pytest.fixture
def report_path(tmp_path, monkeypatch):
    """Point the report at a temporary file."""
    path = tmp_path / "report.json"
    monkeypatch.setenv("DASCORE_DOC_REPORT", str(path))
    return path


def _page(*blocks):
    """Build a qmd page with python code blocks."""
    out = ["# name\n"]
    for block in blocks:
        out.append("```{python}\n" + textwrap.dedent(block).strip() + "\n```\n")
    return "\n".join(out)


class TestReportFile:
    """Tests for reading and writing the report."""

    def test_missing_report(self, report_path):
        """A build which hasn't measured anything yet reads as empty."""
        assert _doc_report.load_report() == {"schema": _doc_report.SCHEMA_VERSION}

    def test_sections_accumulate(self, report_path):
        """Each phase adds to the report the phase before it wrote."""
        _doc_report.update_report("timings", {"first": {"wall": 1.0}})
        _doc_report.update_report("timings", {"second": {"wall": 2.0}})
        _doc_report.update_report("site", {"total_bytes": 3})

        report = json.loads(report_path.read_text())
        assert set(report["timings"]) == {"first", "second"}
        assert report["site"]["total_bytes"] == 3
        assert report["context"]["python_version"]


class TestTimeCommand:
    """Tests for timing a command."""

    def test_records_wall_time(self, report_path):
        """The wall time of a command with no page output is recorded."""
        code = _doc_report.time_command("prep", [sys.executable, "-c", "pass"])

        timing = json.loads(report_path.read_text())["timings"]["prep"]
        assert code == 0
        assert timing["return_code"] == 0
        assert timing["wall"] > 0
        assert "pages" not in timing

    def test_failed_command(self, report_path):
        """A command which fails is recorded, and its code returned."""
        code = _doc_report.time_command(
            "prep", [sys.executable, "-c", "raise SystemExit(3)"]
        )

        assert code == 3
        assert (
            json.loads(report_path.read_text())["timings"]["prep"]["return_code"] == 3
        )

    def test_page_progress(self, report_path, capsys):
        """Quarto's progress lines become per page times."""
        script = (
            "import sys, time\n"
            "for num, name in enumerate(['a.qmd', 'b.qmd', 'c.qmd'], start=1):\n"
            "    sys.stdout.write(f'\\x1b[1m\\r[{num}/3] {name}\\x1b[0m\\n')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.05)\n"
        )
        _doc_report.time_command("quarto_render", [sys.executable, "-c", script])

        timing = json.loads(report_path.read_text())["timings"]["quarto_render"]
        assert timing["page_count"] == 3
        # The last page has no next page to bound it; it lands in finalize.
        assert set(timing["pages"]) == {"a.qmd", "b.qmd"}
        assert timing["page_phase"] == pytest.approx(0.1, abs=0.1)
        assert timing["startup"] + timing["page_phase"] + timing["finalize"] == (
            pytest.approx(timing["wall"], abs=0.01)
        )
        # The command's own output still reaches the build log.
        assert "[1/3] a.qmd" in capsys.readouterr().out


class TestClassifyPage:
    """Tests for saying what output a generated page produces."""

    def test_no_code(self):
        """A page with no code block never starts a kernel."""
        assert _doc_report.classify_page("# name\n\nsome prose\n") == "static"

    def test_source_only(self):
        """Assignments and imports produce nothing a reader sees."""
        page = _page("import dascore as dc\npatch = dc.get_example_patch()")

        assert _doc_report.classify_page(page) == "source_only"

    def test_trailing_expression(self):
        """A cell's last expression is displayed."""
        assert _doc_report.classify_page(_page("patch = 1\npatch")) == "text"

    def test_print(self):
        """A printed value is visible output wherever it appears."""
        assert _doc_report.classify_page(_page("print(1)\nx = 2")) == "text"

    def test_plot(self):
        """A drawn figure is more than text."""
        assert (
            _doc_report.classify_page(_page("patch.viz.waterfall()")) == "plot_or_file"
        )

    def test_asis(self):
        """Raw output is its own category."""
        assert _doc_report.classify_page(_page("#| output: asis\nx = 1")) == "asis"

    def test_unparsed(self):
        """Code which won't parse is flagged rather than assumed harmless."""
        assert _doc_report.classify_page(_page("this is not python(")) == "unparsed"

    def test_most_demanding_block_wins(self):
        """A page is as expensive as its most demanding block."""
        page = _page("x = 1", "x", "plt.show()")

        assert _doc_report.classify_page(page) == "plot_or_file"


class TestMeasureQmd:
    """Tests for measuring the generated qmd files."""

    def test_counts_and_classifies(self, tmp_path):
        """Executable pages are named; static ones are only counted."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "static.qmd").write_text("# static\n")
        (tmp_path / "sub" / "run.qmd").write_text(_page("print(1)"))

        out = _doc_report.measure_qmd(tmp_path)

        assert out["sizes"]["count"] == 2
        assert out["kinds"] == {"static": 1, "text": 1}
        assert len(out["executable_pages"]) == 1


class TestMeasureSidebar:
    """Tests for measuring the generated sidebar."""

    def test_api_block_only(self, tmp_path):
        """Only the generated API block is measured, not the whole config."""
        path = tmp_path / "_quarto.yml"
        path.write_text(
            "website:\n"
            "  sidebar:\n"
            "    - id: API\n"
            "      contents:\n"
            "        - text: read\n"
            "          href: api/read.qmd\n"
            "\n"
            "bibliography: references.bib\n"
        )

        out = _doc_report.measure_sidebar(path)

        assert out["entries"] == 1
        assert out["bytes"] < out["config_bytes"]
        assert "bibliography" not in path.read_text()[: out["bytes"]]


class TestCaseCollisions:
    """Tests for finding pages which differ only in case."""

    def test_no_collision(self, tmp_path):
        """Pages with distinct paths are not reported."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "spool.qmd").write_text("")
        (tmp_path / "sub" / "spool.qmd").write_text("")

        assert _doc_report.find_case_collisions(tmp_path) == []

    def test_collision_found(self, tmp_path):
        """Two pages one published site cannot hold together are reported."""
        (tmp_path / "Spool.qmd").write_text("")
        (tmp_path / "spool.qmd").write_text("")
        if len(list(tmp_path.glob("*.qmd"))) < 2:
            pytest.skip("filesystem is case insensitive")

        assert _doc_report.find_case_collisions(tmp_path) == [
            ["Spool.qmd", "spool.qmd"]
        ]


class TestMeasureSite:
    """Tests for measuring the rendered site."""

    @pytest.fixture
    def site(self, tmp_path):
        """Build a small stand-in for a rendered site."""
        path = tmp_path / "_site"
        (path / "api").mkdir(parents=True)
        (path / "site_libs").mkdir()
        (path / "index.html").write_text("<html>" + "x" * 100 + "</html>")
        (path / "api" / "read.html").write_text('<div class="cell-output">1</div>')
        (path / "api" / "old.html").write_text(
            '<meta http-equiv="refresh" content="0">'
        )
        (path / "api" / "quiet.html").write_text("<html>no output here</html>")
        (path / "site_libs" / "style.css").write_text("body {}")
        return path

    def test_missing_site(self, tmp_path, report_path):
        """A build which produced no site says so rather than failing."""
        out = _doc_report.measure_site(tmp_path / "nope")

        assert "error" in out

    def test_measures_parts_separately(self, site, report_path):
        """Content, redirects, and resources are counted apart."""
        out = _doc_report.measure_site(site)

        assert out["content_html"]["count"] == 3
        assert out["api_html"]["count"] == 2
        assert out["redirect_html"]["count"] == 1
        assert out["output_pages"] == 1
        assert out["resource_bytes"]["site_libs"] > 0
        assert out["total_bytes"] > out["content_html"]["total"]

    def test_archive(self, site, report_path):
        """The uploaded artifact is measured with its own compression."""
        out = _doc_report.measure_site(site, archive=True)

        assert 0 < out["archive_bytes"]


class TestSummary:
    """Tests for the markdown summary."""

    def test_summary_has_each_section(self, report_path):
        """Every phase which measured something appears in the summary."""
        _doc_report.update_report("timings", {"quarto_render": {"wall": 60.0}})
        _doc_report.update_report("index", {"objects": {"total": 7}})
        _doc_report.update_report("site", {"total_bytes": 1024**2})

        summary = _doc_report.build_summary(_doc_report.load_report())

        assert "1.0 min" in summary
        assert "| indexed objects | 7 |" in summary
        assert "1.00 MiB" in summary

    def test_kernel_split(self, report_path):
        """The page phase is split by the kind of output a page produces."""
        _doc_report.update_report(
            "timings",
            {"quarto_render": {"wall": 10.0, "pages": {"a.qmd": 8.0, "b.qmd": 2.0}}},
        )
        _doc_report.update_report(
            "index", {"qmd": {"executable_pages": {"a.qmd": "text"}}}
        )

        summary = _doc_report.build_summary(_doc_report.load_report())

        assert "| text | 1 |" in summary
        assert "| static | 1 |" in summary

    def test_step_summary(self, report_path, tmp_path, monkeypatch):
        """On a runner the summary is appended to the job summary."""
        step_summary = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
        _doc_report.update_report("index", {"objects": {"total": 7}})

        _doc_report.write_summary()

        assert "indexed objects" in step_summary.read_text()


class TestMain:
    """Tests for the command line."""

    def test_time_strips_separator(self, report_path):
        """The -- which ends the options is not part of the command."""
        assert (
            _doc_report.main(["time", "prep", "--", sys.executable, "-c", "pass"]) == 0
        )
        assert "prep" in json.loads(report_path.read_text())["timings"]

    def test_site(self, tmp_path, report_path):
        """The site sub command measures the path it is given."""
        assert _doc_report.main(["site", "--path", str(tmp_path / "nope")]) == 0
        assert "error" in json.loads(report_path.read_text())["site"]

    def test_summary(self, report_path, capsys):
        """The summary sub command prints the report."""
        assert _doc_report.main(["summary"]) == 0
        assert "Documentation build report" in capsys.readouterr().out


class TestClassificationCrossCheck:
    """Tests for checking the classification against what rendered."""

    @pytest.fixture
    def site(self, tmp_path, report_path):
        """A site whose pages disagree with how they were classified."""
        path = tmp_path / "_site"
        (path / "api").mkdir(parents=True)
        (path / "api" / "quiet.html").write_text('<div class="cell-output">1</div>')
        (path / "api" / "loud.html").write_text("<html>nothing rendered</html>")
        _doc_report.update_report(
            "index",
            {"qmd": {"executable_pages": {"api/loud.qmd": "text"}}},
        )
        return path

    def test_disagreements_named(self, site):
        """A page which rendered output it wasn't expected to is named."""
        out = _doc_report.measure_site(site)

        assert out["unclassified_pages"] == {"unexpected": 1, "missing": 1}
        assert out["unclassified_examples"]["unexpected"] == ["api/quiet.html"]
        assert out["unclassified_examples"]["missing"] == ["api/loud.html"]
