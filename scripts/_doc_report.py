"""
Measure what a documentation build costs, in time and in bytes.

Three timings are easy to conflate: the page phase, where quarto renders one
page after another, quarto's wall time, which also holds startup and post
processing, and the end-to-end build, which also holds the preparation the
workflow does first. This script keeps them apart, records them beside the
size of what the build produced, and writes one JSON report, so a change to
the API surface can be judged against a measured baseline.

Sub commands, in the order a build uses them:

    time <name> -- <command>   run a command and record what it cost
    index                      measure the generated qmd files and the index
    site                       measure the rendered site
    summary                    write a markdown summary of the report
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

from _api_urls import compare, current_urls, load_baseline
from _index_api import assert_documenting_this_checkout, parse_project

import dascore as dc

SCHEMA_VERSION = 1

REPO_PATH = Path(__file__).absolute().parent.parent
DOC_PATH = REPO_PATH / "docs"
API_DOC_PATH = DOC_PATH / "api"
SITE_PATH = DOC_PATH / "_site"
CROSS_REF_PATH = DOC_PATH / ".cross_ref.json"
QUARTO_CONFIG_PATH = DOC_PATH / "_quarto.yml"

# The generated block of the sidebar, and the first column zero key after it.
_SIDEBAR_MARKER = "    - id: API\n"
_NEXT_TOP_KEY = re.compile(r"\n(?=\S)")

# Quarto names a page when it starts rendering it, in a line like
# "[42/1395] api/dascore/core/patch/Patch.qmd", wrapped in color codes.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROGRESS = re.compile(r"^\[\d+/\d+\]\s+(?P<path>.+?)\s*$")

_CODE_BLOCK = re.compile(r"^```\{python\}$(?P<code>.*?)^```$", re.M | re.S)

# Enough of a marker to say a code block draws or writes something, rather
# than a complete list. The rendered site is what settles the question; see
# the output_pages count that `site` records.
_PLOT_MARKERS = ("plt.", "matplotlib", ".viz.", "savefig", "figure(", "open(")

# Most to least demanding: a page renders as the most expensive kind of
# output any one of its blocks produces.
_KIND_PRECEDENCE = ("unparsed", "asis", "plot_or_file", "text", "source_only")


def _echo(message):
    """Print a message; this script talks to a build log."""
    print(message)  # noqa: T201


# --- The report file


def report_path() -> Path:
    """Return the path the report is read from and written to."""
    default = REPO_PATH / "doc_build_report.json"
    return Path(os.environ.get("DASCORE_DOC_REPORT", default))


def load_report(path: Path | None = None) -> dict:
    """Load the report, or an empty one if the build hasn't written it yet."""
    path = report_path() if path is None else path
    if not path.exists():
        return {"schema": SCHEMA_VERSION}
    return json.loads(path.read_text())


def _run(command) -> str:
    """Return the output of a command, or an empty string if it can't run."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return out.stdout.strip()


def _get_context() -> dict:
    """Describe the machine and the commit the numbers came from."""
    return {
        "commit": _run(["git", "-C", str(REPO_PATH), "rev-parse", "HEAD"]),
        "quarto_version": _run(["quarto", "--version"]),
        "python_version": sys.version.split()[0],
        "dascore_path": str(Path(dc.__file__).absolute().parent),
        "platform": sys.platform,
        "runner": os.environ.get("RUNNER_NAME", ""),
        "compact_sidebar": os.environ.get("DASCORE_DOC_COMPACT_SIDEBAR", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
    }


def update_report(section: str, values: dict, path: Path | None = None) -> dict:
    """Merge values into one section of the report and write it back."""
    path = report_path() if path is None else path
    report = load_report(path)
    report.setdefault("context", _get_context())
    report.setdefault(section, {}).update(values)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


# --- Timing


def _page_timings(stamps: list[tuple[str, float]], wall: float) -> dict:
    """
    Split a render's wall time into startup, the page phase, and the rest.

    Quarto names a page when it starts rendering it, so a page costs the gap
    to the next name. The last page has no next name, so its cost lands in
    `finalize` along with the post processing.
    """
    if not stamps:
        return {}
    pages = {path: round(nxt - now, 3) for (path, now), (_, nxt) in pairwise(stamps)}
    return {
        "startup": round(stamps[0][1], 3),
        "page_phase": round(stamps[-1][1] - stamps[0][1], 3),
        "finalize": round(wall - stamps[-1][1], 3),
        "page_count": len(stamps),
        "pages": pages,
    }


def time_command(name: str, command: list[str], path: Path | None = None) -> int:
    """Run a command, stream its output, and record what it cost."""
    start = time.monotonic()
    stamps: list[tuple[str, float]] = []
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # newline="" keeps the carriage returns quarto draws its progress with,
    # so the command's output reaches the build log as it wrote it.
    for line in io.TextIOWrapper(process.stdout, errors="replace", newline=""):
        sys.stdout.write(line)
        sys.stdout.flush()
        if match := _PROGRESS.match(_ANSI.sub("", line).strip()):
            stamps.append((match.group("path"), time.monotonic() - start))
    return_code = process.wait()
    timing = {"wall": round(time.monotonic() - start, 3), "return_code": return_code}
    timing.update(_page_timings(stamps, timing["wall"]))
    update_report("timings", {name: timing}, path=path)
    _echo(f"{name} took {timing['wall']:.1f}s")
    return return_code


# --- Size helpers


def _percentile(sizes, fraction: float) -> int:
    """Return the value a fraction of the way through the sorted sizes."""
    if not sizes:
        return 0
    ordered = sorted(sizes)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def _size_stats(sizes) -> dict:
    """Summarize a collection of file sizes."""
    sizes = list(sizes)
    return {
        "count": len(sizes),
        "total": sum(sizes),
        "p50": _percentile(sizes, 0.5),
        "p95": _percentile(sizes, 0.95),
    }


# --- The generated API docs


def _classify_block(code: str) -> str:
    """Say what kind of output one code block produces."""
    if "output: asis" in code:
        return "asis"
    if any(marker in code for marker in _PLOT_MARKERS):
        return "plot_or_file"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "unparsed"
    calls = (x.func for x in ast.walk(tree) if isinstance(x, ast.Call))
    if any(getattr(func, "id", "") == "print" for func in calls):
        return "text"
    # Only a cell's last expression is displayed.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        return "text"
    return "source_only"


def classify_page(text: str) -> str:
    """Say what kind of output a rendered page produces, if any."""
    kinds = {_classify_block(x.group("code")) for x in _CODE_BLOCK.finditer(text)}
    if not kinds:
        return "static"
    return next(x for x in _KIND_PRECEDENCE if x in kinds)


def measure_qmd(api_path: Path = API_DOC_PATH) -> dict:
    """Measure the generated qmd files and classify the executable ones."""
    sizes, kinds, executable = [], Counter(), {}
    for path in sorted(api_path.rglob("*.qmd")):
        sizes.append(path.stat().st_size)
        kind = classify_page(path.read_text())
        kinds[kind] += 1
        if kind != "static":
            executable[str(path.relative_to(api_path.parent))] = kind
    return {
        "sizes": _size_stats(sizes),
        "kinds": dict(kinds),
        "executable_pages": executable,
    }


def measure_objects() -> dict:
    """Count the objects the API index holds, by kind and by package."""
    assert_documenting_this_checkout(dc)
    data = parse_project(dc).values()
    keys = [x["key"] for x in data]
    packages = Counter(".".join(x.split(".")[:2]) for x in keys)
    private = [x for x in keys if any(y.startswith("_") for y in x.split("."))]
    return {
        "total": len(keys),
        "kinds": dict(Counter(x["data_type"] for x in data)),
        "packages": dict(sorted(packages.items())),
        "private_segment_keys": len(private),
    }


def measure_sidebar(config_path: Path = QUARTO_CONFIG_PATH) -> dict:
    """Measure the generated API block of the quarto config."""
    text = config_path.read_text()
    rest = text[text.index(_SIDEBAR_MARKER) :]
    if match := _NEXT_TOP_KEY.search(rest):
        rest = rest[: match.start() + 1]
    return {
        "bytes": len(rest.encode()),
        "entries": len(re.findall(r"^\s*- (?:text|section):", rest, re.M)),
        "config_bytes": len(text.encode()),
    }


def measure_cross_ref(cross_ref_path: Path = CROSS_REF_PATH) -> dict:
    """Count the symbolic keys the cross reference resolves."""
    mapping = json.loads(cross_ref_path.read_text())
    api = {k: v for k, v in mapping.items() if v.startswith("/api/")}
    return {
        "keys": len(mapping),
        "api_keys": len(api),
        "api_targets": len(set(api.values())),
    }


def find_case_collisions(api_path: Path = API_DOC_PATH) -> list[list[str]]:
    """
    Return groups of API pages whose paths differ only in case.

    A case insensitive filesystem, and the published site, can hold only one
    of each group, so a redirect cannot be promised for both.
    """
    groups = defaultdict(list)
    for path in api_path.rglob("*.qmd"):
        relative = str(path.relative_to(api_path))
        groups[relative.lower()].append(relative)
    return sorted(sorted(x) for x in groups.values() if len(x) > 1)


def measure_urls() -> dict:
    """Compare the URLs this build publishes against the frozen baseline."""
    baseline = load_baseline()
    if not baseline:
        return {"baseline": 0}
    difference = compare(current_urls(), baseline)
    out = {"baseline": len(baseline)}
    out.update({k: len(v) for k, v in difference.items()})
    out["removed_examples"] = difference["removed"][:20]
    out["moved_examples"] = dict(list(difference["moved"].items())[:20])
    return out


def measure_index(path: Path | None = None) -> dict:
    """Measure everything the pre-render phase produced."""
    values = {
        "objects": measure_objects(),
        "qmd": measure_qmd(),
        "sidebar": measure_sidebar(),
        "cross_ref": measure_cross_ref(),
        "case_collisions": find_case_collisions(),
        "urls": measure_urls(),
    }
    update_report("index", values, path=path)
    return values


# --- The rendered site


def _is_redirect(text: str) -> bool:
    """Return True for a page whose only job is to point at another page."""
    return 'http-equiv="refresh"' in text[:2000]


# A page which produces no output for a reader is one which could render
# without a kernel; one which produces output the classification missed is a
# page an aggregation would silently change. Both are worth naming.
_QUIET_KINDS = frozenset({"static", "source_only"})


def _record_surprise(surprises, relative, executable, has_output) -> None:
    """Note a page whose rendered output disagrees with its classification."""
    kind = executable.get(str(relative.with_suffix(".qmd")), "static")
    if has_output and kind in _QUIET_KINDS:
        surprises["unexpected"].append(str(relative))
    elif not has_output and kind not in _QUIET_KINDS:
        surprises["missing"].append(str(relative))


def _archive_bytes(site_path: Path) -> int:
    """Return the size of the site compressed the way an artifact upload is."""
    with TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "site.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in site_path.rglob("*"):
                if path.is_file():
                    zipped.write(path, path.relative_to(site_path))
        return archive.stat().st_size


def measure_site(
    site_path: Path = SITE_PATH,
    archive: bool = False,
    path: Path | None = None,
) -> dict:
    """Measure the rendered site, separating content from everything else."""
    if not site_path.exists():
        values = {"error": f"{site_path} does not exist"}
        update_report("site", values, path=path)
        _echo(f"No site to measure at {site_path}")
        return values
    classified = load_report(path).get("index", {}).get("qmd", {})
    executable = classified.get("executable_pages", {})
    content, api, redirects = [], [], []
    resources: Counter = Counter()
    surprises: dict[str, list[str]] = {"unexpected": [], "missing": []}
    total, output_pages = 0, 0
    for file_path in site_path.rglob("*"):
        if not file_path.is_file():
            continue
        size = file_path.stat().st_size
        total += size
        relative = file_path.relative_to(site_path)
        if file_path.suffix != ".html":
            resources[relative.parts[0] if len(relative.parts) > 1 else "root"] += size
            continue
        text = file_path.read_text(errors="replace")
        if _is_redirect(text):
            redirects.append(size)
            continue
        content.append(size)
        if relative.parts[0] == "api":
            api.append(size)
            has_output = "cell-output" in text
            output_pages += has_output
            _record_surprise(surprises, relative, executable, has_output)
    values = {
        "total_bytes": total,
        "content_html": _size_stats(content),
        "api_html": _size_stats(api),
        "redirect_html": _size_stats(redirects),
        "resource_bytes": dict(sorted(resources.items())),
        "output_pages": output_pages,
        "unclassified_pages": {k: len(v) for k, v in surprises.items()},
        "unclassified_examples": {k: sorted(v)[:20] for k, v in surprises.items()},
    }
    if archive:
        values["archive_bytes"] = _archive_bytes(site_path)
    update_report("site", values, path=path)
    return values


# --- The summary


def _mib(value) -> str:
    """Format a byte count in MiB."""
    return f"{value / 1024**2:.2f} MiB"


def _minutes(value) -> str:
    """Format a duration in minutes."""
    return f"{value / 60:.1f} min"


def _kernel_lines(report: dict) -> list[str]:
    """Report the page phase split by whether a page ran a kernel."""
    pages = report.get("timings", {}).get("quarto_render", {}).get("pages", {})
    kinds = report.get("index", {}).get("qmd", {}).get("executable_pages", {})
    if not pages:
        return []
    grouped = defaultdict(list)
    for page, seconds in pages.items():
        grouped[kinds.get(page, "static")].append(seconds)
    out = ["", "| Page kind | Pages | Mean | Total |", "|---|---:|---:|---:|"]
    for kind, times in sorted(grouped.items()):
        mean = sum(times) / len(times)
        out.append(f"| {kind} | {len(times)} | {mean:.2f} s | {_minutes(sum(times))} |")
    return out


def build_summary(report: dict) -> str:
    """Build a markdown summary of the report."""
    timings = report.get("timings", {})
    index = report.get("index", {})
    site = report.get("site", {})
    lines = ["## Documentation build report", ""]
    if timings:
        lines += ["| Phase | Time |", "|---|---:|"]
        total = 0.0
        for name, timing in sorted(timings.items()):
            total += timing.get("wall", 0)
            lines.append(f"| {name} | {_minutes(timing.get('wall', 0))} |")
            if "page_phase" in timing:
                lines.append(f"| {name} (pages) | {_minutes(timing['page_phase'])} |")
        lines.append(f"| **end to end** | **{_minutes(total)}** |")
    lines += _kernel_lines(report)
    if index:
        objects, qmd = index.get("objects", {}), index.get("qmd", {})
        sidebar_kib = index.get("sidebar", {}).get("bytes", 0) / 1024
        urls = index.get("urls", {})
        lines += [
            "",
            "| Content | Value |",
            "|---|---:|",
            f"| indexed objects | {objects.get('total', 0)} |",
            f"| generated pages | {qmd.get('sizes', {}).get('count', 0)} |",
            f"| API sidebar | {sidebar_kib:.1f} KiB |",
            f"| cross reference keys | {index.get('cross_ref', {}).get('keys', 0)} |",
            f"| case collisions | {len(index.get('case_collisions', []))} |",
            f"| URLs removed since baseline | {urls.get('removed', 0)} |",
            f"| URLs moved since baseline | {urls.get('moved', 0)} |",
        ]
        for kind, count in sorted(qmd.get("kinds", {}).items()):
            lines.append(f"| pages, {kind} | {count} |")
    if site and "error" not in site:
        content = site.get("content_html", {})
        unclassified = site.get("unclassified_pages", {})
        lines += [
            "",
            "| Site | Value |",
            "|---|---:|",
            f"| rendered site | {_mib(site.get('total_bytes', 0))} |",
            f"| API html | {_mib(site.get('api_html', {}).get('total', 0))} |",
            f"| content page p50 | {content.get('p50', 0) / 1024:.1f} KiB |",
            f"| content page p95 | {content.get('p95', 0) / 1024:.1f} KiB |",
            f"| redirect pages | {site.get('redirect_html', {}).get('count', 0)} |",
            f"| pages with output | {site.get('output_pages', 0)} |",
            f"| classification misses | {sum(unclassified.values())} |",
        ]
        if "archive_bytes" in site:
            lines.append(f"| uploaded artifact | {_mib(site['archive_bytes'])} |")
    return "\n".join(lines) + "\n"


def write_summary(path: Path | None = None) -> str:
    """Write the markdown summary to the build log and the step summary."""
    summary = build_summary(load_report(path))
    _echo(summary)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as fi:
            fi.write(summary)
    return summary


# --- Command line


def _get_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    timer = sub.add_parser("time", help="run a command and record what it cost")
    timer.add_argument("name", help="the phase name to record the time under")
    timer.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the command")
    sub.add_parser("index", help="measure the generated qmd files and the index")
    site = sub.add_parser("site", help="measure the rendered site")
    site.add_argument("--path", type=Path, default=SITE_PATH)
    site.add_argument("--archive", action="store_true", help="compress the site too")
    sub.add_parser("summary", help="write a markdown summary of the report")
    return parser


def main(args=None) -> int:
    """Run one sub command."""
    parsed = _get_parser().parse_args(args)
    if parsed.command == "time":
        command = parsed.rest[1:] if parsed.rest[:1] == ["--"] else parsed.rest
        return time_command(parsed.name, command)
    if parsed.command == "index":
        measure_index()
    elif parsed.command == "site":
        measure_site(parsed.path, archive=parsed.archive)
    else:
        write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
