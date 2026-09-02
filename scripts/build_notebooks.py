"""
Render the tutorial pages to notebooks for the JupyterLite site.

The .qmd files stay the source of truth; the notebooks are build artifacts.
Each page is rendered through quarto (from inside the docs project, so the
cross-reference filter runs) and then adjusted for a browser kernel:

* links to other doc pages become absolute urls on the site being built, since
  a notebook is opened outside the site and cannot resolve relative ones,
* the kernelspec points at the Pyodide kernel rather than a local interpreter,
* a setup cell installing dascore is prepended.

Cells are rendered without executing them; the point of the notebook is that
the reader runs them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dascore.constants import DATA_VERSION
from dascore.utils.downloader import fetch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TUTORIAL_DIR = DOCS / "tutorial"
# Not configurable: docs/filters/lite_button.lua looks for the notebooks here,
# and prep_doc_build hands the same path to `jupyter lite build`.
CONTENTS_DIR = DOCS / "lite_contents"
OUT_DIR = CONTENTS_DIR / "tutorial"
# Where the mirrored example data lands inside the site. The browser kernel
# mounts the site's contents at /drive, and pooch appends DATA_VERSION, so
# DFS_DATA_DIR=/drive/data resolves to /drive/data/<DATA_VERSION>/<file>.
DATA_DIR_NAME = "data"
BROWSER_DATA_DIR = f"/drive/{DATA_DIR_NAME}"

# Links in the notebooks point back at the rendered site, which differs by
# deployment: dev publishes to netlify and releases publish to dascore.org.
# Pointing dev-built notebooks at dascore.org breaks every link to a page that
# has not been released yet. Set like DASCORE_DOC_BRANCH in the doc workflows.
# The site is served from the domain root; the `site-path: /docs` in
# _quarto.yml does not appear in the public urls.
DEFAULT_SITE_URL = "https://dascore.org"

KERNELSPEC = {
    "name": "python",
    "display_name": "Python (Pyodide)",
    "language": "python",
}

# Packages the doc examples need that are not dascore install requirements.
# pyyaml -> inventory serialization, tabulate -> DataFrame.to_markdown,
# findiff -> velocity_to_strain_rate. IPython ships with the kernel.
EXTRA_PACKAGES = ("pyyaml", "tabulate", "findiff")

# Example data shipped with the site so the tutorials read it locally instead
# of downloading it from GitHub, which rate-limits per IP and would fail a
# room full of people on one connection. Only the files the tutorial pages
# reach for; a page needing something else still downloads it as before, so
# this list going stale costs a download, not a broken notebook. To refresh
# it, run the tutorial doc-code tests with dascore.utils.downloader.fetch
# wrapped and record the names.
MIRRORED_DATA = ("terra15_das_1_trimmed.hdf5", "example_dasdae_event_1.h5")

SETUP_SOURCE = f"""\
# This notebook runs DASCore in your browser through Pyodide; nothing is
# installed on your computer. Run this cell first, it takes a moment.
%pip install -q dascore {" ".join(EXTRA_PACKAGES)}
import os

# The example data ships with this site, so DASCore reads it from here rather
# than downloading it.
os.environ["DFS_DATA_DIR"] = "{BROWSER_DATA_DIR}"
"""

SETUP_CELL_ID = "dascore-setup"

# Prefix for the transient per-page copies build() renders through quarto.
TEMP_PREFIX = "lite_tmp_"

# Executable mermaid blocks, which quarto would rasterize through chrome.
MERMAID_REGEX = re.compile(r"^```\{mermaid\}", re.MULTILINE)

# Matches a markdown link whose target is a .qmd page, with optional anchor.
LINK_REGEX = re.compile(r"(?<=]\()(?P<target>[^)\s]+\.qmd)(?P<anchor>#[^)\s]*)?(?=\))")


def get_site_url() -> str:
    """Return the base url the notebooks should link back to."""
    return os.environ.get("DASCORE_DOC_SITE_URL", DEFAULT_SITE_URL).rstrip("/")


def qmd_link_to_url(target: str, anchor: str, source: Path, site_url: str) -> str:
    """Convert a .qmd link target into an absolute url on the docs site."""
    if target.startswith("/"):
        # Already site-absolute (the cross-reference filter emits these).
        path = Path(target)
    else:
        # Relative to the directory of the page being rendered.
        resolved = (source.parent / target).resolve()
        if not resolved.is_relative_to(DOCS.resolve()):
            sys.exit(f"{source.name}: link target {target!r} escapes {DOCS}")
        path = Path("/") / resolved.relative_to(DOCS.resolve())
    return f"{site_url}{path.with_suffix('.html').as_posix()}{anchor or ''}"


def rewrite_links(cell_source: list[str], source: Path, site_url: str) -> list[str]:
    """Point every .qmd link in a markdown cell at the published site."""

    def _replace(match: re.Match) -> str:
        return qmd_link_to_url(match["target"], match["anchor"] or "", source, site_url)

    return [LINK_REGEX.sub(_replace, line) for line in cell_source]


def prepare_source(source: Path) -> Path:
    """Write a notebook-friendly copy of a page beside it, returning its path.

    Executable mermaid blocks are turned into plain fenced blocks. Quarto
    renders `{mermaid}` to an image through headless chrome, which the notebook
    build has no use for and CI has no browser for; as a fence the diagram
    survives as markdown, which JupyterLab renders natively.
    """
    text = MERMAID_REGEX.sub("```mermaid", source.read_text())
    # The copy has to stay inside the quarto project: project mode is what
    # applies the cross-reference filter, and an excluded (underscore-prefixed)
    # file renders without it. Build() sweeps any copy a crash leaves behind.
    prepared = source.parent / f"{TEMP_PREFIX}{source.name}"
    prepared.write_text(text)
    return prepared


def render_one(source: Path, out_dir: Path) -> Path | None:
    """Render a single qmd to a notebook, returning its path."""
    prepared = prepare_source(source)
    try:
        relative = prepared.relative_to(DOCS)
        # Rendering from inside the docs project makes quarto apply the project
        # filters; without them cross references stay as `dascore.X` aliases.
        result = subprocess.run(
            [
                "quarto",
                "render",
                str(relative),
                "--to",
                "ipynb",
                "--no-execute",
                "--output-dir",
                str(out_dir),
            ],
            cwd=DOCS,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            # Quarto puts the useful line in stderr; a traceback here would
            # bury it.
            sys.exit(f"quarto failed on {source.name}:\n{result.stderr.strip()}")
    finally:
        prepared.unlink(missing_ok=True)
    rendered = out_dir / relative.with_suffix(".ipynb")
    return rendered if rendered.exists() else None


def post_process(notebook_path: Path, source: Path, site_url: str) -> dict | None:
    """Adjust a rendered notebook for the browser, or None if it has no code."""
    notebook = json.loads(notebook_path.read_text())
    cells = notebook.get("cells", [])
    # Non-python executable blocks render as empty code cells; an empty cell
    # in place of content is worse than no cell at all. This has to happen
    # before the check below, or a page whose only code cells are empty
    # becomes a notebook holding nothing but the setup cell, which the docs
    # would then advertise as runnable.
    cells = [
        c for c in cells if c["cell_type"] != "code" or "".join(c["source"]).strip()
    ]
    if not any(cell["cell_type"] == "code" for cell in cells):
        return None
    for cell in cells:
        if cell["cell_type"] == "markdown":
            cell["source"] = rewrite_links(cell["source"], source, site_url)
        elif cell["cell_type"] == "code":
            # The reader executes these; shipping stale output would be worse
            # than shipping none, and it keeps the notebooks small.
            cell["outputs"] = []
            cell["execution_count"] = None
    setup = {
        "cell_type": "code",
        # nbformat 4.5 requires an id on every cell; quarto gives the cells it
        # renders one, so without this the setup cell is the only invalid cell.
        "id": SETUP_CELL_ID,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        # No trailing newline on the last line, which would render as a blank
        # line at the bottom of the first cell every reader sees.
        "source": SETUP_SOURCE.rstrip("\n").splitlines(keepends=True),
    }
    notebook["cells"] = [setup, *cells]
    notebook["metadata"]["kernelspec"] = KERNELSPEC
    return notebook


def find_unrewritten_links(notebook: dict) -> list[str]:
    """Return any .qmd link targets a notebook still carries.

    LINK_REGEX only recognises the plain `](target.qmd)` form, so a titled or
    angle-bracketed target would pass through and ship a relative link that
    404s in JupyterLite. Checking the result catches that at build time rather
    than leaving it for a reader to click.
    """
    markdown = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ]
    return re.findall(r"\]\([^)]*\.qmd[^)]*\)", "\n".join(markdown))


def mirror_data(contents_dir: Path) -> int:
    """Copy the example data the tutorials read into the site's contents.

    Returns the number of bytes mirrored. The layout has to match what pooch
    expects under DFS_DATA_DIR, which is `<dir>/<DATA_VERSION>/<name>`.
    """
    target = contents_dir / DATA_DIR_NAME / DATA_VERSION
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    total = 0
    for name in MIRRORED_DATA:
        # fetch() reads the local cache when it is primed (as it is in CI) and
        # downloads otherwise.
        source = Path(fetch(name))
        shutil.copy2(source, target / name)
        total += source.stat().st_size
    print(  # noqa: T201
        f"Mirrored {len(MIRRORED_DATA)} data files "
        f"({total / 1_000_000:.1f} MB) into {target}"
    )
    return total


def build(out_dir: Path, site_url: str) -> int:
    """Render every tutorial page with code into out_dir."""
    if shutil.which("quarto") is None:
        sys.exit("quarto is required to build the notebooks but was not found")
    # Rebuilt from scratch: a renamed page, or one that loses its last code
    # cell, would otherwise leave a notebook behind that still gets published
    # and still gets a launch link, with content the page no longer matches.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # A crashed run can leave a copy behind, and it would then be rendered
    # into the real site by the next doc build.
    for stale in TUTORIAL_DIR.glob(f"{TEMP_PREFIX}*"):
        # Quarto leaves a `<stem>_files/` sidecar beside the page, so this has
        # to remove directories too; skipping them republishes them as a
        # project resource named after an internal temp file.
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()
    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for source in sorted(TUTORIAL_DIR.glob("*.qmd")):
            rendered = render_one(source, tmp_dir)
            if rendered is None:
                print(f"  {source.name}: quarto produced no notebook, skipping")  # noqa: T201
                continue
            notebook = post_process(rendered, source, site_url)
            if notebook is None:
                print(f"  {source.name}: no code cells, skipping")  # noqa: T201
                continue
            if leftover := find_unrewritten_links(notebook):
                sys.exit(
                    f"{source.name}: these links were not rewritten and would "
                    f"break in the notebook: {leftover}"
                )
            destination = out_dir / f"{source.stem}.ipynb"
            destination.write_text(json.dumps(notebook, indent=1) + "\n")
            print(f"  {source.name} -> {destination.name}")  # noqa: T201
            written += 1
    if not written:
        sys.exit("No notebooks were built; the tutorial directory looks wrong.")
    print(f"Built {written} notebooks in {out_dir} linking to {site_url}")  # noqa: T201
    mirror_data(out_dir.parent)
    return written


def main() -> None:
    """Parse arguments and build the notebooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-url",
        default=get_site_url(),
        help="Base url the notebooks link back to (env DASCORE_DOC_SITE_URL).",
    )
    args = parser.parse_args()
    build(OUT_DIR, args.site_url.rstrip("/"))


if __name__ == "__main__":
    main()
