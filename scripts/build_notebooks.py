"""
Build the tutorial pages as marimo notebooks that run in the browser.

Render each page inside the Quarto project for cross-references, make doc links
absolute, convert the result to a marimo notebook, and export the notebooks as one
WebAssembly site. The notebooks install dascore from a bundled wheel. Notebooks are
build artifacts; qmd files remain the source.
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

from dascore.utils.downloader import fetch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "dascore" / "docs"
TUTORIAL_DIR = DOCS / "tutorial"
# The exported site. Not configurable: dascore/docs/filters/marimo_button.lua
# links to /marimo/<stem>.html when that file exists.
SITE_DIR = DOCS / "marimo"

# Use the deployment URL so dev notebooks can link to unreleased pages.
# The site is served at the domain root, without the configured /docs prefix.
DEFAULT_SITE_URL = "https://dascore.org"

# Packages the doc examples need that are not dascore install requirements.
# pyyaml -> inventory serialization, tabulate -> DataFrame.to_markdown,
# findiff -> velocity_to_strain_rate.
EXTRA_PACKAGES = ("pyyaml", "tabulate", "findiff")

# Mirror tutorial data to avoid GitHub IP rate limits in shared classrooms.
# Unlisted files still download. Refresh this list by recording fetch calls
# during tutorial doc-code tests.
MIRRORED_DATA = ("terra15_das_1_trimmed.hdf5", "example_dasdae_event_1.h5")

# marimo runs the setup cell before every other cell. In the browser it points
# pooch at the copies this site ships, so a file downloads from the site only
# when a cell first asks for it; elsewhere it does nothing.
SETUP_SOURCE = f"""\
import marimo as mo


# Download the example data from this site rather than from GitHub.
def _use_bundled_data():
    import sys

    if sys.platform != "emscripten":
        return
    from dascore.utils.downloader import get_fetcher

    fetcher = get_fetcher()
    for name in {MIRRORED_DATA!r}:
        url = mo.notebook_location() / "public" / "data" / name
        fetcher.urls[name] = str(url)


_use_bundled_data()"""

# marimo convert names the cell "setup" and hides its code from this metadata.
# Importing mo here also stops it adding a visible import cell for mo.md.
SETUP_CELL = {
    "cell_type": "code",
    "metadata": {"marimo": {"name": "setup", "config": {"hide_code": True}}},
    "source": SETUP_SOURCE,
}

# Prefix for the transient per-page copies build() renders through quarto.
TEMP_PREFIX = "marimo_tmp_"

# Executable mermaid blocks, which quarto would rasterize through chrome.
MERMAID_REGEX = re.compile(r"^```\{mermaid\}", re.MULTILINE)

# Matches a markdown link whose target is a .qmd page, with optional anchor.
LINK_REGEX = re.compile(r"(?<=]\()(?P<target>[^)\s]+\.qmd)(?P<anchor>#[^)\s]*)?(?=\))")

# Links the notebook cannot follow: qmd pages and cell attachments.
UNREWRITTEN_REGEX = re.compile(r"\]\([^)]*\.qmd[^)]*\)|attachment:[^\s\"')]+")


def get_site_url() -> str:
    """Return the base url the notebooks should link back to."""
    return os.environ.get("DASCORE_DOC_SITE_URL", DEFAULT_SITE_URL).rstrip("/")


def script_header(wheel: Path) -> str:
    """
    Return the PEP 723 header installing dascore from a local wheel.

    marimo's export copies the wheel into the site and points the header at it.
    """
    dependencies = [f"dascore @ {wheel}", *EXTRA_PACKAGES]
    lines = [f'#     "{dependency}",' for dependency in dependencies]
    return "\n".join(["# /// script", "# dependencies = [", *lines, "# ]", "# ///"])


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


def inline_attachments(cell: dict) -> list[str]:
    """
    Return a markdown cell's source with its attached images as data urls.

    Quarto stores local images as cell attachments, which marimo convert drops.
    """
    source = cell["source"]
    for name, bundle in cell.get("attachments", {}).items():
        ((mime, payload),) = bundle.items()
        url = f"data:{mime};base64,{payload}"
        source = [line.replace(f"attachment:{name}", url) for line in source]
    return source


def _run(command: list[str], name: str, **kwargs) -> None:
    """Run a build tool on name, exiting with its own output if it fails."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        output = (result.stdout + result.stderr).strip()
        sys.exit(f"{command[0]} failed on {name}:\n{output}")


def prepare_source(source: Path) -> Path:
    """
    Write a notebook-compatible page copy beside the source and return its path.

    Convert executable mermaid blocks to plain fences, avoiding Quarto's headless
    Chrome rendering.
    """
    text = MERMAID_REGEX.sub("```mermaid", source.read_text())
    # Keep the copy in the Quarto project, without an underscore prefix, so the
    # cross-reference filter runs. build() removes copies left by crashes.
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
        command = ["quarto", "render", str(relative), "--to", "ipynb"]
        command += ["--no-execute", "--output-dir", str(out_dir)]
        _run(command, source.name, cwd=DOCS)
    finally:
        prepared.unlink(missing_ok=True)
    rendered = out_dir / relative.with_suffix(".ipynb")
    return rendered if rendered.exists() else None


def post_process(
    notebook_path: Path, source: Path, site_url: str, wheel: Path
) -> dict | None:
    """Prepare a rendered notebook for marimo, or None if it has no code."""
    notebook = json.loads(notebook_path.read_text())
    cells = notebook.get("cells", [])
    # Non-Python blocks render as empty code cells, which marimo drops; a page
    # with only those would get a notebook holding just the setup cell.
    if not any(
        c["cell_type"] == "code" and "".join(c["source"]).strip() for c in cells
    ):
        return None
    for cell in cells:
        if cell["cell_type"] == "markdown":
            cell["source"] = rewrite_links(inline_attachments(cell), source, site_url)
    notebook["cells"] = [SETUP_CELL, *cells]
    notebook["metadata"]["marimo"] = {"header": script_header(wheel)}
    return notebook


def find_unrewritten_links(notebook: dict) -> list[str]:
    """
    Return qmd and attachment links left in a notebook.

    Catch titled or angle-bracketed targets missed by LINK_REGEX, and images
    without an attachment, before they become broken notebook links.
    """
    markdown = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ]
    return UNREWRITTEN_REGEX.findall("\n".join(markdown))


def mirror_data(public_dir: Path) -> None:
    """
    Copy the example data the tutorials read into the site's public files.

    The setup cell points pooch at `public/data/<name>`.
    """
    target = public_dir / "data"
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


def build(wheel: Path, site_url: str) -> None:
    """Build a marimo notebook for every tutorial page with code and export them."""
    for tool in ("quarto", "marimo", "uv"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} is required to build the notebooks but was not found")
    # Rebuild to remove stale pages and launch links after tutorials change.
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
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
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        notebook_dir = tmp_dir / "notebooks"
        notebook_dir.mkdir()
        for source in sorted(TUTORIAL_DIR.glob("*.qmd")):
            rendered = render_one(source, tmp_dir)
            if rendered is None:
                print(f"  {source.name}: quarto produced no notebook, skipping")  # noqa: T201
                continue
            notebook = post_process(rendered, source, site_url, wheel)
            if notebook is None:
                print(f"  {source.name}: no code cells, skipping")  # noqa: T201
                continue
            if leftover := find_unrewritten_links(notebook):
                sys.exit(
                    f"{source.name}: these links were not rewritten and would "
                    f"break in the notebook: {leftover}"
                )
            rendered.write_text(json.dumps(notebook))
            destination = notebook_dir / f"{source.stem}.py"
            command = ["marimo", "-q", "-y", "convert", str(rendered)]
            _run([*command, "-o", str(destination)], source.name)
            print(f"  {source.name} -> {destination.name}")  # noqa: T201
        notebooks = sorted(notebook_dir.glob("*.py"))
        if not notebooks:
            sys.exit("No notebooks were built; the tutorial directory looks wrong.")
        # Fails on notebooks marimo could not run, such as a dependency cycle.
        _run(["marimo", "check", *map(str, notebooks)], "the notebooks")
        mirror_data(notebook_dir / "public")
        # Exporting into one directory lets the notebooks share marimo's assets.
        for notebook in notebooks:
            output = SITE_DIR / f"{notebook.stem}.html"
            command = ["marimo", "-q", "-y", "export", "html-wasm", str(notebook)]
            command += ["-o", str(output), "--mode", "edit", "--no-sandbox"]
            _run(command, notebook.name)
    # The export adds an assistant prompt for editing notebooks, which agents
    # working in this checkout would otherwise pick up as instructions.
    (SITE_DIR / "CLAUDE.md").unlink(missing_ok=True)
    print(f"Exported {len(notebooks)} notebooks to {SITE_DIR} linking to {site_url}")  # noqa: T201


def main() -> None:
    """Parse arguments and build the notebooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-url",
        default=get_site_url(),
        help="Base url the notebooks link back to (env DASCORE_DOC_SITE_URL).",
    )
    args = parser.parse_args()
    # The newest wheel is the one just built from this checkout.
    wheels = (ROOT / "dist").glob("dascore-*.whl")
    wheel = max(wheels, key=lambda path: path.stat().st_mtime, default=None)
    if wheel is None:
        sys.exit("No dascore wheel in dist/; build one with `python -m build --wheel`.")
    build(wheel, args.site_url.rstrip("/"))


if __name__ == "__main__":
    main()
