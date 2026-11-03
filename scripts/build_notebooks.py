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
# The marimo notebooks, with the wheel and data they bundle beside them.
CONTENTS_DIR = DOCS / "marimo_contents" / "tutorial"
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

# marimo runs the setup cell before every other cell. In the browser it copies
# the data this site ships into a cache laid out the way pooch reads it,
# `<DFS_DATA_DIR>/<DATA_VERSION>/<name>`; elsewhere it does nothing.
SETUP_SOURCE = f"""\
# Read the example data this site ships rather than downloading it.
def _use_bundled_data():
    import os
    import sys
    import tempfile

    if sys.platform != "emscripten":
        return
    from pathlib import Path

    import marimo
    from js import XMLHttpRequest

    from dascore.constants import DATA_VERSION

    cache = Path(tempfile.gettempdir()) / "dascore-data"
    (cache / DATA_VERSION).mkdir(parents=True, exist_ok=True)
    for name in {MIRRORED_DATA!r}:
        url = marimo.notebook_location() / "public" / "data" / name
        request = XMLHttpRequest.new()
        # Synchronous, which the notebook's web worker allows.
        request.open("GET", str(url), False)
        request.responseType = "arraybuffer"
        request.send(None)
        data = request.response.to_py().tobytes()
        (cache / DATA_VERSION / name).write_bytes(data)
    os.environ["DFS_DATA_DIR"] = str(cache)


_use_bundled_data()"""

# marimo convert names the cell "setup" and hides its code from this metadata.
SETUP_CELL = {
    "cell_type": "code",
    "id": "dascore-setup",
    "metadata": {"marimo": {"name": "setup", "config": {"hide_code": True}}},
    "execution_count": None,
    "outputs": [],
    "source": SETUP_SOURCE,
}

# The markdown cells call mo.md, so the notebook needs mo defined once.
MO_CELL = {
    "cell_type": "code",
    "id": "marimo-import",
    "metadata": {"marimo": {"config": {"hide_code": True}}},
    "execution_count": None,
    "outputs": [],
    "source": "import marimo as mo",
}

# Prefix for the transient per-page copies build() renders through quarto.
TEMP_PREFIX = "marimo_tmp_"

# Executable mermaid blocks, which quarto would rasterize through chrome.
MERMAID_REGEX = re.compile(r"^```\{mermaid\}", re.MULTILINE)

# Matches a markdown link whose target is a .qmd page, with optional anchor.
LINK_REGEX = re.compile(r"(?<=]\()(?P<target>[^)\s]+\.qmd)(?P<anchor>#[^)\s]*)?(?=\))")


def get_site_url() -> str:
    """Return the base url the notebooks should link back to."""
    return os.environ.get("DASCORE_DOC_SITE_URL", DEFAULT_SITE_URL).rstrip("/")


def find_wheel(dist: Path) -> Path:
    """Return the one dascore wheel in dist, or exit naming the problem."""
    wheels = sorted(dist.glob("dascore-*.whl"))
    if len(wheels) != 1:
        sys.exit(
            f"Expected one dascore wheel in {dist}, found {len(wheels)}; "
            "build one with `python -m build --wheel` or pass --wheel."
        )
    return wheels[0]


def script_header(wheel_name: str) -> str:
    """Return the PEP 723 header installing dascore from the bundled wheel."""
    dependencies = [f"dascore @ ./wheels/{wheel_name}", *EXTRA_PACKAGES]
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


def _run(command: list[str], **kwargs) -> None:
    """Run a build tool, exiting with its own output if it fails."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        output = (result.stdout + result.stderr).strip()
        sys.exit(f"{' '.join(command[:2])} failed:\n{output}")


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
        _run([*command, "--no-execute", "--output-dir", str(out_dir)], cwd=DOCS)
    finally:
        prepared.unlink(missing_ok=True)
    rendered = out_dir / relative.with_suffix(".ipynb")
    return rendered if rendered.exists() else None


def post_process(
    notebook_path: Path, source: Path, site_url: str, wheel_name: str
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
            cell["source"] = rewrite_links(cell["source"], source, site_url)
    notebook["cells"] = [SETUP_CELL, MO_CELL, *cells]
    notebook["metadata"]["marimo"] = {"header": script_header(wheel_name)}
    return notebook


def find_unrewritten_links(notebook: dict) -> list[str]:
    """
    Return qmd links left in a notebook.

    Catch titled or angle-bracketed targets missed by LINK_REGEX before they become
    broken notebook links.
    """
    markdown = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ]
    return re.findall(r"\]\([^)]*\.qmd[^)]*\)", "\n".join(markdown))


def mirror_data(public_dir: Path) -> int:
    """Copy the example data the tutorials read into the site's public files.

    Returns the number of bytes mirrored. The setup cell fetches each file from
    `public/data/<name>`.
    """
    target = public_dir / "data"
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


def build(wheel: Path, site_url: str) -> int:
    """Build a marimo notebook for every tutorial page with code and export them."""
    for tool in ("quarto", "marimo"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} is required to build the notebooks but was not found")
    # Rebuild to remove stale notebooks and launch links after pages change.
    for directory in (CONTENTS_DIR, SITE_DIR):
        if directory.exists():
            shutil.rmtree(directory)
    CONTENTS_DIR.mkdir(parents=True)
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
        for source in sorted(TUTORIAL_DIR.glob("*.qmd")):
            rendered = render_one(source, tmp_dir)
            if rendered is None:
                print(f"  {source.name}: quarto produced no notebook, skipping")  # noqa: T201
                continue
            notebook = post_process(rendered, source, site_url, wheel.name)
            if notebook is None:
                print(f"  {source.name}: no code cells, skipping")  # noqa: T201
                continue
            if leftover := find_unrewritten_links(notebook):
                sys.exit(
                    f"{source.name}: these links were not rewritten and would "
                    f"break in the notebook: {leftover}"
                )
            rendered.write_text(json.dumps(notebook))
            destination = CONTENTS_DIR / f"{source.stem}.py"
            _run(
                ["marimo", "-q", "-y", "convert", str(rendered), "-o", str(destination)]
            )
            print(f"  {source.name} -> {destination.name}")  # noqa: T201
    notebooks = sorted(CONTENTS_DIR.glob("*.py"))
    if not notebooks:
        sys.exit("No notebooks were built; the tutorial directory looks wrong.")
    # Fails on notebooks marimo could not run, such as a dependency cycle.
    _run(["marimo", "check", *map(str, notebooks)])
    (CONTENTS_DIR / "wheels").mkdir()
    shutil.copy2(wheel, CONTENTS_DIR / "wheels" / wheel.name)
    mirror_data(CONTENTS_DIR / "public")
    # Exporting into one directory lets the notebooks share marimo's assets.
    for notebook in notebooks:
        output = SITE_DIR / f"{notebook.stem}.html"
        command = ["marimo", "-q", "-y", "export", "html-wasm", str(notebook)]
        _run([*command, "-o", str(output), "--mode", "edit", "--no-sandbox"])
    # The export adds an assistant prompt for editing notebooks, which agents
    # working in this checkout would otherwise pick up as instructions.
    (SITE_DIR / "CLAUDE.md").unlink(missing_ok=True)
    print(f"Exported {len(notebooks)} notebooks to {SITE_DIR} linking to {site_url}")  # noqa: T201
    return len(notebooks)


def main() -> None:
    """Parse arguments and build the notebooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-url",
        default=get_site_url(),
        help="Base url the notebooks link back to (env DASCORE_DOC_SITE_URL).",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="The dascore wheel the notebooks install (default: the one in dist/).",
    )
    args = parser.parse_args()
    wheel = args.wheel or find_wheel(ROOT / "dist")
    if not wheel.is_file():
        sys.exit(f"No wheel at {wheel}")
    build(wheel.resolve(), args.site_url.rstrip("/"))


if __name__ == "__main__":
    main()
