"""A script to build quartos main config file."""

from __future__ import annotations

import os
from pathlib import Path

from _render_api import get_template

import dascore as dc

API_PATH = Path(__file__).absolute().parent.parent / "docs" / "api"


def _build_content_string(path, api_path):
    """Build a content string."""
    out = [
        f"- text: {path.with_suffix('').name}",
        f"  href: {path.relative_to(api_path)}",
    ]
    return out


def build_api_toc_tree(api_path=API_PATH):
    """
    Build the API toc tree: one entry per top level section.

    Naming every documented object put 165 KiB in the generated config and
    repeated it into all 1,388 rendered pages, which cost more than anything
    else the build did: against a section level tree, quarto's page phase
    fell from 105 to 34 minutes and the API html from 635 to 42 MiB. Readers
    reach an object from its owner's page, which lists what it owns.
    """
    base_path = api_path.parent
    out = []
    for path in sorted((api_path / "dascore").glob("*.qmd")):
        out.extend(_build_content_string(path, base_path))
    return out


def _get_dascore_title():
    """Get the DASCore title with the docs version."""
    doc_version = os.environ.get("DASCORE_DOC_VERSION")
    if doc_version is not None:
        vstr = doc_version
    else:
        version_str = str(dc.__version__)
        if "dev" not in version_str:
            vstr = version_str
        else:
            vstr = version_str.split("+")[0]
    return f"DASCore ({vstr})"


def _get_repo_branch():
    """Get the branch the docs' "edit this page" links should point to."""
    return os.environ.get("DASCORE_DOC_BRANCH", "master")


def create_quarto_qmd():
    """Create the _quarto.yml file."""
    temp = get_template("_quarto.yml")
    version_str = _get_dascore_title()
    api_toc_tree = build_api_toc_tree()
    out = temp.render(
        dascore_version_str=version_str,
        api_toc_tree=api_toc_tree,
        repo_branch=_get_repo_branch(),
    )
    path = Path(__file__).parent.parent / "docs" / "_quarto.yml"
    with path.open("w") as fi:
        fi.write(out)


if __name__ == "__main__":
    create_quarto_qmd()
