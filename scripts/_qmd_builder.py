"""
A script to build quartos main config file.
"""
from __future__ import annotations
import os
from pathlib import Path

from _render_api import get_template

import dascore as dc

API_PATH = Path(__file__).absolute().parent.parent / "docs" / "api"

# separation for each level of toc tree
LEVEL_SEP = "    "


def _build_content_string(path, api_path):
    """Build a content string."""
    out = [
        f"- text: {path.with_suffix('').name}",
        f"  href: {path.relative_to(api_path)}",
    ]
    return out


def _build_section_string(path, api_path):
    """Build a string for entire sections."""
    out = [
        f"- section: {path.with_suffix('').name}",
        f"  href: {path.relative_to(api_path)}",
        "  contents:",
    ]
    return out


def _get_level(path, base_path):
    """Get the level of directory nested for path from base_path."""
    level = len(str(path.relative_to(base_path)).split(os.sep)) - 1
    return level


def build_amp_toc_tree(api_path=API_PATH):
    """Build the toc tree for the API."""
    # get all sub directories
    base_path = api_path.parent
    sub_dirs = sorted(x for x in api_path.rglob("*") if x.is_dir())
    sub_dir_set = set(sub_dirs)
    out = []
    # iterate and make contents
    for dir_path in sub_dirs:
        # this is an un-cleaned up quarto dir from rendering
        bad_endings = ["execute-results", "_files"]
        bad_ending = any(dir_path.name.endswith(x) for x in bad_endings)
        # the expected path of the qmd file related to this directory.
        section_qmd = dir_path.with_suffix(".qmd")
        if bad_ending and not section_qmd.exists():
            continue
        assert section_qmd.exists()
        level = _get_level(section_qmd, api_path)
        # see how deep we are in toc tree for determine spaces needed
        section_list = _build_section_string(section_qmd, base_path)
        for val in section_list:
            out.append(LEVEL_SEP * (level) + val)
        # now go through contents
        contents = sorted(
            x for x in dir_path.glob("*.qmd") if x.with_suffix("") not in sub_dir_set
        )
        for content in contents:
            content_list = _build_content_string(content, base_path)
            for val in content_list:
                out.append(LEVEL_SEP * (level + 1) + val)
    return out


def create_quarto_qmd():
    """Create the _quarto.yml file."""

    def _get_nice_version_string():
        """Just get a simplified version string."""
        version_str = str(dc.__version__)
        if "dev" not in version_str:
            vstr = version_str
        else:
            vstr = version_str.split("+")[0]
        return f"DASCore ({vstr})"

    temp = get_template("_quarto.yml")
    version_str = _get_nice_version_string()
    api_toc_tree = build_amp_toc_tree()
    out = temp.render(dascore_version_str=version_str, api_toc_tree=api_toc_tree)
    path = Path(__file__).parent.parent / "docs" / "_quarto.yml"
    with path.open("w") as fi:
        fi.write(out)


if __name__ == "__main__":
    create_quarto_qmd()
