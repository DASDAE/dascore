"""Script to validate links in qmd files."""
from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path


def _get_docs_path():
    """Find the documentation path."""
    path = Path(__file__).parent.parent / "docs"
    return path


def get_qmd_files(
    path=None,
):
    """Yield all QMD files."""
    path = _get_docs_path() if path is None else path
    yield from path.rglob("*qmd")


def yield_links(text, pattern=r"(?<=\]\(`).*?(?=`\))"):
    """Yield links found in documentation."""
    matches = re.findall(pattern, text)
    yield from matches


@cache
def load_index(path=None):
    """Load the index with the linked locations."""
    if path is None:
        path = _get_docs_path() / "api" / "cross_ref.json"
    with open(path) as fi:
        out = json.load(fi)
    return out


def validate_all_links():
    """Scan all documentation files and ensure the links are valid."""
    index = load_index()
    good_links, bad_links, file_count = 0, 0, 0
    bad = []
    for path in get_qmd_files():
        file_count += 1
        text = path.read_text()
        for link in yield_links(text):
            if link not in index:
                bad.append((str(path), link))
                bad_links += 1
            else:
                good_links += 1
    print(  # noqa
        f"Validated links in documentation. Scanned {file_count} files, "
        f"found {good_links} good links and {bad_links} bad links"
    )
    if bad_links:
        msg = "Please fix the following (path/link)\n"
        max_len = max(len(x[0]) for x in bad)
        out = []
        for path, link in bad:
            path_str = path.ljust(max_len + 3)
            out.append(f"{path_str} {link}")
        new_str = msg + "\n".join(out)
        raise ValueError(new_str)


if __name__ == "__main__":
    validate_all_links()
    pass
