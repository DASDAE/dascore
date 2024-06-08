"""Find all python files which dont have "from __future__ import annotation"."""

from __future__ import annotations

from pathlib import Path


def has_search_str(path, search_string):
    """Return True if search string in path."""
    for line in path.read_text().splitlines():
        if search_string in line:
            return True
    return False


def rewrite_file(path, import_str):
    """Rewrite file to have import statement."""
    out = path.read_text().splitlines()
    try:
        index = out[1:].index('"""')
    except ValueError:
        try:
            index = out[1:].index("'''")
        except ValueError:
            index = -1
    out.insert(index + 2, import_str)
    new_str = "\n".join(out)
    with open(path, "w") as fi:
        fi.write(new_str)


if __name__ == "__main__":
    search_str = "from __future__ import annotations"
    base = Path(__file__).parent.parent
    missing = []
    for path in base.rglob("*.py"):
        if not has_search_str(path, search_str):
            rewrite_file(path, search_str)
    for path in missing:
        print(path)  # noqa
