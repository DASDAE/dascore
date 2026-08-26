"""
Freeze the URLs the documentation publishes, so a later change can be checked.

The cross reference maps every symbolic key the docs can link to, one per
public object and one per alias of it, onto the qmd file that key resolves
to. Those paths are the published URLs. Aggregating members onto their
owner's page moves them, and a moved URL is a broken link for anyone who
saved one, so the mapping is frozen here before it changes.

Freezing a URL is not a promise that the object behind it is public; it only
records where the object is published today.

    freeze    write the current mapping to the baseline
    check     report the keys which moved or vanished since the baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_PATH = Path(__file__).absolute().parent.parent
CROSS_REF_PATH = REPO_PATH / "docs" / ".cross_ref.json"
BASELINE_PATH = Path(__file__).absolute().parent / "_baselines" / "api_urls.tsv"

_HEADER = (
    "# The API URLs the docs published when this baseline was written.\n"
    "# Regenerate with `python scripts/_api_urls.py freeze`, and say in the\n"
    "# pull request why each key moved or vanished.\n"
)


def _echo(message):
    """Print a message; this script talks to a build log."""
    print(message)  # noqa: T201


def current_urls(cross_ref_path: Path = CROSS_REF_PATH) -> dict[str, str]:
    """Return the API part of the cross reference, key to published path."""
    mapping = json.loads(cross_ref_path.read_text())
    return {k: v for k, v in mapping.items() if v.startswith("/api/")}


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, str]:
    """Load the frozen mapping, or an empty one if nothing is frozen yet."""
    if not path.exists():
        return {}
    lines = (x for x in path.read_text().splitlines() if x and not x.startswith("#"))
    return dict(x.split("\t", 1) for x in lines)


def write_baseline(urls: dict[str, str], path: Path = BASELINE_PATH) -> None:
    """Write the mapping as one sorted key and path per line."""
    path.parent.mkdir(exist_ok=True, parents=True)
    rows = "".join(f"{key}\t{urls[key]}\n" for key in sorted(urls))
    path.write_text(_HEADER + rows)


def compare(current: dict[str, str], baseline: dict[str, str]) -> dict:
    """Say which keys were added, which moved, and which vanished."""
    moved = {k: (v, current[k]) for k, v in baseline.items() if current.get(k, v) != v}
    return {
        "added": sorted(set(current) - set(baseline)),
        "removed": sorted(set(baseline) - set(current)),
        "moved": {k: list(v) for k, v in sorted(moved.items())},
    }


def _summarize(difference: dict) -> str:
    """Describe a comparison in one line."""
    counts = ", ".join(
        f"{len(difference[x])} {x}" for x in ("added", "removed", "moved")
    )
    return f"API URLs against the baseline: {counts}"


def check(strict: bool = False, path: Path = BASELINE_PATH) -> int:
    """Report how the current URLs differ from the frozen ones."""
    baseline = load_baseline(path)
    if not baseline:
        _echo(f"No frozen API URLs at {path}; run `_api_urls.py freeze` first.")
        return 1 if strict else 0
    difference = compare(current_urls(), baseline)
    _echo(_summarize(difference))
    for name in ("removed", "moved"):
        for key in list(difference[name])[:20]:
            _echo(f"  {name}: {key}")
    broken = len(difference["removed"]) + len(difference["moved"])
    return 1 if strict and broken else 0


def _get_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write the current mapping to the baseline")
    checker = sub.add_parser("check", help="report keys which moved or vanished")
    checker.add_argument("--strict", action="store_true", help="fail if any did")
    return parser


def main(args=None) -> int:
    """Run one sub command."""
    parsed = _get_parser().parse_args(args)
    if parsed.command == "freeze":
        urls = current_urls()
        write_baseline(urls)
        _echo(f"Froze {len(urls)} API URLs to {BASELINE_PATH}")
        return 0
    return check(strict=parsed.strict)


if __name__ == "__main__":
    sys.exit(main())
