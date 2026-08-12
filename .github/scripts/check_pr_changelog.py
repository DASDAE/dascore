"""
Validate the Changelog section of a pull request body.

Release notes are assembled from merged pull requests, so each one must state
its user-facing changes in a machine-readable form. This script checks the body
supplied in the PR_BODY environment variable and exits non-zero, explaining
what is wrong, when it does not conform.
"""

from __future__ import annotations

import os
import re
import sys

CATEGORIES = ("added", "changed", "deprecated", "removed", "fixed", "security")

# "- <category>: text", optionally marked "- <category> **breaking**: text".
# The category is matched case-insensitively; a capitalized one is a harmless
# slip, unlike a misplaced marker.
ENTRY = re.compile(
    rf"^-\s+(?P<category>{'|'.join(CATEGORIES)})"
    r"(?P<breaking>\s+\*\*breaking\*\*)?:\s+(?P<text>\S.*)$",
    re.IGNORECASE,
)

# A marker anywhere but right after the category parses as ordinary text, so the
# entry would pass while its breaking status went unrecorded. Reject it instead.
_STRAY_MARKER = re.compile(r"\*\*\s*breaking\s*\*\*", re.IGNORECASE)

# The section this script validates, and the start of whatever section follows.
_HEADING = re.compile(r"^##\s+changelog\s*$", re.IGNORECASE)
_NEXT_HEADING = re.compile(r"^##\s+\S")

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_ANY_BULLET = re.compile(r"^[-*+]\s+")

_HELP = (
    "The Changelog section must hold one bullet per user-facing change, each "
    f"starting with a category ({', '.join(CATEGORIES)}), or the single word "
    "'none' when the change is internal. Mark an entry '**breaking**' when it "
    "can break code written against the last released version. Example:\n"
    "    - added: `Patch.enrich` copies inventory metadata onto a patch.\n"
    "    - changed **breaking**: `dc.set_config` is no longer a context manager."
)


def extract_section(body: str) -> str | None:
    """Return the text of the Changelog section, or None if it is absent."""
    lines = _COMMENT.sub("", body).splitlines()
    for index, line in enumerate(lines):
        if not _HEADING.match(line.strip()):
            continue
        section = []
        for following in lines[index + 1 :]:
            if _NEXT_HEADING.match(following.strip()):
                break
            section.append(following)
        return "\n".join(section)
    return None


def validate(body: str | None) -> list[str]:
    """Return a list of problems with the body; empty means it conforms."""
    if not (body or "").strip():
        return ["The pull request body is empty.", _HELP]
    section = extract_section(body or "")
    if section is None:
        return ["The pull request body has no '## Changelog' section.", _HELP]
    stripped = section.strip()
    if not stripped:
        return ["The Changelog section is empty.", _HELP]
    if stripped.lower() == "none":
        return []
    bullets = [x for x in section.splitlines() if _ANY_BULLET.match(x.strip())]
    if not bullets:
        return ["The Changelog section has no entries.", _HELP]
    problems = [x.strip() for x in bullets if not ENTRY.match(x.strip())]
    if problems:
        listed = "\n".join(f"    {x}" for x in problems)
        return [f"These Changelog entries are malformed:\n{listed}", _HELP]
    misplaced = [
        x.strip()
        for x in bullets
        if _STRAY_MARKER.search(ENTRY.match(x.strip()).group("text"))
    ]
    if misplaced:
        listed = "\n".join(f"    {x}" for x in misplaced)
        return [
            "These entries carry '**breaking**' in their text, where it is read "
            f"as prose rather than as the marker:\n{listed}\n"
            "Put it directly after the category, before the colon, e.g. "
            "'- changed **breaking**: ...'.",
        ]
    return []


def main() -> int:
    """Check PR_BODY and report any problems."""
    problems = validate(os.environ.get("PR_BODY"))
    if not problems:
        print("Changelog section is valid.")  # noqa: T201
        return 0
    for problem in problems:
        print(problem, file=sys.stderr)  # noqa: T201
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
