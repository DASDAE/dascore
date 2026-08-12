"""
Tests which enforce that DASCore keeps no changelog.

Release notes are assembled from the pull requests merged since the last tag,
each of which describes its own user-facing and breaking changes. The changelog
page survives only as a stub preserving its published URL, so these tests fail
if anything starts accumulating there again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_PATH = _REPO_ROOT / "docs"
_CHANGELOG_PATH = _DOC_PATH / "changelog.qmd"

_RELEASES_URL = "https://github.com/DASDAE/dascore/releases"

# The stub is ~75 words; the cap leaves room to reword it but not to log changes.
_MAX_WORDS = 150

# Markdown list items, which is how changelog entries have always been written.
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")

# Any heading below the page title, e.g. the old "## Unreleased API Changes".
_SUB_HEADING = re.compile(r"^\s*#{2,}\s")

_POLICY = (
    "DASCore does not keep a changelog. Describe user-facing and breaking "
    "changes in the pull request that makes them, under the headings in "
    ".github/pull_request_template.md; they are collected into the release "
    "notes when a version is tagged. See .agents/agents.md."
)

pytestmark = pytest.mark.skipif(
    not _DOC_PATH.is_dir(),
    reason="docs are only present in a source checkout",
)


@pytest.fixture(scope="module")
def changelog_text() -> str:
    """The contents of the changelog page."""
    if not _CHANGELOG_PATH.exists():
        pytest.fail(f"{_CHANGELOG_PATH} is missing; see test_page_exists.")
    return _CHANGELOG_PATH.read_text(encoding="utf-8")


class TestChangelogIsAStub:
    """The changelog page must stay a pointer to the releases page."""

    def test_page_exists(self):
        """The page stays put; deleting it 404s the published changelog URL."""
        assert _CHANGELOG_PATH.exists(), (
            f"{_CHANGELOG_PATH} is missing. The page is published as "
            "dascore.org/changelog.html, so removing it breaks existing links. "
            "Keep the stub."
        )

    def test_points_to_releases_page(self, changelog_text):
        """The page must send readers to the releases page."""
        assert _RELEASES_URL in changelog_text, (
            f"The changelog page must link to {_RELEASES_URL}, which is where "
            "changes between versions are tracked."
        )

    def test_explains_where_changes_are_described(self, changelog_text):
        """The page must say that each PR describes its own changes."""
        assert "pull request" in changelog_text.lower(), (
            "The changelog page must tell readers that each pull request "
            f"describes its own changes. {_POLICY}"
        )

    def test_has_no_entries(self, changelog_text):
        """No list items: this is how changelog entries get added."""
        entries = [x for x in changelog_text.splitlines() if _LIST_ITEM.match(x)]
        assert not entries, (
            f"The changelog page has {len(entries)} list item(s), starting with "
            f"{entries[0].strip()!r}. {_POLICY}"
        )

    def test_has_no_sections(self, changelog_text):
        """No sub-headings, e.g. the old 'Unreleased API Changes' section."""
        headings = [x for x in changelog_text.splitlines() if _SUB_HEADING.match(x)]
        assert not headings, (
            f"The changelog page has added section(s): {headings}. {_POLICY}"
        )

    def test_stays_short(self, changelog_text):
        """A pointer, not a log."""
        word_count = len(changelog_text.split())
        assert word_count <= _MAX_WORDS, (
            f"The changelog page has grown to {word_count} words (limit "
            f"{_MAX_WORDS}). {_POLICY}"
        )
