"""
Test which enforces that DASCore keeps no changelog.

Release notes are assembled from the pull requests merged since the last tag,
each of which describes its own user-facing and breaking changes. The changelog
page survives only as a stub preserving its published URL, so this test pins its
contents exactly and fails if anything is added to it.
"""

from __future__ import annotations

# E501: the expected text below is a byte-for-byte copy of a markdown page whose
# prose is deliberately not hard-wrapped; rewrapping it would break the match.
# ruff: noqa: E501
from pathlib import Path

import pytest

_DOC_PATH = Path(__file__).resolve().parents[1] / "docs"
_CHANGELOG_PATH = _DOC_PATH / "changelog.qmd"

# Run wherever the docs are, skip where they are not. The sdist grafts tests but
# ships only docs/LICENSE, so the directory existing is not enough to tell the
# two apart; index.qmd is present iff the real docs tree is. Deliberately not
# keyed on changelog.qmd, or deleting the page would skip rather than fail.
_DOCS_PRESENT = (_DOC_PATH / "index.qmd").is_file()

_EXPECTED = """# Changelog

DASCore's changelog is the [releases page](https://github.com/DASDAE/dascore/releases), which tracks changes from one version to another.

This page remains only so that existing links to it keep working; it is not part of the site navigation and is never updated per pull request. Each pull request describes its own user-facing and breaking changes, and those descriptions are collected into the release notes when a version is tagged. See [publish a new release](contributing/publish_a_new_release.qmd) for that workflow.
"""

_POLICY = (
    "DASCore does not keep a changelog. Describe user-facing and breaking "
    "changes in the pull request that makes them, under the headings in "
    ".github/pull_request_template.md; they are collected into the release "
    "notes when a version is tagged. See .agents/agents.md. The page itself "
    "must stay exactly as pinned here, since it is published as "
    "dascore.org/changelog.html."
)


@pytest.mark.skipif(
    not _DOCS_PRESENT,
    reason="the docs are not present (e.g. running from an sdist)",
)
class TestChangelogIsAStub:
    """The changelog page must stay a pointer to the releases page."""

    def test_contents_are_unchanged(self):
        """The page must match the expected stub exactly."""
        assert _CHANGELOG_PATH.exists(), f"{_CHANGELOG_PATH} is missing. {_POLICY}"
        contents = _CHANGELOG_PATH.read_text(encoding="utf-8")
        assert contents.strip() == _EXPECTED.strip(), _POLICY
