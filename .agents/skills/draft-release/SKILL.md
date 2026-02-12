---
name: draft-release
description: Draft the next release version and changelog by fetching tags, computing the next semantic v* tag, collecting merged PRs into master since last release via gh, and printing the proposed version and categorized changelog.
---

# draft-release

Draft the next release version and changelog from merged PRs.

## Inputs

- `release_type` (optional): `major`, `minor`, `patch`, or `bugfix`.
- If omitted, default to `bugfix` (`patch` and `bugfix` are equivalent).

## Workflow

0. Ask for elevated permissions with network access to run `git fetch --all --tags` and the GitHub CLI PR commands used to collect merged PRs (for example `gh pr list`, `gh pr view`, or `gh api`).
1. Fetch the latest refs and tags:

```bash
git fetch --all --tags
```

2. Determine the next version number:
- Consider only tags that start with `v` and match strict semver:
  `^v[0-9]+\.[0-9]+\.[0-9]+$` (ignore pre-release/build suffixes).
- Find the latest existing release tag from that set (e.g., `v1.2.3`).
- If no matching tags exist, use `v0.0.0` as the base.
- Bump according to `release_type`:
  - `major`: `X+1.0.0`
  - `minor`: `X.Y+1.0`
  - `patch`/`bugfix` (default): `X.Y.Z+1`
- Output the computed new tag as `vX.Y.Z`.

3. Collect merged PRs since the last release:
- Use the previous release tag identified in step 2 as the lower bound.
- Define PR scope as changes reachable in `last_release_tag..origin/master`
  (or `..origin/<default_branch>` if default branch is not `master`).
- Use GitHub CLI (`gh`) to read merged PRs for that scope.
- Include at minimum PR number, title, merge date, URL, labels, and body text.
- Look at the git diffs to extract additional info if needed.

4. Draft a changelog with these sections:
- `New Features`
- `Bug Fixes`
- `Breaking Changes`

5. Classify PRs deterministically:
- `Breaking Changes` if any of:
  - title contains `!` in conventional-commit style segment, or
  - label indicates breaking change (e.g., `breaking`), or
  - body contains `BREAKING CHANGE`.
- Otherwise `New Features` if labels/titles indicate feature work
  (e.g., `feature`, `enhancement`, `feat`).
- Otherwise `Bug Fixes`.
- Sort entries by PR number ascending.
- Include a link to the PR in the changelog.

6. Print to screen:
- The new version tag.
- The drafted changelog.

## Output Format

```text
Next Version: vX.Y.Z

## New Features
- #123: Short summary (https://github.com/OWNER/REPO/pull/123)

## Bug Fixes
- #124: Short summary (https://github.com/OWNER/REPO/pull/124)

## Breaking Changes
- #125: Short summary (https://github.com/OWNER/REPO/pull/125)
```

## Notes

- If there are no items for a section, include the section with `- None`.
- Prefer explicit, user-facing PR summaries over internal implementation details.
- If no merged PRs are found in scope, still print the next version and include
  all sections with `- None`.
