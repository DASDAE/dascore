---
name: draft-release
description: Draft the next release version and changelog by fetching tags, computing the next semantic v* tag, collecting merged PRs into the release branch (usually dev) since last release via gh, and printing the proposed version and categorized changelog.
---

# draft-release

Draft the next release version and changelog from merged PRs.

## Inputs

- `release_type` (optional): `major`, `minor`, `patch`, or `bugfix`.
- If omitted, default to `bugfix` (`patch` and `bugfix` are equivalent).

## Workflow

0. Ask for elevated permissions with network access to run the `git fetch` and GitHub CLI PR commands used to collect merged PRs (for example `gh pr list`, `gh pr view`, or `gh api`).
1. Fetch the latest refs and tags from `origin` only. Do **not** use
   `git fetch --all` — it also fetches unrelated remotes (e.g. a personal fork or
   another contributor's remote) and can fail on those or clobber local tags,
   aborting the whole fetch:

```bash
git fetch --tags origin
```

   If you only need to read the tags without touching local state, use
   `git ls-remote --tags origin` instead.

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
- Pre-releases break both of these rules, and drafting from `dev` (step 3) is
  exactly when they occur. Before computing anything, check whether the target
  branch carries a pre-release tag (`aN`, `bN`, `rcN`) newer than the latest
  stable one. If it does: the lower bound for step 3 is that pre-release tag, not
  the last stable release, or the notes will repeat what the pre-release already
  published; and the next tag continues the same series (`v0.2.0b1` →
  `v0.2.0b2`), or finalizes it to `vX.Y.Z` only when the caller says the series
  is ending. Ask which is intended rather than assuming the stable tag.

3. Collect merged PRs since the last release:
- Use the previous release tag identified in step 2 as the lower bound.
- Define PR scope as changes reachable in `last_release_tag..origin/<target>`,
  where `<target>` is the branch the release will be cut from. Ask if it was not
  given. It is **not** always `master`: feature work lands on `dev` and
  pre-releases are tagged there (see the release docs), so scoping to `master`
  silently omits everything not yet merged down. Sanity-check the choice before
  drafting — if `git rev-list --count last_release_tag..origin/<target>` is far
  larger than the count against `origin/master`, `dev` is the branch you want.
- Use GitHub CLI (`gh`) to read merged PRs for that scope.
- Include at minimum PR number, title, merge date, URL, labels, and body text.
- Look at the git diffs to extract additional info if needed.

4. Draft a changelog with these sections, in this order, omitting any that end
   up empty:
- `Breaking Changes` — every entry marked `**breaking**`, repeated here from the
   category section it also belongs to, so an upgrader sees them first.
- `Added`
- `Changed`
- `Deprecated`
- `Removed`
- `Fixed`
- `Security`

5. Drop reverted pairs first. If a PR in scope reverts another PR that is also
   in scope (revert PRs usually say "Revert ..." and name the reverted PR or
   commit in the title/body), the two cancel out to no net user-facing change.
   Omit both from the sections and instead list them under a short
   `Reverted (no net change)` note at the end, so the reader knows why those PR
   numbers are absent.

6. Collect the entries. The repository no longer maintains a changelog, so the
   PR bodies are the primary record of user-facing intent. Each body carries a
   `## Changelog` section whose bullets are already written and categorized by
   the author, and CI rejects a PR whose section is missing or malformed. Read
   them rather than re-deriving them:

```text
- <category>: <text>
- <category> **breaking**: <text>
```

   where `<category>` is one of `added`, `changed`, `deprecated`, `removed`,
   `fixed`, `security`. Take each bullet as one entry, place it in the section
   named by its category, and additionally list every `**breaking**` entry under
   `Breaking Changes`. Preserve the author's wording; tighten only for length.

   `**breaking**` means the entry can break code written against the *last
   released version*. Work that only ever existed on `dev` is not marked, so do
   not add the marker yourself from the diff — a change that looks drastic may be
   breaking nothing any user has.

   The section is uninformative in two distinguishable ways: a literal `none` is
   the author stating the PR has no user-facing effect and is trustworthy, while
   an *empty* or absent section means it predates this policy (or slipped
   through) — fall back to the title, body, and diff there rather than dropping
   the PR, and classify it yourself using the guidance below.

   One-time note, applying only to the first release drafted once
   `docs/changelog.qmd` is a stub pointing at the releases page. Until then the
   page still carries its own curated entries and you should read the working
   copy. Afterwards, those entries — richer than the PR bodies alone for work
   merged before the template gained its release-note sections — are still in
   git history. Find the commit that reduced the page to a stub, then read the
   revision before it:

```bash
git log --oneline -- docs/changelog.qmd
git show <commit-before-the-stub>:docs/changelog.qmd
```

   For those fallback PRs only, this repo does not use conventional-commit
   markers, and its labels are topical (`proc`, `viz`, `spool`, `IO`,
   `transform`, `bug`, ...) rather than semantic, so labels alone are not enough
   — use judgment, assigning the same categories the authors would have:
- `Removed` for deleted public API, `Deprecated` for API marked for removal,
   `Security` for a vulnerability fix.
- `Added` if the PR introduces a capability or option, `Fixed` if it corrects
   wrong behavior, `Changed` for anything else observable, including notable
   performance changes.
- Mark an entry `**breaking**` only if it breaks code written against the last
   released version. Check the tag rather than guessing: an API introduced after
   that tag cannot break anyone.
- Prefer user-facing behavior over internal implementation when deciding and
   when summarizing.
- A PR is not limited to one entry. Large PRs routinely carry several unrelated
   user-facing changes spanning more than one category — split them into separate
   entries rather than compressing the PR into a single line.
- Omit PRs with no user-facing effect: the author wrote `none`, or the diff shows
   the change is purely internal (refactors, CI, typing, tests, docs
   infrastructure). Omitted PRs belong in no section; do not let them fall
   through to `Fixed`. Judge an empty section from the diff, not from the
   emptiness itself.
- Sort entries within each section by PR number ascending.
- Include a link to the PR in the changelog.

7. Print to screen:
- The new version tag.
- The drafted changelog.

## Output Format

```text
Next Version: vX.Y.Z

## Breaking Changes
- #125: Short summary (https://github.com/OWNER/REPO/pull/125)

## Added
- #123: Short summary (https://github.com/OWNER/REPO/pull/123)

## Changed
- #125: Short summary (https://github.com/OWNER/REPO/pull/125)

## Fixed
- #124: Short summary (https://github.com/OWNER/REPO/pull/124)

Reverted (no net change): #126 reverted by #127
```

## Notes

- Omit a section entirely when it has no entries, rather than printing `- None`.
  `Deprecated`, `Removed`, and `Security` are empty in most releases.
- A `**breaking**` entry appears twice: under `Breaking Changes` and under its
  own category, as #125 does above.
- Prefer explicit, user-facing PR summaries over internal implementation details.
- If no merged PRs are found in scope, print the next version and say plainly
  that no user-facing changes were found.
- Omit the `Reverted (no net change)` line when no reverted pairs exist.
