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

4. Draft a changelog with these sections:
- `New Features`
- `Bug Fixes`
- `Breaking Changes`

5. Drop reverted pairs first. If a PR in scope reverts another PR that is also
   in scope (revert PRs usually say "Revert ..." and name the reverted PR or
   commit in the title/body), the two cancel out to no net user-facing change.
   Omit both from the sections and instead list them under a short
   `Reverted (no net change)` note at the end, so the reader knows why those PR
   numbers are absent.

6. Classify the remaining PRs. The repository no longer maintains a changelog, so
   the PR bodies are the primary record of user-facing intent. Start from the
   `User-facing changes` and `Breaking changes` headings of the PR template.
   Distinguish the two ways those can be uninformative: a literal "None" is the
   author stating the PR has no user-facing effect and is trustworthy, while an
   *empty* heading means the author skipped it — fall back to the title, body,
   and diff there rather than dropping the PR. Older PRs predate the headings
   entirely and always need the fallback.

   One-time note for the first release drafted after the changelog was retired:
   `docs/changelog.qmd` previously accumulated curated entries for unreleased
   work, and that text is richer than what the PR bodies alone give you for those
   PRs. Read it before drafting:

```bash
git show "$(git log -1 --format=%H -- docs/changelog.qmd)^:docs/changelog.qmd"
```

   This repo does not use conventional-commit markers, and its
   labels are topical (`proc`, `viz`, `spool`, `IO`, `transform`, `bug`, ...)
   rather than semantic, so labels alone are not enough — use judgment:
- `Breaking Changes` if the change removes or alters existing public API,
   defaults, or behavior in a way that can break callers — regardless of whether
   any `!`, `breaking` label, or `BREAKING CHANGE` text is present. A signature
   or keyword change to a documented `Patch`/`dc` method is breaking even when
   unlabeled; when unsure, list it here with a one-line note on what changed.
- Otherwise `New Features` if the PR adds a capability, option, or notable
   performance improvement (judge from the title/body, not just a
   `feature`/`enhancement` label, which is often missing).
- Otherwise `Bug Fixes`.
- Prefer user-facing behavior over internal implementation when deciding and
   when summarizing.
- A PR is not limited to one entry. Large PRs routinely carry several unrelated
   user-facing changes spanning more than one section — split them into separate
   entries under the sections they belong to rather than compressing the PR into
   a single line. The `User-facing changes` and `Breaking changes` headings of
   the PR body are usually already itemized this way.
- Omit PRs with no user-facing effect: the author wrote "None" under *every*
   heading, or the diff shows the change is purely internal (refactors, CI,
   typing, tests, docs infrastructure). A "None" applies only to the heading it
   sits under — an ordinary non-breaking feature writes a real `User-facing
   changes` section and "None" under `Breaking changes`, and must still appear in
   the notes. Omitted PRs belong in no section; do not let them fall through to
   `Bug Fixes`. Judge an empty heading from the diff, not from the emptiness
   itself.
- Sort entries within each section by PR number ascending.
- Include a link to the PR in the changelog.

7. Print to screen:
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

Reverted (no net change): #126 reverted by #127
```

## Notes

- If there are no items for a section, include the section with `- None`.
- Prefer explicit, user-facing PR summaries over internal implementation details.
- If no merged PRs are found in scope, still print the next version and include
  all sections with `- None`.
- Omit the `Reverted (no net change)` line when no reverted pairs exist.
