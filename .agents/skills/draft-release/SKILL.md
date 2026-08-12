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
- Pre-releases break both rules. If the target branch carries an `aN`/`bN`/`rcN`
  tag newer than the latest stable one, use *it* as step 3's lower bound (or the
  notes repeat what it already published) and continue its series
  (`v0.2.0b1` → `v0.2.0b2`). Ask before finalizing to a stable tag instead.

3. Collect merged PRs since the last release:
- Use the previous release tag identified in step 2 as the lower bound.
- Define PR scope as `last_release_tag..origin/<target>`, where `<target>` is the
  branch being released; ask if not given. It is **not** always `master` — work
  lands on `dev`, so scoping to `master` silently omits everything not yet merged
  down. Compare `git rev-list --count` for both before choosing.
- Use GitHub CLI (`gh`) to read merged PRs for that scope.
- Include at minimum PR number, title, merge date, URL, labels, and body text.
- Look at the git diffs to extract additional info if needed.

4. Draft a changelog using the sections, in the order, listed under "Draft the
   release notes" in `docs/contributing/publish_a_new_release.qmd`. Omit any
   section that ends up empty.

5. Drop reverted pairs first. If a PR in scope reverts another PR that is also
   in scope (revert PRs usually say "Revert ..." and name the reverted PR or
   commit in the title/body), the two cancel out to no net user-facing change.
   Omit both from the sections and instead list them under a short
   `Reverted (no net change)` note at the end, so the reader knows why those PR
   numbers are absent.

6. Collect the entries. Each PR body carries a `## Changelog` section, already
   written and categorized by its author in the format defined in
   `docs/contributing/general_guidelines.qmd` and enforced by
   `.github/scripts/check_pr_changelog.py`. Read those bullets rather than
   re-deriving them: each becomes one entry under the section its category
   names, with every `**breaking**` entry also listed under `Breaking Changes`.
   Preserve the author's wording; tighten only for length. Never add the marker
   yourself from the diff — it means "breaks the last *released* version", so a
   drastic-looking change may break nothing any user has.

   A literal `none` is trustworthy. An *empty* or absent section means the PR
   predates this policy — fall back to its title, body, and diff, and classify
   it yourself using the guidance below.

   For the first release after `docs/changelog.qmd` became a stub, also read its
   pre-stub revision, as described under "Draft the release notes" in
   `docs/contributing/publish_a_new_release.qmd`.

   Classifying fallback PRs yourself: labels here are topical (`proc`, `spool`,
   `IO`, ...) rather than semantic, so judge from the change — `Added` for a new
   capability, `Fixed` for corrected behavior, `Changed` for anything else
   observable including performance. Check the tag before marking anything
   breaking; an API introduced after it cannot break anyone.
- Split a PR's unrelated changes into separate entries rather than one line.
- Omit PRs with no user-facing effect; do not let them fall through to `Fixed`.
- Sort entries within each section by PR number ascending, and link each PR.

7. Print to screen:
- The new version tag.
- The drafted changelog.

## Output Format

```text
Next Version: vX.Y.Z

## Breaking Changes
- #125: Short summary (https://github.com/OWNER/REPO/pull/125)

## Changed
- #125: Short summary (https://github.com/OWNER/REPO/pull/125)

Reverted (no net change): #126 reverted by #127
```

## Notes

- A `**breaking**` entry appears twice, as #125 does above: under `Breaking
  Changes` and under its own category.
- Omit empty sections, and the `Reverted` line when no reverted pairs exist.
- If no merged PRs are in scope, print the version and say so plainly.
