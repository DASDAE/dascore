 <!--
Thanks for contributing to DASCore, community contributions are most welcomed!

Before contributing, please read through the [contributors doc](https://dascore.org/contributing/contributing.html)

Before making big changes to the code or adding large complex features, it is a good idea to [open a discussion](https://github.com/DASDAE/dascore/discussions). Don't hesitate to ask a question or ask for
help if something isn't clear.
-->

## Description

<!--
Please describe your PR here. What problem are you trying to solve, or what feature are you adding?

Also link any relevant issues/discussions (this can be done using the issue/discussion number preceded by a
pound sign, e.g. `#12` without the backticks)
-->

## Changelog

<!--
Required, and checked by CI. DASCore keeps no changelog file; the release notes are assembled from
merged pull requests, so these bullets are this PR's changelog entry.

Write one bullet per user-facing change, each starting with a category:

    added, changed, deprecated, removed, fixed, security

Add **breaking** after the category when the change can break code written against the last released
version. Breaking only against unreleased work on dev does not count, since users never saw it.

Describe what changes for the reader and what to do about it; leave implementation detail to the diff.
Write the single word "none" if nothing user-facing changes (refactors, CI, typing, tests).

    - added: `Patch.enrich` copies inventory metadata onto a patch.
    - changed **breaking**: `dc.set_config` is no longer a context manager; use `dc.config_context`.
    - fixed: chunking a coordinate in feet no longer drops samples.
-->

## Checklist

I have:

- [ ] filled in the Changelog section above, writing "none" if nothing user-facing changes, since release notes are assembled from PRs.

I have (if applicable):

- [ ] referenced the GitHub issue this PR closes.
- [ ] documented the new feature with docstrings and/or appropriate doc page.
- [ ] included tests. See [testing guidelines](https://dascore.org/contributing/testing.html).
- [ ] added the "ready_for_review" tag once the PR is ready to be reviewed.
