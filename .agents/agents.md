# DASCore Agent Guide

Checklist for contributing to DASCore. Also load `.agents/agents.local.md` if present.

## Priorities

Minimal, targeted, test-backed changes. Follow nearby patterns over personal preference, and prefer the simpler behavior-preserving option. State assumptions in the PR.

## Workflow

- Branch, and work in a worktree at `worktrees/{slug}` — never `.agents/worktrees`. One logical change per commit.
- Open PRs against `dev`; it merges to `master` at release time.
- Use an environment named for the worktree slug (mamba or uv) so editable installs stay isolated.

```bash
pip install -e ".[dev]" && pre-commit install -f
```

## Checks

Not finished until these pass.

```bash
pre-commit run --all                                    # twice; first pass auto-fixes
pytest tests/path/to/affected_test.py                   # then: pytest tests
pytest tests --cov dascore --cov-report term-missing
pytest dascore --doctest-modules
```

## Tests

- Under `tests/`, mirroring the package, grouped in classes.
- Fixtures close to use: class, module, then `conftest.py`.
- Test boundaries, not implementation. Short names, detail in the docstring.

## Code

- Imports at module top, tests included; function-level only for optional dependencies (`dascore.utils.misc.optional_import`) or circular imports, with a comment saying which.
- Dataframes: snake_case columns, getitem not getattr, non-inplace unless required.
- Type hints on public functions. NumPy-style docstrings with short examples, short docstrings on private objects, comments only where intent is unclear.
- Suppress warnings only through `dascore.utils.misc.suppress_warnings`.

## Docs

- `.qmd` in the top-level section directories (`tutorial/`, `recipes/`, `notes/`, `contributing/`, `about/`), with `index.qmd` as the landing page. API docs come from docstrings. Update with any behavior or API change. Do not hard-wrap prose.
- Site structure, styling and the curated API reference are configured in `great-docs.yml`; the `great-docs/` directory is generated. New public API needs a reference entry, and `tests/test_doc_coverage.py` fails when it has none.

```bash
pip install "great-docs>=0.16"
great-docs build      # writes great-docs/_site
great-docs preview    # serves the built site
```

### Changelog

No changelog file, and do not add one — no `CHANGELOG.md`, `changelog.d/`, or "unreleased" sections; the site's changelog page is generated from the GitHub releases at build time, and `tests/test_changelog.py` fails if a source page appears. Put the summary in the PR's required `## Changelog` section, formatted per "Changelog entries" in `contributing/general_guidelines.qmd`; `.github/scripts/check_pr_changelog.py` is the parser CI runs.

## Before handing off

Changed paths run; tests and lint pass; docs cover user-visible changes; no unrelated refactors bundled in.
