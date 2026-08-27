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

- `.qmd` under `docs/`; API docs come from docstrings. Update with any behavior or API change. Do not hard-wrap prose.
- Edit `scripts/_templates/_quarto.yml` for site structure; `docs/_quarto.yml` is generated.

```bash
python scripts/build_api_docs.py && quarto render docs
```

### Changelog

No changelog file, and do not add one — no `CHANGELOG.md`, `changelog.d/`, or "unreleased" sections. `docs/changelog.qmd` is a stub pinned by `tests/test_changelog.py`. Put the summary in the PR's required `## Changelog` section, formatted per "Changelog entries" in `docs/contributing/general_guidelines.qmd`; `.github/scripts/check_pr_changelog.py` is the parser CI runs.

## Before handing off

Changed paths run; tests and lint pass; docs cover user-visible changes; no unrelated refactors bundled in.
