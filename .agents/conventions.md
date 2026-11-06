# DASCore conventions

Read the section relevant to your change.

## Code

- Put imports at module top, including tests. Function-level imports are for optional dependencies (use `dascore.utils.misc.optional_import`) or circular imports; comment why.
- Dataframe columns use snake_case and getitem access. Avoid inplace operations unless required.
- Public functions have type hints and NumPy-style docstrings with short examples. Private objects have short docstrings; comment unclear intent.
- Suppress warnings through `dascore.utils.misc.suppress_warnings`.

## Tests

- Mirror the package under `tests/` and group tests in classes.
- Keep fixtures near use: class, module, then `conftest.py`.
- Test behavior and boundaries; use short names and explanatory docstrings.
- Start with `python -m pytest tests/path/to/affected_test.py`. For broad changes use `python -m pytest tests`; add `--cov dascore --cov-report term-missing` when coverage is useful, rather than repeating the suite.
- Check changed docstring examples with `python -m pytest path/to/module.py --doctest-modules`.

## Docs

- Edit `.qmd` files under `dascore/docs/`; API docs come from docstrings. Do not hard-wrap prose.
- A skill's text lives in `dascore/.agents/skills/<name>/SKILL.md`. Its docs page only includes that file, and `dascore/docs/skills.yml` maps the skill to the page. Write links in a skill file relative to the page that includes it.
- Edit `scripts/_templates/_quarto.yml` for site structure; `dascore/docs/_quarto.yml` is generated.
- For changes needing a docs build: `python scripts/build_api_docs.py` then `quarto render dascore/docs`.
- The changelog stub is pinned by `tests/test_changelog.py`; PR entries are validated by `.github/scripts/check_pr_changelog.py`.
