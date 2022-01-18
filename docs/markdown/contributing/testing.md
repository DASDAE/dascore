
# Testing DASCore

DASCore's test suite is run with [pytest](https://docs.pytest.org/en/stable/). While in the base dascore repo
(and after [installing DASCore for development](dev_install.md)) invoke pytest from the command line:

<!--pytest-codeblocks:skip-->
```bash
pytest tests
```

You can also use the cov flags to check coverage.

<!--pytest-codeblocks:skip-->
```bash
pytest tests --cov dascore --cov-report term-missing
```

## Writing Tests

Tests should go into the `Tests/` folder, which mirrors the structure of the main package.
For example, if you are writing tests for `dascore.Patch`, whose class definition is
located in `dascore/core/patch` it should go in `tests/test_core/test_patch.py`.

In general, tests should be grouped together in classes. Fixtures go as close as
possible to the test(s) that need them, going from class, module, and conftest fixtures.
Checkout the pytest documentation for a [review on fixtures](https://docs.pytest.org/en/6.2.x/fixture.html)
(and why to use them).
