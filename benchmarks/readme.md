# Benchmarks

DASCore's benchmark suite uses [CodSpeed](https://codspeed.io/) for continuous performance monitoring.

## Running Benchmarks Locally

To run benchmarks locally:

```bash
# Install test dependencies, pytest-codspeed and the optional readers, as CI
# does; a format whose reader is missing is skipped, not benchmarked
pip install -e ".[profile,extras]"

# Run all benchmarks
pytest benchmarks/ --codspeed

# Run only the smoke set, as pull requests do
pytest benchmarks -m smoke --codspeed

# Run specific benchmark files
pytest benchmarks/test_patch_benchmarks.py --codspeed
pytest benchmarks/test_io_benchmarks.py --codspeed
pytest benchmarks/test_spool_benchmarks.py --codspeed
pytest benchmarks/test_import_benchmarks.py --codspeed
```

## Benchmark Structure

Benchmarks are now organized as pytest tests in the `benchmarks/` directory:

- `test_patch_benchmarks.py` - Core Patch processing, transform, and visualization benchmarks
- `test_io_benchmarks.py` - File I/O operations benchmarks
- `test_spool_benchmarks.py` - Spool chunking and selection benchmarks
- `test_lookup_benchmarks.py` - In-memory lookups on hot paths (format resolution, remote-cache and IO handle resolution, repeat spool access). These are deliberately small: a change of a few microseconds per lookup is invisible in the end-to-end benchmarks above, because one file read costs far more than the lookups it makes.
- `test_import_benchmarks.py` - Import benchmarks (dascore's own modules; third party dependencies stay warm)

Each benchmark uses the `@pytest.mark.benchmark` decorator to automatically measure performance.

The IO benchmark uses the full data registry (all test-suite data files are hosted in the DASDAE/test_data repo and cached by CI; see `dascore/docs/contributing/adding_test_data.qmd`).

## Continuous Performance Monitoring

CodSpeed runs one of two sets:

- The full suite runs on pushes to `dev` and `master`, on manual runs, and on pull requests labelled `benchmark`.
- The smoke set, the benchmarks marked `smoke`, runs on every other pull request. CodSpeed reports the benchmarks a pull request did not run as skipped, carrying the base branch's results.

Each smoke benchmark covers a different end-to-end path: import, format detection, scan and read, indexing a directory, selecting and chunking a spool, building a spool from patches, patch processing, lazy-array slicing and format lookup. To add a benchmark to the set, mark it `smoke`:

```python
@pytest.mark.benchmark
@pytest.mark.smoke
def test_something(self):
    ...
```

or mark one parameter with `pytest.param(value, marks=pytest.mark.smoke)`. Keep the set small and cheap: under CodSpeed's simulation a benchmark costs about a hundred times its plain pytest run.

Performance results are tracked by CodSpeed and reported in pull requests, helping identify performance regressions before they're merged.

## Migration Notes

The legacy ASV benchmarks in the `benchmarks/` directory have been converted to pytest format. The new benchmarks maintain the same functionality while providing better integration with the existing test suite.
