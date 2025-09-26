# Benchmarks

DASCore's benchmark suite uses [CodSpeed](https://codspeed.io/) for continuous performance monitoring.

## Running Benchmarks Locally

To run benchmarks locally:

```bash
# Install test dependencies (includes pytest-codspeed)
pip install -e ".[test]"

# Run all benchmarks
pytest benchmarks/ --codspeed

# Run specific benchmark files
pytest benchmarks/test_patch_benchmarks.py --codspeed
pytest benchmarks/test_io_benchmarks.py --codspeed
pytest benchmarks/test_spool_benchmarks.py --codspeed
```

## Benchmark Structure

Benchmarks are now organized as pytest tests in the `benchmarks/` directory:

- `test_patch_benchmarks.py` - Core Patch processing, transform, and visualization benchmarks
- `test_io_benchmarks.py` - File I/O operations benchmarks
- `test_spool_benchmarks.py` - Spool chunking and selection benchmarks

Each benchmark uses the `@pytest.mark.benchmark` decorator to automatically measure performance.

## Continuous Performance Monitoring

Benchmarks automatically run on:
- Push to main/master branch
- Pull requests

Performance results are tracked by CodSpeed and reported in pull requests, helping identify performance regressions before they're merged.

## Migration Notes

The legacy ASV benchmarks in the `benchmarks/` directory have been converted to pytest format. The new benchmarks maintain the same functionality while providing better integration with the existing test suite.
