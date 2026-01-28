# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

DASCore is a Python library for distributed fiber optic sensing (DAS/DTS data processing). The library is designed around the core concept of "Patches" - data containers that hold fiber optic sensing data with associated metadata and coordinates.

## Core Architecture

### Key Components
- **Patch** (`dascore/core/patch.py`): Primary data container holding fiber optic data arrays and metadata
- **Spool** (`dascore/core/spool.py`): Collection/sequence of patches for batch processing
- **CoordManager** (`dascore/core/coordmanager.py`): Manages coordinate systems and transformations
- **IO System** (`dascore/io/`): Extensive support for various DAS file formats (20+ formats including OptoDAS, Terra15, TDMS, H5, etc.)
- **Processing** (`dascore/proc/`): Signal processing operations (filtering, resampling, coordinate selection, etc.)
- **Transform** (`dascore/transform/`): Mathematical transformations (FFT, spectrograms, dispersion, strain calculations)
- **Visualization** (`dascore/viz/`): Plotting utilities for DAS data

### Entry Points System
The project uses extensive entry points (`pyproject.toml`, see the entry points section) to register IO formatters for different DAS file formats. Each format (e.g., `OPTODAS__V9`, `TERRA15__V6`) maps to specific IO handler classes.

## Coding guidelines
All public docstrings must have a "full" docstring which includes parameters and examples. The example should normally be doctest style. Each example should have a number and a comment. Examples are separated by empty lines. 
Private methods/functions (start with _) may have a simpler docstring with only short description. 

See docs/contributing for more guidelines.

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate dascore

# Or install with pip
pip install -e ".[dev]"
```

### Testing
```bash
# Run full test suite with coverage
pytest -s --cov dascore --cov-append --cov-report=xml

# Run specific test directory
pytest tests/test_core/

# Run single test file
pytest tests/test_core/test_patch.py

# Run doctests
pytest dascore --doctest-modules
```

### Linting and Code Quality
```bash
# Run all pre-commit hooks
pre-commit run --all

# Run specific tools
ruff check --fix
ruff format
```

### Documentation
Scripts in `scripts/` directory handle documentation generation:
- `scripts/build_api_docs.py` - API documentation generation
- `scripts/_render_api.py` - API rendering utilities



## Code Organization Patterns

### IO Module Structure
Each format in `dascore/io/` follows a consistent pattern:
- `core.py` - Main formatter classes inheriting from base IO classes
- `utils.py` - Format-specific utilities and helpers
- Format classes implement `read()`, `write()`, `scan()`, and `get_format()` methods

### Processing Functions
Functions in `dascore/proc/` typically:
- Accept a Patch as first argument
- Return a modified Patch
- Use `@patch_function` decorator for method chaining
- Follow functional programming patterns

### Coordinate System
The coordinate system is central to DASCore:
- Time coordinates (usually datetime64)
- Distance coordinates (usually float64 for meters)
- Other dimensions as needed
- Coordinates are managed through the CoordManager system

## Testing Patterns

### Test Structure
- `tests/` mirrors `dascore/` directory structure
- `conftest.py` contains shared fixtures
- IO tests use example data and mock file formats
- Integration tests in `test_integrations/`
- Tests are grouped into classes

### Key Test Utilities
- Fixture-based example patches and spools
- Mock file system for IO testing
- Parameterized tests for multiple formats/conditions

## Build System

- Uses `setuptools` with `setuptools-scm` for version management
- Version determined from git tags
- Package metadata in `pyproject.toml`
- Supports Python 3.10-3.14
