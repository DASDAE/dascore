#%% md

# Contributing

Contributions to fios are welcomed and appreciated. Before proceeding please be aware of our [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).


## Getting setup

The following steps are needed to setup `fios` for development:

### 1. Clone fios

<!--pytest-codeblocks:skip-->
```bash
git clone https://github.com/dasdae/fios
cd fios
```

### 2. Pull tags

Make sure to pull all of the latest git tags.

**NOTE:** You may need to do this periodically to keep tags in sync.

<!--pytest-codeblocks:skip-->
```bash
git pull origin master --tags
```

### 3. Create a virtual environment (optional)

Create and activate a virtual environment so fios will not mess with the base (or system) python installation.

If you are using [Anaconda](https://www.anaconda.com/), simply use the environment provided:

<!--pytest-codeblocks:skip-->
```bash
conda env create -f environment.yml
conda activate fios
```

### 4. Install fiosin development mode

<!--pytest-codeblocks:skip-->
```bash
pip install -e .[test,docs]
```

### 5. Setup precommit hooks

fios uses several [pre-commit](https://pre-commit.com/) hooks to ensure the code stays tidy. Please install and use them!

<!--pytest-codeblocks:skip-->
```bash
pre-commit install -f
```

## Branching and versioning

We create new features or bug fixes in their own branches and merge them into master via pull requests. We may switch to a more complex branching model if the need arises.

If substantial new features have been added since the last release we will bump the minor version.  If only bug fixes/minor changes have been made, only the patch version will be bumped. Like most python projects, we loosely follow [semantic versioning](https://semver.org/) in terms that we will not bump the major version until fios is more stable.


## Running the tests

The tests suite is run with [pytest](https://docs.pytest.org/en/stable/). While in the base fios repo (and after installing fios) invoke pytest from the command line:

<!--pytest-codeblocks:skip-->
```bash
pytest tests
```

You can also check coverage

<!--pytest-codeblocks:skip-->
```bash
pytest tests --cov fios --cov-report term-missing
```

## Contributing to the documentation

The documentation is primarily done in markdown but easily converted to jupyter notebooks using
[jupytext](https://github.com/mwouts/jupytext).


## Building the documentation

The documentation can be built using the script called "make_docs.py" in the scripts directory. If you have followed the instructions above all the required dependencies should be installed.

<!--pytest-codeblocks:skip-->
```bash
python scripts/make_docs.py
```

The docs can then be accessed by double-clicking on the newly created html index at docs/_build/html/index.html.


## General guidelines


fios uses [Black](https://github.com/ambv/black) and [flake8](http://flake8.pycqa.org/en/latest/) for code linting. If you have properly installed fios' pre-commit hooks they will be invoked automatically when you make a git commit. If any complaints are raised simply address them and try again.

Use [numpy style docstrings](https://docs.scipy.org/doc/numpy/docs/howto_document.html). All public code (doesn't start with a `_`) should have a "full" docstring but private code (starts with a `_`) can have an abbreviated docstring.


fios makes extensive use of Python 3's [type hints](https://docs.python.org/3/library/typing.html). Use them to annotate any public functions/methods.


Prefer `pathlib.Path` to strings when working with paths. However, when dealing with many many files (e.g., indexers) strings may be preferred for efficiency.



### Example functions

```python
from typing import Optional, Union

import fios
from fios.constants import PatchType


# example public Patch function
@fios.patch_function()
def example_func(patch: PatchType, to_add: Optional[Union[int, float]]) -> PatchType:
    """
    A simple, one line explanation of what this function does.

    Additional details which might be useful, and are not limited to one line.
    In fact, they might span several lines, especially if the author of the
    docstring tends to include more details than needed.

    Parameters
    ----------
    patch
        A description of this parameter
    to_add
        A description of this parameter

    Returns
    -------
    If needed, more information about what this function returns. You shouldn't
    simply specify the type here since that is already given by the type annotation.
    If the returned object is self-explanatory feel free to skip this section.

    Examples
    --------
    >>> # Examples are included in the doctest style
    >>> import fios
    ... pa = fios.get_example_patch()
    ...
    ... out = example_func(pa)
    """
    data = pa.data
    if to_add is not None:
        data += data
    return patch.__class__(data=data, attrs=patch.atts, coords=patch.coords)


# example private function
def _recombobulate(df, arg1, arg2):
    """
    A private function can have a simple (multi-line) snippet and doesn't need as
    much detail or type hinting as a public function.
    """
```


## Working with dataframes

Column names should be snake_cased whenever possible.


Always access columns with getitem and not getattr (ie use `df['column_name']` not `df.column_name`).


Prefer creating a new `DataFrame`/`Series` to modifying them inplace. Inplace modifications should require opting in (usually through an `inplace` key word argument).
