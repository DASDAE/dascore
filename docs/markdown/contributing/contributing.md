# Contributing

Contributions to DASCore are welcomed and appreciated. Before contributing
please be sure to read and understand our [code of conduct](code_of_conduct.md).

[Development Installation](dev_install.md)

[Testing](testing.md)

[Adding Support for a File Format](new_format.md)


## General Guidelines

### Branching and versioning

We create new features or bug fixes in their own branches and merge them into master via pull requests. We may switch
to a more complex branching model if the need arises.

If substantial new features have been added since the last release we will bump the minor version.  If only bug
fixes/minor changes have been made, only the patch version will be bumped. Like most python projects, we loosely
follow [semantic versioning](https://semver.org/) in terms that we will not bump the major version until DASCore
is more stable.


### Linting

DASCore uses [Black](https://github.com/ambv/black) and [flake8](http://flake8.pycqa.org/en/latest/) for code linting.
If you have properly installed DASCore' pre-commit hooks they will be invoked automatically when you make a git commit.
If any complaints are raised simply address them and try again.

As a reminder, you can install pre-commit hooks like so:

```shell
pip install pre-commit
pre-commit install
```

Then run all the hooks like this:

```shell
pre-commit run --all
```

### Docstring Style, Type Hints

Use [numpy style docstrings](https://docs.scipy.org/doc/numpy/docs/howto_document.html). All public code
(doesn't start with a `_`) should have a "full" docstring but private code (starts with a `_`) can have an
abbreviated docstring.


DASCore makes extensive use of Python 3's [type hints](https://docs.python.org/3/library/typing.html).
Use them to annotate any public functions/methods.

Here is an example:

```python
from typing import Optional, Union

import dascore
from dascore.constants import PatchType


# example public Patch function
@dascore.patch_function()
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
    >>> import dascore
    ... pa = dascore.get_example_patch()
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

## Paths

Prefer `pathlib.Path` to strings when working with paths. However, when dealing with many many files (e.g., indexers)
strings may be preferred for efficiency.

## Working with dataframes

Column names should be snake_cased whenever possible.

Always access columns with getitem and not getattr (ie use `df['column_name']` not `df.column_name`).

Prefer creating a new `DataFrame`/`Series` to modifying them inplace. Inplace modifications should require opting in
(usually through an `inplace` key word argument).
