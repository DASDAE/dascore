"""Utilities for documentation."""

from __future__ import annotations

import inspect
import os
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

import dascore as dc


def format_dtypes(dtype_dict: dict[str, Any]) -> str:
    """
    Convert dict to string for printing.

    Convert a dictionary of {name: type} to a string printable format for
    displaying in docstrings.

    Parameters
    ----------
    dtype_dict
        A dict of columns (keys) and values (dtypes).

    Returns
    -------
    A string formatted for
    [compose_docstring](`dascore.utils.docs.compose_docstring`)
    """

    def _format_cls_str(cls):
        """Format str representation of classes to be nicer on the eyes."""
        cls_str = str(cls)
        return cls_str.replace("<class ", "").replace(">", "").replace("'", "")

    str_list = [f"{x}: {_format_cls_str(y)}" for x, y in dtype_dict.items()]
    out = "\n".join(str_list)
    return out


def compose_docstring(**kwargs: str | Sequence[str]):
    """
    Decorator for composing docstrings.

    This allows components of docstrings which are often repeated to be
    specified in a single place. Values provided to this function should
    have string keys and string or list values. Keys are found in curly
    brackets in the wrapped functions docstring and their values are
    substituted with proper indentation.

    Notes
    -----
    A function's docstring can be accessed via the `__docs__` attribute.

    Examples
    --------
    ```{python}
    from dascore.utils.docs import compose_docstring

    @compose_docstring(some_value='10')
    def example_function():
        '''
        Some useful description

        The following line will be the string '10':
        {some_value}
        '''
    ```
    """

    def _wrap(func):
        docstring = func.__doc__
        assert isinstance(docstring, str)
        used_keys = set()
        # iterate each provided value and look for it in the docstring
        for key, value in kwargs.items():
            value = value if isinstance(value, str) else "\n".join(value)
            # strip out first line if needed
            value = textwrap.dedent(value).lstrip()
            search_value = f"{{{key}}}"
            # find all lines that match values
            lines = [x for x in docstring.split("\n") if search_value in x]
            if lines:
                used_keys.add(key)
            for line in lines:
                # determine number of spaces used before matching character
                spaces = line.split(search_value)[0]
                # ensure only spaces precede search value
                assert set(spaces) == {" "} or not len(spaces)
                new = textwrap.indent(textwrap.dedent(value), spaces)
                docstring = docstring.replace(line, new)
        if kwargs and not used_keys:
            msg = (
                f"compose_docstring did not replace any placeholders on "
                f"{func.__name__}."
            )
            raise ValueError(msg)
        unused_keys = sorted(set(kwargs) - used_keys)
        if unused_keys:
            unused_str = ", ".join(unused_keys)
            msg = (
                f"compose_docstring received unused keys for {func.__name__}: "
                f"{unused_str}."
            )
            raise ValueError(msg)

        func.__doc__ = docstring
        return func

    return _wrap


def get_plugin_table() -> pd.DataFrame:
    """
    Return a DataFrame of registered third-party plugins.

    Reads all CSV files in the plugin registry, deduplicates on namespace,
    and sorts alphabetically. Columns are ``namespace``, ``package_name``,
    and ``package_url``.
    """
    from dascore.utils.namespace import _PLUGIN_REGISTRY_DIR

    frames = [pd.read_csv(p) for p in sorted(_PLUGIN_REGISTRY_DIR.glob("*.csv"))]
    if not frames:
        return pd.DataFrame(columns=["namespace", "package_name", "package_url"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="namespace")
        .sort_values("namespace")[["namespace", "package_name", "package_url"]]
        .reset_index(drop=True)
    )


def objs_to_doc_df(doc_dict, cross_reference=True):
    """
    Convert a dictionary of documentable entities to a pandas dataframe.
    """
    out = {}
    base = Path(dc.__file__).parent.parent
    for key, obj in doc_dict.items():
        if cross_reference:
            path = Path(inspect.getfile(obj)).relative_to(base)
            name = obj.__name__
            address = str(path).replace(".py", "").replace(os.sep, ".")
            key = f"[`{key}`](`{address + '.' + name}`)"
        doc = str(getattr(obj, "__func__", obj).__doc__).strip()
        out[key] = doc.splitlines()[0]
    df = pd.Series(out).to_frame().reset_index()
    df.columns = ["Name", "Description"]
    return df
