"""Utilities for documentation."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
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


def get_docstring(obj) -> str:
    """
    Return an object's docstring.

    `__doc__` is `str | None` on anything, but the callers here compose a
    documented function's docstring into their own, so a missing one means
    the docstring was deleted, not that there is nothing to compose.
    """
    docstring = obj.__doc__
    assert docstring is not None, f"{obj} has no docstring to compose."
    return docstring


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
    from dascore.utils.namespace import _PLUGIN_REGISTRY_DIR  # noqa: PLC0415

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


# ---------------------------------------------------------------------------
# Rendering a module's public API onto a single documentation page.
#
# Modules of small helpers get one page each rather than a page per function:
# the summary and signature stay visible while the parameter table, examples
# and notes sit in a collapsed callout. Every entry keeps an explicit anchor so
# cross references can link straight to it.
# ---------------------------------------------------------------------------


def get_doc_anchor(dotted: str) -> str:
    """Return the html anchor used for a dotted path on a rendered API page."""
    return dotted.replace(".", "-").replace("_", "-").lower()


def _fmt_annotation(anno) -> str:
    """Render an annotation the way it was written in the source."""
    if isinstance(anno, str):
        return anno
    return getattr(anno, "__name__", None) or str(anno).replace("typing.", "")


def _fmt_default(value) -> str:
    """Render a default value compactly (classes by name, not repr)."""
    if inspect.isclass(value) or inspect.isfunction(value):
        return value.__name__
    return repr(value)


def _signature(obj) -> str:
    """Render a signature without the quoting artifacts of string annotations."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return getattr(obj, "__name__", "?") + "(...)"
    parts, seen_kw_only = [], False
    for p in sig.parameters.values():
        if p.kind is p.KEYWORD_ONLY and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        text = p.name
        if p.kind is p.VAR_POSITIONAL:
            text = f"*{text}"
        elif p.kind is p.VAR_KEYWORD:
            text = f"**{text}"
        if p.annotation is not p.empty:
            text += f": {_fmt_annotation(p.annotation)}"
        if p.default is not p.empty:
            text += f" = {_fmt_default(p.default)}"
        parts.append(text)
    ret = ""
    if sig.return_annotation is not sig.empty:
        ret = f" -> {_fmt_annotation(sig.return_annotation)}"
    return f"{obj.__name__}({', '.join(parts)}){ret}"


def _param_types(obj) -> dict[str, str]:
    """Map parameter name to its annotation, for the parameter table."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return {}
    return {
        p.name: _fmt_annotation(p.annotation)
        for p in sig.parameters.values()
        if p.annotation is not p.empty
    }


def _parse(doc: str):
    """Parse a numpydoc docstring into griffe sections."""
    from griffe import Docstring  # noqa: PLC0415

    # Griffe logs a warning for every parameter documented without a type,
    # which is DASCore's house style (types come from the annotations).
    logging.getLogger("griffe").setLevel(logging.ERROR)
    return Docstring(doc, parser="numpy").parse("numpy")


def _render_sections(sections, types: dict[str, str]) -> tuple[str, list[str]]:
    """Return the summary line and the collapsible body blocks."""
    summary, blocks = "", []
    for sec in sections:
        kind = sec.kind.value
        if kind == "text":
            text = str(sec.value).strip()
            if not summary:
                first, _, rest = text.partition("\n\n")
                summary = " ".join(first.split())
                if rest.strip():
                    blocks.append(rest.strip())
            else:
                blocks.append(text)
        elif kind == "parameters":
            rows = ["| Parameter | Type | Description |", "|---|---|---|"]
            for p in sec.value:
                anno = types.get(p.name) or (
                    "" if p.annotation is None else str(p.annotation)
                )
                anno = f"`{anno}`" if anno else ""
                desc = " ".join(str(p.description).split())
                rows.append(f"| `{p.name}` | {anno} | {desc} |")
            blocks.append("\n".join(rows))
        elif kind in {"returns", "yields"}:
            items = []
            for r in sec.value:
                anno = "" if r.annotation is None else f"`{r.annotation}` — "
                items.append(f"{anno}{' '.join(str(r.description).split())}")
            blocks.append(f"**{kind.title()}:** " + "; ".join(items))
        elif kind == "raises":
            items = [
                f"`{r.annotation}` — {' '.join(str(r.description).split())}"
                for r in sec.value
            ]
            blocks.append("**Raises:** " + "; ".join(items))
        elif kind == "examples":
            for _, text in sec.value:
                blocks.append("```python\n" + str(text).strip() + "\n```")
        elif kind == "admonition":
            body = getattr(sec.value, "description", sec.value)
            blocks.append(f"**{sec.title or 'Note'}:** {str(body).strip()}")
        else:
            blocks.append(str(sec.value).strip())
    return summary, blocks


def iter_public(module_name: str):
    """Yield (name, object) for public objects defined in the module."""
    mod = importlib.import_module(module_name)
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", "") != module_name:
            continue
        yield name, obj


def render_module_api(module_name: str) -> str:
    """Render every public object defined in a module as markdown."""
    out: list[str] = []
    for name, obj in iter_public(module_name):
        dotted = f"{module_name}.{name}"
        doc = inspect.getdoc(obj) or ""
        summary, blocks = (
            _render_sections(_parse(doc), _param_types(obj)) if doc else ("", [])
        )
        out.append(f"#### {name} {{#{get_doc_anchor(dotted)}}}\n")
        out.append(f"```python\n{_signature(obj)}\n```\n")
        if summary:
            out.append(f"{summary}\n")
        if blocks:
            out.append(
                '::: {.callout-note collapse="true" appearance="simple" '
                'title="Details"}\n'
            )
            out.extend(b + "\n" for b in blocks)
            out.append(":::\n")
    return "\n".join(out)


def render_package_api(package_name: str, skip_empty: bool = True) -> str:
    """
    Render the public API of every module in a package as markdown.

    Each module becomes a level-two heading and each object a level-three
    heading below it, so the page table of contents lists modules while every
    object still has an anchor to link to.

    Parameters
    ----------
    package_name
        The importable name of the package, e.g. "dascore.utils".
    skip_empty
        If True, omit modules which define no public objects.
    """
    package = importlib.import_module(package_name)
    out = []
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            body = render_module_api(info.name)
        except ImportError:  # an optional dependency the doc env lacks
            continue
        if skip_empty and not body.strip():
            continue
        out.append(f"### {info.name} {{#{get_doc_anchor(info.name)}}}\n")
        module_doc = inspect.getdoc(importlib.import_module(info.name)) or ""
        if module_doc:
            out.append(" ".join(module_doc.split("\n\n")[0].split()) + "\n")
        out.append(body)
    return "\n".join(out)
