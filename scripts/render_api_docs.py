"""
Script to make the API docs for dascore.
"""
from __future__ import annotations

import inspect
import json
import re
import textwrap
import numpy as np
from collections import defaultdict
from functools import cache
from pathlib import Path
from types import MethodType, ModuleType, FunctionType
from typing import Literal

import docstring_parser
import jinja2
import pandas as pd
from pretty_html_table import build_table

import dascore

RENDER_FUNCS = {}
DOC_PATH = Path(__file__).absolute().parent.parent / "docs"
API_DOC_PATH = DOC_PATH / "api"
API_DOC_PATH.mkdir(exist_ok=True, parents=True)
TEMPLATE_PATH = DOC_PATH / "_templates"
NUMPY_STYLE = docstring_parser.DocstringStyle.NUMPYDOC


class Render:
    """A class to render html into the qmd file."""

    # The order in which to render the tables.
    _table_order = (
        'parameter', 'method', 'attribute', 'function', 'class', 'module'
    )

    def __init__(self, data_dict, object_id):
        self._data_dict = data_dict
        self._data = data_dict[object_id]

    def _render_table(self, df):
        return build_table(df, "blue_light", font_size="large")

    def has_subsection(self, name):
        """Return True if a module has a subsection."""
        children = self.get_children_object_ids(self._data.get(name, {}))
        return bool(children)

    @property
    def is_callable(self):
        """Return True if object is callable (has a signature)"""
        return self._data['signature'] is not None

    def make_signature(self):
        """Create a nice looking signature."""

        def _format_params(sig):
            """format the parameter."""
            out = []
            # add parameters
            for name, param in sig.parameters.items():
                if name == 'self' or param.annotation is inspect._empty:
                    out.append(f"{name},")
                    continue
                out.append(f"{name}: {param.annotation},")
            return out

        sig = self.signature

        joined = '\n    '.join(_format_params(sig))
        out = f"{self.name}(\n    {joined}\n)"
        # add return
        if sig.return_annotation is not inspect._empty:
            out += f" -> {sig.return_annotation}"

        return out

    def render_linked_table(self, name):
        """Render a table for a specific name."""
        out = []
        for oid in self._data[name]:
            if oid not in self._data_dict:
                continue
            sub_data = self._data_dict[oid]
            linked_name = f"[{sub_data['name']}](`{sub_data['key']}`)"
            desc = sub_data["short_description"]
            out.append((linked_name, desc))
        df = pd.DataFrame(out, columns=["name", "description"])
        return self._render_table(df)

    def __getattr__(self, item):
        return self._data[item]

    def get_children_object_ids(self, ids):
        """Of an object id list, return the children of the current object."""
        key = self._data["key"]
        out = {
            x for x in ids if x in self._data_dict and key in self._data_dict[x]["key"]
        }
        return out

    def get_signature_html(self):
        """Make the fancy html signature block."""
        return ''

    def _style_indents(self, docstr):
        """Replace indents with quarto styled fencing."""
        regex = r"([ \t]{2,})(.*?)"
        lines = docstr.split('\n')
        matches = [bool(re.match(regex, x)) for x in lines]
        # no indents, bail out
        if not any(matches):
            return docstr
        # otherwise iterate and format.
        str_array = np.array(lines)
        match_ar = np.array(matches, dtype=bool)
        groups = np.cumsum(~match_ar) * match_ar
        for gnum in np.unique(groups):
            if not gnum:
                continue
            mask = groups == gnum

            current_lines = '\n'.join(str_array[mask])
            new_lines = f"<pre>{current_lines}</pre>"
            docstr = docstr.replace(current_lines, new_lines)
        return docstr

    def _get_headings(self, docstr, heading='##'):
        """Replace numpy style headings with markdown ones."""
        # this regex identifies a heading
        regex = r"(\r\n|\r|\n)(.*?)(\r\n|\r|\n)(-+)(\s*)(\r\n|\r|\n)"
        for match in re.findall(regex, docstr):
            new_title = f"\n{heading} {match[1]}\n"
            docstr = docstr.replace(''.join(match), new_title)
        return docstr

    def render_markdown(self, heading='##'):
        """Convert numpy docstrings to markdown with fancy signature block."""
        data = self._data
        dedented_docstr = textwrap.dedent(data['docstring'])
        docstr = self._get_headings(dedented_docstr, heading)
        # Preserve indentation
        docstr = self._style_indents(docstr)
        tables = [
            f"\n# {_simple_plural(x)}\n{self.render_linked_table(x)}\n"
            for x in self._table_order
            if self.has_subsection(x)
        ]
        signature = self.get_signature_html()
        out = f"# {self._data['name']}\n{signature}\n{docstr}\n{''.join(tables)}\n"
        return out

    # def


def _simple_plural(text):
    """return plural form of string"""
    if text.endswith('s'):
        return text + 'es'
    return text + 's'


@cache
def get_template(name, template_path=TEMPLATE_PATH):
    """Load a template."""
    path = template_path / f"{name}.qmd"
    assert path.exists()
    return jinja2.Template(str(path.read_text()))


def unwrap_obj(obj):
    """Unwrap a decorated object. """
    while getattr(obj, '__wrapped__', None) is not None:
        obj = obj.__wrapped__
    return obj


def get_type(obj, parent_is_class=False) -> None | Literal["module", "function", "method", "class"]:
    """Return a string of the type of object."""
    obj = unwrap_obj(obj)
    if isinstance(obj, ModuleType):
        return "module"
    elif isinstance(obj, MethodType):
        return "method"
    elif isinstance(obj, FunctionType):
        # since this is a class not instance we need this check.
        if parent_is_class:
            return "method"
        else:
            return "function"
    elif isinstance(obj, type):
        return "class"
    return None


def parse_project(obj, key=None):
    """Parse the project create dict of data and data_type"""
    key = key or getattr(obj, "__name__", None)
    base_path = Path(_get_file_path(obj)).parent.parent
    data_dict = {}
    traverse(obj, key, data_dict, str(base_path))
    return data_dict


def _get_file_path(obj):
    """Try to get the file of a python object."""
    obj = unwrap_obj(obj)
    try:
        path = inspect.getfile(obj)
    except TypeError:
        path = ""
    return Path(path)


def get_base_address(path, base_path):
    """
    Get the base address inherent in the path.

    For example, dascore/core/patch.py should return

    dascore.core.patch
    """
    try:
        out = Path(path).relative_to(Path(base_path))
    except ValueError:
        return ""
    new = str(out).replace("/__init__.py", "").replace(".py", "")
    return new.replace("/", ".")


def get_data(obj, key, base_address):
    """Get data from object. """

    def extract_data(obj):
        """
        Make lists of attributes, methods, etc.

        These can be feed to render templates.
        """

        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            sig = None

        docstr = inspect.getdoc(obj)
        doc = docstring_parser.parse(docstr, style=NUMPY_STYLE)
        dtype = get_type(obj)
        data = defaultdict(list)
        data["long_description"] = doc.long_description
        data["short_description"] = doc.short_description
        data['docstring'] = docstr
        data['signature'] = sig
        data["data_type"] = dtype
        for i in sorted(dir(obj)):
            if i.startswith("_"):
                continue
            sub_obj = getattr(obj, i)
            sub_dtype = get_type(getattr(obj, i), dtype=='class')
            data[sub_dtype].append(str(id(sub_obj)))

        return data

    data = extract_data(obj)
    data["key"] = key
    data["name"] = key.split(".")[-1]
    data["base_address"] = base_address
    return data


def traverse(obj, key, data_dict, base_path):
    """Traverse the tree and write out markdown."""
    obj = unwrap_obj(obj)
    obj_id = str(id(obj))
    path = _get_file_path(obj)
    # we are looking at an alias apart from where the file is defined.
    # skip so that only the source def gets picked up
    base_address = get_base_address(path, base_path)
    if not base_address or base_address not in key:
        return
    if obj_id in data_dict:
        return
    # get data about object, store in data_dict
    data_dict[obj_id] = get_data(obj, key, base_address)
    # recurse members
    for (member_name, member) in inspect.getmembers(obj):
        if member_name.startswith("_"):
            continue
        new_key = ".".join([key, member_name])
        traverse(member, new_key, data_dict, base_path)


def render_project(data_dict, obj_dict, api_path=API_DOC_PATH):
    """Render the markdown files."""
    # key: md_path
    path_mapping = {}

    for obj_id, data in data_dict.items():
        render = Render(data_dict, obj_id)
        # template = get_template(data["data_type"])
        markdown = render.render_markdown()
        sub_dir = api_path / "/".join(data["key"].split(".")[:-1])
        # Ensure expected path exists
        path = api_path / sub_dir / f"{data['name']}.qmd"
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(markdown)
        path_mapping[data['key']] = f"/{path.relative_to(DOC_PATH)}"

    # add alias (eg import shortcuts)
    for alias in set(obj_dict) - set(path_mapping):
        obj_id = obj_dict[alias]
        if obj_id not in data_dict:
            continue
        main_key = data_dict[obj_id]['key']
        path_mapping[alias] = path_mapping[main_key]

    # dump the json mapping to disk in doc folder
    with open(Path(api_path) / 'cross_ref.json', 'w') as fi:
        json.dump(path_mapping, fi, indent=2)


def get_alias_mapping(module, key=None):
    """Return a dict of {object_path: id} to construct cross refs."""

    def traverse_simple(obj, key, data_dict, base_path):
        """Traverse the tree and write out markdown."""
        obj = unwrap_obj(obj)
        obj_id = str(id(obj))
        path = _get_file_path(obj)
        if not get_base_address(path, base_path):
            return
        data_dict[key] = obj_id
        for (member_name, member) in inspect.getmembers(obj):
            if member_name.startswith("_"):
                continue
            new_key = ".".join([key, member_name])
            if new_key.split('.')[-1] in key:
                continue
            traverse_simple(member, new_key, data_dict, base_path)

    data_dict = {}
    key = key or getattr(module, "__name__", None)
    base_path = Path(_get_file_path(module)).parent.parent
    traverse_simple(module, key, data_dict, base_path)
    return data_dict


if __name__ == "__main__":
    data_dict = parse_project(dascore)
    obj_dict = get_alias_mapping(dascore)
    render_project(data_dict, obj_dict)
