"""
Script to make the API docs for dascore.
"""
from __future__ import annotations

import inspect
from inspect import signature
from typing import get_type_hints
from pathlib import Path
from typing import Literal
from types import MethodType, ModuleType, FunctionType
from functools import cache
from collections import defaultdict

import pandas as pd
from pretty_html_table import build_table

import dascore
import docstring_parser
import jinja2
from dascore.utils.misc import register_func

RENDER_FUNCS = {}
DOC_PATH = Path(__file__).absolute().parent.parent / 'docs'
API_DOC_PATH = DOC_PATH / 'api'
API_DOC_PATH.mkdir(exist_ok=True, parents=True)
TEMPLATE_PATH = DOC_PATH / '_templates'
NUMPY_STYLE = docstring_parser.DocstringStyle.NUMPYDOC


class Render:
    """A class to render html into the qmd file."""
    def __init__(self, data_dict, object_id):
        self._data_dict = data_dict
        self._data = data_dict[object_id]

    def _render_table(self, df):
        width_dict = ['100px', 'auto',]

        return build_table(df, "blue_light", font_size='large')

    def has_subsection(self, name):
        """Return True if a module has a subsection."""
        children = self.get_children_object_ids(self._data.get(name, {}))
        return bool(children)

    def render_linked_table(self, name):
        """Render a table for a specific name."""
        out = []
        for oid in self._data[name]:
            if oid not in self._data_dict:
                continue
            sub_data = self._data_dict[oid]
            linked_name = f"[{sub_data['name']}](`{sub_data['key']}`)"
            desc = sub_data['short_description']
            out.append((linked_name, desc))
        df = pd.DataFrame(out, columns=['name', 'description'])
        return self._render_table(df)

    def __getattr__(self, item):
        return self._data[item]

    def get_children_object_ids(self, ids):
        """Of an object id list, return the children of the current object."""
        key = self._data['key']
        out = {
            x for x in ids
            if x in self._data_dict and key in self._data_dict[x]['key']
        }
        return out

    # def



@cache
def get_template(name, template_path=TEMPLATE_PATH):
    """Load a template."""
    path = template_path / f"{name}.qmd"
    assert path.exists()
    return jinja2.Template(str(path.read_text()))


def get_styled_signature(obj):
    """Style the signature for inclusion in markdown file."""


def extract_data(obj):
    """
    Make lists of attributes, methods, etc.

    These can be feed to render templates.
    """

    def add_parameters(obj, doc, sig):
        """Add parameters to doc list."""
        if not sig:
            return
        return doc.params

    def add_signature(obj, doc, sig):
        """Add the object signature."""
        if not sig:
            return

    def add_attributes(obj, doc):
        """Add the object attributes."""
        return None

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        sig = None

    doc_str = inspect.getdoc(obj)
    doc = docstring_parser.parse(doc_str, style=NUMPY_STYLE)
    dtype = get_type(obj)

    data = defaultdict(list)
    data['long_description'] = doc.long_description
    data['short_description'] = doc.short_description
    data['signature'] = add_signature(obj, doc, sig)
    data['parameters'] = add_parameters(obj, doc, sig)
    data['attributes'] = add_attributes(obj, doc)
    data['data_type'] = dtype
    for i in sorted(dir(obj)):
        if i.startswith('_'):
            continue
        sub_obj = getattr(obj, i)
        sub_dtype = get_type(getattr(obj, i))
        data[sub_dtype].append(str(id(sub_obj)))

    return data




def is_private(name):
    """Return True if the object is private."""


def get_type(obj) -> None | Literal['module', 'function', 'method', 'class']:
    """Return a string of the type of object."""
    if isinstance(obj, ModuleType):
        return 'module'
    elif isinstance(obj, FunctionType):
        return 'function'
    elif isinstance(obj, MethodType):
        return 'method'
    elif isinstance(obj, type):
        return 'class'
    return None


def parse_project(obj, key=None):
    """Parse the project create dict of data and data_type"""
    key = key or getattr(obj, '__name__', None)
    base_path = Path(_get_file_path(obj)).parent.parent
    data_dict = {}
    traverse(obj, key, data_dict, str(base_path))
    return data_dict


def _get_file_path(obj):
    """Try to get the file of a python object."""
    try:
        path = inspect.getfile(obj)
    except TypeError:
        path = ''
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
        return ''
    new = str(out).replace('/__init__.py', '').replace('.py', '')
    return new.replace('/', '.')


def traverse(obj, key, data_dict, base_path):
    """Traverse the tree and write out markdown."""
    obj_id = str(id(obj))
    path = _get_file_path(obj)
    # we are looking at an alias apart from where the file is defined.
    # skip so that only the source def gets picked up
    base_address = get_base_address(path, base_path)
    if not base_address or base_address not in key or obj_id in data_dict:
        return
    data_dict[obj_id] = extract_data(obj)
    data_dict[obj_id]['key'] = key
    data_dict[obj_id]['name'] = key.split('.')[-1]
    data_dict[obj_id]['base_address'] = base_address
    for (member_name, member) in inspect.getmembers(obj):
        if member_name.startswith('_'):
            continue
        new_key = '.'.join([key, member_name])
        traverse(member, new_key, data_dict, base_path)


def render_project(data_dict, api_path=API_DOC_PATH):
    """Render the markdown files."""

    for key, data in data_dict.items():
        render = Render(data_dict, key)

        template = get_template(data['data_type'])
        markdown = template.render(rend=render)

        sub_dir = api_path / '/'.join(data['key'].split('.')[:-1])
        # Ensure expected path exists
        path = api_path / sub_dir / f"{data['name']}.qmd"
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(markdown)



if __name__ == "__main__":
    data_dict  = parse_project(dascore)
    render_project(data_dict)
    breakpoint()
