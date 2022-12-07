"""
Create the html tables for parameters and such from dataframes.
"""
import inspect
import json
import os
import re
import shutil
import textwrap
from functools import cache
from pathlib import Path

import docstring_parser
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

RENDER_FUNCS = {}
DOC_PATH = Path(__file__).absolute().parent.parent / "docs"
API_DOC_PATH = DOC_PATH / "api"
API_DOC_PATH.mkdir(exist_ok=True, parents=True)
TEMPLATE_PATH = DOC_PATH / "_templates"
NUMPY_STYLE = docstring_parser.DocstringStyle.NUMPYDOC
GITHUB_PATH = 'https://github.com'
GITHUB_REPOSITORY = 'DASDAE/dascore'
GITHUB_REF = "/master"


@cache
def get_env():
    """Get the template environment."""
    template_path = Path(__file__).absolute().parent / '_templates'
    env = Environment(loader=FileSystemLoader(template_path))
    return env


@cache
def get_template(name):
    """Get the template for rendering tables."""
    env = get_env()
    template = env.get_template(name)
    return template


def _simple_plural(text):
    """return plural form of string"""
    if text.endswith('s'):
        return text + 'es'
    return text + 's'


def build_table(df, caption=None):
    """
    An opinionated function to make a dataframe into html table.
    """
    template = get_template('table.html')
    columns = [x.capitalize() for x in df.columns]
    rows = df.to_records(index=False).tolist()
    out = template.render(columns=columns, rows=rows, caption=caption)
    return out


def build_signature(data):
    """Return html of signature block."""

    def get_params(sig):
        params = [str(x).replace("'", "") for x in sig.parameters.values()]
        return params

    def get_return_line(sig):
        return_an = sig.return_annotation
        if return_an is inspect._empty:
            return_str = ')'
        else:
            return_str = f')-> {return_an}'
        return return_str

    def get_sig_dict(sig, data):
        """Create a dict of render-able signature stuff."""
        # TODO: add links
        out = dict(
            params=get_params(sig),
            return_line=get_return_line(sig),
            data_type=data['data_type'],
            name=data['name'],
        )
        return out

    # no need to do anything if entity is not callable.
    sig = data['signature']
    if not sig:
        return ''

    template = get_template('signature.html')
    sig_dict = get_sig_dict(sig, data)
    out = template.render(**sig_dict)
    return out


class DocStr:
    """Class which parases/process docstrings."""
    headings = "##"

    def __init__(self, data):
        self._data = data

    def _style_indents(self, docstr):
        """put indents in pre tags"""
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

    def __call__(self):
        """Process the docstring."""
        dedented_docstr = textwrap.dedent(self._data['docstring'])
        docstr = self._get_headings(dedented_docstr)
        # Preserve indentation in parameters
        docstr = self._style_indents(docstr)
        return docstr


class Render:
    """A class to render API docs for each entity."""

    # The order in which to render the tables.
    _table_order = (
        'parameter', 'method', 'attribute', 'function', 'class', 'module'
    )

    def __init__(self, data_dict, object_id):
        self._data_dict = data_dict
        self._data = data_dict[object_id]

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
            desc = sub_data["short_description"]
            out.append((linked_name, desc))
        df = pd.DataFrame(out, columns=["name", "description"])
        return build_table(df, _simple_plural(name))

    def get_children_object_ids(self, ids):
        """Of an object id list, return the children of the current object."""
        key = self._data["key"]
        out = {
            x for x in ids if x in self._data_dict and key in self._data_dict[x]["key"]
        }
        return out

    def _get_github_source(self, data):
        """Get the github source url."""
        obj = data['object']
        source_code, line_start = inspect.getsourcelines(obj)
        line_end = line_start + len(source_code)
        rel_path = data['path'].relative_to(data['base_path'])
        # grab repo stuff from environment (on GH actions) or defaults
        github_url = os.environ.get('GITHUB_SERVER_URL', GITHUB_PATH)
        owner_repo = os.environ.get('GITHUB_REPOSITORY', GITHUB_REPOSITORY)
        branch = os.environ.get("GITHUB_REF", GITHUB_REF).split('/')[-1]
        source_url = (
            f"{github_url}/{owner_repo}/blob/{branch}/{str(rel_path)}"
            f"#L{line_start+1}-L{line_end}"
        )
        return source_url

    def _get_parent_source_block(self, data):
        """Create a parent block with a link."""
        parent = data['key'].split(data['name'])[0].rstrip('.')
        if parent:
            parent_str = f" of [{parent}](`{parent}`)"
        else:
            parent_str = ""
        origin_txt = f"*{self._data['data_type']}* {parent_str}"
        source_url = self._get_github_source(data)
        template = get_template('parent_source_block.html')
        out = template.render(origin_txt=origin_txt, source_url=source_url)
        return out


    def render_markdown(self, heading='##'):
        """Convert numpy docstrings to markdown with some html styling."""
        data = self._data
        docstr = DocStr(data)

        tables = [
            f"\n{self.render_linked_table(x)}\n"
            for x in self._table_order
            if self.has_subsection(x)
        ]
        signature = build_signature(data)
        out = (
            f"# {self._data['name']}\n\n"
            f"{self._get_parent_source_block(data)}\n\n"
            f"{signature}\n"
            f"{docstr()}\n"
            f"{''.join(tables)}\n")
        return out


def create_json_mapping(data_dict, obj_dict, api_path):
    """Create the mapping which links address to file path."""
    out = {}

    # add code paths
    for obj_id, data in data_dict.items():
        sub_dir = api_path / "/".join(data["key"].split(".")[:-1])
        path = api_path / sub_dir / f"{data['name']}.qmd"
        out[data['key']] = f"/{path.relative_to(DOC_PATH)}"

    # add alias (eg import shortcuts/imports from other modules)
    for alias in set(obj_dict) - set(out):
        obj_id = obj_dict[alias]
        if obj_id not in data_dict:
            continue
        main_key = data_dict[obj_id]['key']
        out[alias] = out[main_key]
    return out


def write_api_markdown(data_dict, api_path, debug=False):
    """write all the markdown to disk."""
    for obj_id, data in data_dict.items():
        render = Render(data_dict, obj_id)
        # template = get_template(data["data_type"])
        markdown = render.render_markdown()
        sub_dir = api_path / "/".join(data["key"].split(".")[:-1])
        # Ensure expected path exists
        path = api_path / sub_dir / f"{data['name']}.qmd"
        path.parent.mkdir(exist_ok=True, parents=True)
        if debug and not path.name.endswith('iresample.qmd'):
            continue
        path.write_text(markdown)


def render_project(data_dict, obj_dict, api_path=API_DOC_PATH, debug=False):
    """Render the markdown files."""
    # clear old contents from API directory
    if api_path.exists() and api_path.is_dir():
        shutil.rmtree(api_path)
    # write each of the markdown files
    write_api_markdown(data_dict, api_path, debug=debug)
    # dump the json mapping to disk in doc folder
    path_mapping = create_json_mapping(data_dict, obj_dict, api_path)
    with open(Path(api_path) / 'cross_ref.json', 'w') as fi:
        json.dump(path_mapping, fi, indent=2)
