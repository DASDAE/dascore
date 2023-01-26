"""
Create the html tables for parameters and such from dataframes.
"""
import hashlib
import inspect
import json
import os
import typing
from functools import cache
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

RENDER_FUNCS = {}
DOC_PATH = Path(__file__).absolute().parent.parent / "docs"
API_DOC_PATH = DOC_PATH / "api"
API_DOC_PATH.mkdir(exist_ok=True, parents=True)
TEMPLATE_PATH = DOC_PATH / "_templates"
GITHUB_PATH = "https://github.com"
GITHUB_REPOSITORY = "DASDAE/dascore"
GITHUB_REF = "/master"


@cache
def get_env():
    """Get the template environment."""
    template_path = Path(__file__).absolute().parent / "_templates"
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
    if text.endswith("s"):
        return text + "es"
    return text + "s"


def sha_256(path_or_str: Path | str) -> str:
    """Get the Sha256 hash of a file or a string."""
    if isinstance(path_or_str, Path) and Path(path_or_str).exists():
        with open(path_or_str) as fi:
            path_or_str = fi.read()
    return hashlib.sha256(str(path_or_str).encode('utf-8')).hexdigest()


def build_table(df: pd.DataFrame, caption=None):
    """
    An opinionated function to make a dataframe into html table.
    """
    df = df.drop_duplicates()
    template = get_template("table.html")
    columns = [x.capitalize() for x in df.columns]
    rows = df.to_records(index=False).tolist()
    out = template.render(columns=columns, rows=rows, caption=caption)
    return out


def is_it_subclass(obj, cls):
    """A more intelligent issubclass that doesn't through TypeErrors."""
    try:
        return issubclass(obj, cls)
    except TypeError:
        return False


def _deduplicate_list(seq):
    """Return a list with duplicate removed, preserves order."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def unpact_annotation(obj, data_dict, address_dict) -> str:
    """Convert an annotation to string with linking"""
    str_id = str(id(obj))
    str_rep = str(obj)
    # this is a basic type, just get its name.
    if is_it_subclass(obj, (str, int, float)):
        out = str_rep.replace("<class ", "").replace(">", "").replace("'", "")
    # this is a forward ref, strip out name and hope for the best ;)
    elif isinstance(obj, typing.ForwardRef):
        # todo: this needs to be more generic, but forward refs are hard!
        # just assume the bound string is resolvable
        key = str(obj).split("('")[-1].replace("')", "")
        name = key.split(".")[-1]
        return f"[{name}](`{key}`)"
    # Array-like thing from numpy #TODO improve this
    elif 'numpy.typing' in str_rep:
        return 'ArrayLike'
    # this is a union, unpack it
    elif str_rep.startswith("typing.Union") or str_rep.startswith('typing.Optional'):
        annos = [unpact_annotation(x, data_dict, address_dict) for x in obj.__args__]
        out = " | ".join(_deduplicate_list(annos))
    # This is a Dict, List, Tuple, etc. Just unpack.
    elif hasattr(obj, "__args__"):
        name = str_rep.replace("typing.", "").split("[")[0]
        sub = [unpact_annotation(x, data_dict, address_dict) for x in obj.__args__]
        out = f"{name}[{', '.join(sub)}]"
    # this is a typevar
    elif hasattr(obj, "__bound__"):
        out = unpact_annotation(obj.__bound__, data_dict, address_dict)
    # this is a class in dascore
    elif str_id in data_dict:
        data = data_dict[str_id]
        out = f"[{data['name']}](`{data['key']}`)"
    # this is a typing type, like Any
    elif str_rep.startswith("typing."):
        out = str_rep.split("typing.")[-1]
    # NoneType should just be None
    elif is_it_subclass(obj, type(None)):
        out = "None"
    # ...
    elif obj is Ellipsis:
        out = "..."
    # generic class from another library
    elif str_rep.startswith("<class"):
        out = str_rep.split(".")[-1].replace("'>", "")
    # return a self type.
    elif str_rep.endswith('.Self'):
        out = 'Self'
    # probably a literal, just give up here
    else:
        if isinstance(obj, str):  # make str look like str
            out = f"'{str(obj)}'"
        else:
            out = str(obj)
    return out.strip()


def build_signature(data, data_dict, address_dict):
    """Return html of signature block."""

    def get_annotation_str(param):
        """Get string of annotation."""
        if not param:
            return ""
        annotation_str = unpact_annotation(param, data_dict, address_dict)
        return f": {annotation_str}"

    def get_param_prefix(kind):
        """Get the prefix for a parameter (eg ** in **kwargs)"""
        if kind == inspect.Parameter.VAR_POSITIONAL:
            return '*'
        elif kind == inspect.Parameter.VAR_KEYWORD:
            return '**'
        return ''

    def get_params(sig, annotations):
        out = []
        for param, value in sig.parameters.items():
            prefix = get_param_prefix(value.kind)
            annotation_str = get_annotation_str(annotations.get(param))
            out.append(f"{prefix + param}{annotation_str}\n")
        return out

    def get_return_line(sig):
        annotation = sig.return_annotation
        if annotation is inspect._empty:
            return_str = ")"
        else:
            return_str = unpact_annotation(annotation, data_dict, address_dict)
            return_str = f")-> {return_str}"
        return return_str

    def get_sig_dict(data):
        """Create a dict of render-able signature stuff."""
        sig = data["signature"]
        annotations = typing.get_type_hints(data["object"])

        out = dict(
            params=get_params(sig, annotations),
            return_line=get_return_line(sig),
            name=data["name"],
        )
        return out

    # no need to do anything if entity is not callable.
    if not data["signature"]:
        return ""

    template = get_template("signature.html")
    sig_dict = get_sig_dict(data)
    out = template.render(**sig_dict)
    return out


class NumpyDocStrParser:
    """Class which parses/process docstrings written in numpy style."""

    heading_char = "##"

    def __init__(self, data):
        self._data = data

    def parse_sections(self, docstr):
        out = {}
        doc_lines = docstr.split("\n")
        dividers = (
                [0]
                + [
                    num
                    for num, x in enumerate(doc_lines)
                    if (set(x).issubset({" ", "-", "\t"}) and ("-" in x))
                ]
                + [len(doc_lines) + 2]
        )
        for start, stop in zip(dividers[:-1], dividers[1:]):
            # determine header or just use 'pre' for txt before headings
            header = "pre" if start == 0 else doc_lines[start - 1].strip()
            # skips the ----- line where applicable
            lstart = start if header == "pre" else start + 1
            out[header] = "\n".join(doc_lines[lstart: stop - 2]).strip()
        return out

    def style_parameters(self, param_str):
        """Style the parameters block."""
        lines = param_str.split("\n")
        # parameters dont have spaces at the start
        param_start = [num for num, x in enumerate(lines) if not x.startswith(" ")]
        param_start.append(len(lines))
        # parse parameters into (parameter): txt
        param_desc = []
        for ind_num, ind in enumerate(param_start[:-1]):
            key = lines[ind].strip()
            vals = [x.strip() for x in lines[ind + 1: param_start[ind_num + 1]]]
            param_desc.append((key, "\n".join(vals)))
        table = pd.DataFrame(param_desc, columns=["Parameter", "Description"])
        return build_table(table)

    def style_examples(self, example_str):
        """
        Styles example blocks.

        Example blocks ban be written in standard doc-test or quarto styles.
        """

        def style_docstest_string(docstr_split):
            """style the doctest string"""
            # need to add comment before lines which dont have comment
            # or dont start with >>> (these are outputs)
            out = []
            for line in docstr_split:
                # the [1:] for > and . lines is to strip off the first space
                if line.startswith(">"):
                    out.append(line.strip(">")[1:])
                elif line.startswith(".."):
                    out.append(line.strip(".")[1:])
                elif line.startswith("#") or not line.strip():
                    out.append(line)
                else:
                    out.append(f"# {line}")
            out_str = "\n".join(out)
            return "\n```{python}\n" + out_str + "\n```"

        # determine if this is doctest by looking for >>>
        line_split = example_str.split("\n")
        if any(x.startswith(">>") for x in example_str.split("\n")):
            return style_docstest_string(line_split)
        else:
            # just need to make python in brackets for quarto
            return example_str.replace("```python", "```{python}")

    def style_notes(self, notes_str):
        """styles notes section of str."""
        template = get_template("notes.md")
        return template.render(note_text=notes_str)

    def style_sections(self, raw_sections):
        """Apply styling to sections."""
        out = dict(raw_sections)
        lower2upper = {x.lower(): x for x in out}
        overlap = set(lower2upper) & set(self.stylers)
        for name in overlap:
            upper_name = lower2upper[name]
            out[upper_name] = self.stylers[name](self, out[upper_name])
        return out

    def __call__(self):
        """Process the docstring."""
        sections_raw = self.parse_sections(self._data["docstring"])
        sections_styled = self.style_sections(sections_raw)
        out = []
        for name, content in sections_styled.items():
            if name == "pre" or "note" in name.lower():
                out.append(content)
            else:
                out.append(f"\n{self.heading_char} {name}\n {content}")
        return "\n".join(out)

    # stylers is used to decide which style function to apply for various
    # sections.
    stylers = {
        "parameter": style_parameters,
        "parameters": style_parameters,
        "examples": style_examples,
        "example": style_examples,
        "notes": style_notes,
        "note": style_notes,
    }


class Render:
    """A class to render API docs for each entity."""

    # The order in which to render the tables.
    _table_order = ("parameter", "method", "attribute", "function", "class", "module")

    def __init__(self, data_dict, object_id, address_dict):
        self._data_dict = data_dict
        self._data = data_dict[object_id]
        self._address_dict = address_dict

    def has_subsection(self, name):
        """Return True if a module has a subsection."""
        possible_children = self._data.get(name, {})
        children = self.get_children_object_ids(possible_children)
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
        return build_table(df)

    def get_children_object_ids(self, ids):
        """Of an object id list, return the children of the current object."""
        key = self._data["key"]
        out = {
            x for x in ids if x in self._data_dict and key in self._data_dict[x]["key"]
        }
        return out

    def _get_github_source(self, data):
        """Get the github source url."""
        obj = data["object"]
        source_code, line_start = inspect.getsourcelines(obj)
        line_end = line_start + len(source_code)
        rel_path = data["path"].relative_to(data["base_path"])
        # grab repo stuff from environment (on GH actions) or defaults
        github_url = os.environ.get("GITHUB_SERVER_URL", GITHUB_PATH)
        owner_repo = os.environ.get("GITHUB_REPOSITORY", GITHUB_REPOSITORY)
        branch = os.environ.get("GITHUB_REF", GITHUB_REF).split("/")[-1]
        source_url = (
            f"{github_url}/{owner_repo}/blob/{branch}/{str(rel_path)}"
            f"#L{line_start + 1}-L{line_end}"
        )
        return source_url

    def _get_parent_source_block(self, data):
        """Create a parent block with a link."""
        parent = data['key'].removesuffix(f".{data['name']}")
        if parent:
            parent_str = f" of [{parent}](`{parent}`)"
        else:
            parent_str = ""
        origin_txt = f"*{self._data['data_type']}* {parent_str}"
        source_url = self._get_github_source(data)
        template = get_template("parent_source_block.html")
        out = template.render(origin_txt=origin_txt, source_url=source_url)
        return out

    def render_markdown(self, heading="##"):
        """Convert numpy docstrings to markdown with some html styling."""
        data = self._data
        docstr = NumpyDocStrParser(data)

        tables = [
            f"\n{heading} {_simple_plural(x).capitalize()}\n"
            f"{self.render_linked_table(x)}\n"
            for x in self._table_order
            if self.has_subsection(x)
        ]
        signature = build_signature(data, self._data_dict, self._address_dict)
        out = (
            f"# {self._data['name']}\n\n"
            f"{self._get_parent_source_block(data)}\n\n"
            f"{signature}\n"
            f"{docstr()}\n"
            f"{''.join(tables)}\n"
        )
        return out


def create_json_mapping(data_dict, obj_dict, api_path):
    """Create the mapping which links address to file path."""
    out = {}

    # add code paths
    for obj_id, data in data_dict.items():
        sub_dir = api_path / "/".join(data["key"].split(".")[:-1])
        path = api_path / sub_dir / f"{data['name']}.qmd"
        out[data["key"]] = f"/{path.relative_to(DOC_PATH)}"

    # add alias (eg import shortcuts/imports from other modules)
    for alias in set(obj_dict) - set(out):
        obj_id = obj_dict[alias]
        if obj_id not in data_dict:
            continue
        main_key = data_dict[obj_id]["key"]
        out[alias] = out[main_key]
    return out


def write_api_markdown(data_dict, api_path, address_dict, debug=False):
    """write all the markdown to disk."""
    files_to_delete = set(Path(api_path).rglob('*.qmd'))
    for obj_id, data in data_dict.items():
        # get path and ensure parents exist
        sub_dir = api_path / "/".join(data["key"].split(".")[:-1])
        path = api_path / sub_dir / f"{data['name']}.qmd"
        path.parent.mkdir(exist_ok=True, parents=True)
        # remove path from files to delete
        if path in files_to_delete:
            files_to_delete.remove(path)
        # dont render non-target file if debugging
        if debug and data['name'] != 'read':
            continue
        # render and write
        render = Render(data_dict, obj_id, address_dict)
        markdown = render.render_markdown()
        # check if file has changed, if not don't write
        if path.exists() and sha_256(path) == sha_256(markdown):
            continue
        path.write_text(markdown)
    # remove files that are no longer writen. This can happen when the code is
    # refactored or objects are deleted.
    for path in files_to_delete:
        path.unlink()


def render_project(data_dict, address_dict, api_path=API_DOC_PATH, debug=False):
    """Render the markdown files."""
    # Create and write the qmd files for each function/class/module
    write_api_markdown(data_dict, api_path, address_dict, debug=debug)
    # dump the json mapping to disk in doc folder
    path_mapping = create_json_mapping(data_dict, address_dict, api_path)
    with open(Path(api_path) / "cross_ref.json", "w") as fi:
        json.dump(path_mapping, fi, indent=2)
