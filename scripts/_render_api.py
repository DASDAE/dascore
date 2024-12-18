"""Create the html tables for parameters and such from dataframes."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import typing
from collections import defaultdict
from functools import cache
from itertools import pairwise
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
    """Return plural form of string."""
    if text.endswith("s"):
        return text + "es"
    return text + "s"


def sha_256(path_or_str: Path | str) -> str:
    """Get the Sha256 hash of a file or a string."""
    if isinstance(path_or_str, Path) and Path(path_or_str).exists():
        with open(path_or_str) as fi:
            path_or_str = fi.read()
    return hashlib.sha256(str(path_or_str).encode("utf-8")).hexdigest()


def build_table(df: pd.DataFrame, caption=None):
    """An opinionated function to make a dataframe into html table."""
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
    """Convert an annotation to string with linking."""
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
        # dc. is the same as dascore.
        if key.startswith("dc."):
            key = f"dascore.{key[3:]}"
        return f"[{name}](`{key}`)"
    # Array-like thing from numpy #TODO improve this
    elif "numpy.typing" in str_rep:
        return "ArrayLike"
    # this is a union, unpack it
    elif str_rep.startswith("typing.Union") or str_rep.startswith("typing.Optional"):
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
    elif str_rep.endswith(".Self"):
        out = "Self"
    # probably a literal, just give up here
    else:
        if isinstance(obj, str):  # make str look like str
            out = f"'{obj!s}'"
        else:
            out = str(obj)
    return out.strip()


def build_signature(data, data_dict, address_dict):
    """Return html of signature block."""
    sentinel = object()  # to know missing values

    def get_annotation_str(param):
        """Get string of annotation."""
        if not param:
            return ""
        annotation_str = unpact_annotation(param, data_dict, address_dict)
        return f": {annotation_str}"

    def get_param_prefix(kind):
        """Get the prefix for a parameter (eg ** in **kwargs)."""
        if kind == inspect.Parameter.VAR_POSITIONAL:
            return "*"
        elif kind == inspect.Parameter.VAR_KEYWORD:
            return "**"
        return ""

    def get_default_value(param, sig):
        """Get the default value for a parameter."""
        default = sig.parameters[param].default
        if default is inspect._empty:
            return sentinel
        if default == "":
            default = '""'
        return str(default)

    def get_params(sig, annotations):
        out = []
        for param, value in sig.parameters.items():
            prefix = get_param_prefix(value.kind)
            annotation_str = get_annotation_str(annotations.get(param))
            return_str = get_default_value(param, sig)
            param_str = f"{prefix + param}{annotation_str}\n"
            if return_str is not sentinel:
                param_str += f" = {return_str}"
            out.append(param_str)
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
        """Parse the sections of the docstring."""
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
        for start, stop in pairwise(dividers):
            # determine header or just use 'pre' for txt before headings
            header = "pre" if start == 0 else doc_lines[start - 1].strip()
            # skips the ----- line where applicable
            lstart = start if header == "pre" else start + 1
            out[header] = "\n".join(doc_lines[lstart : stop - 2]).strip()
        return out

    def style_parameters(self, param_str):
        """
        Style the parameters block.
        """
        lines = param_str.split("\n")
        # parameters dont have spaces at the start
        param_start = [num for num, x in enumerate(lines) if not x.startswith(" ")]
        param_start.append(len(lines))
        # parse parameters into (parameter): txt
        param_desc = []
        for ind_num, ind in enumerate(param_start[:-1]):
            key = lines[ind].strip()
            # get the number of indents (usually 4)
            in_char = (len(lines[1]) - len(lines[1].lstrip())) * " "
            desc_lines = lines[ind + 1 : param_start[ind_num + 1]]
            vals = [
                # strip out the first indentation line
                (x[len(in_char) :] if x.startswith(in_char) and in_char else x)
                for x in desc_lines
            ]
            # breakpoint()
            param_desc.append((key, "<br>".join(vals)))
        table = pd.DataFrame(param_desc, columns=["Parameter", "Description"])
        return build_table(table)

    def style_examples(self, example_str):
        """
        Styles example blocks.

        Example blocks can be written in standard doc-test or quarto styles.
        """
        return to_quarto_code(example_str)

    def style_notes(self, notes_str):
        """Styles notes section of str."""
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
                out.append(f"\n{self.heading_char} {name}\n{content}")
        return "\n".join(out)

    # stylers are used to decide which style function to apply for various
    # sections.
    stylers = {  # noqa
        "parameter": style_parameters,
        "parameters": style_parameters,
        "attributes": style_parameters,
        "examples": style_examples,
        "example": style_examples,
        "notes": style_notes,
        "note": style_notes,
    }


def to_quarto_code(code_lines):
    """Class for parsing code (eg examples in docstrings) to quarto output."""
    option_chars = "#|"

    def _get_blocks(code_lines):
        """Get code blocks. Code blocks are divided by titles "###"."""
        code_blocks = defaultdict(list)
        current_key = ""
        for line in code_lines:
            if line.startswith("###"):
                current_key = line.replace("###", "").lstrip()
            else:
                code_blocks[current_key].append(line)
        return code_blocks

    def _is_doctest_line(line):
        strip = line.lstrip()
        doc_str_line = strip.startswith(">>>") or strip.startswith("...")
        return doc_str_line

    def _strip_code(code_str):
        """Strip off spaces and the doctest stuff (..., >>>, etc)."""
        out = []
        code_lines = code_str.splitlines()
        # first determine if this is a docstring or not
        is_docstring = False
        for line in code_lines:
            strip = line.lstrip()
            if strip.startswith(">>>"):
                is_docstring = True
        # first determine how much to strip from each line.
        start_index = 999
        for line in code_lines:
            striped = line.lstrip().lstrip(">").lstrip(".").lstrip()
            if not len(striped):
                continue
            if is_docstring and not _is_doctest_line(line):
                continue
            strip_len = len(line) - len(striped)
            start_index = min([start_index, strip_len])
        # then do the stripping
        for line in code_lines:
            if (is_docstring and _is_doctest_line(line)) or not is_docstring:
                out.append(line[start_index:])
        return out

    def _get_options(code_lines):
        """Get all quarto options used."""
        options = []
        code_segments = []
        # first get min space to strip.
        for line in code_lines:
            if line.startswith(option_chars):
                options.append(line)
            # Strip out the python code specifier.
            elif line.startswith("```{"):
                continue
            else:
                code_segments.append(line)
        return code_segments, options

    def _strip_blocks(code_blocks):
        """Strip empty lines from start and end of list of codes."""
        while len(code_blocks) and not code_blocks[0].strip():
            code_blocks = code_blocks[1:]
        while len(code_blocks) and not code_blocks[-1].strip():
            code_blocks = code_blocks[:-1]
        return code_blocks

    def _to_quarto(options, blocks):
        """Convert output to quarto string."""
        out = []
        for section_name, code_blocks in blocks.items():
            title = [] if not section_name else [f"### {section_name}"]
            stripped_block = _strip_blocks(code_blocks)
            if not stripped_block:
                continue
            new_code = [*title, "```{python}", *options, *stripped_block, "```"]
            out.append("\n".join(new_code).lstrip())
        return "\n".join(out)

    code_lines_stripped = _strip_code(code_lines)
    code_lines_no_options, options = _get_options(code_lines_stripped)
    blocks = _get_blocks(code_lines_no_options)
    out = _to_quarto(options, blocks)
    return out


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
            f"{github_url}/{owner_repo}/blob/{branch}/{rel_path!s}"
            f"#L{line_start + 1}-L{line_end}"
        )
        return source_url

    def _get_class_parent_string(self, data):
        """Get the class parent string."""
        obj = data["object"]
        parents = [x for x in obj.__mro__[1:-1]]
        # this class only inherits from object (or type if metaclass)
        if not len(parents):
            return ""
        parent_list = []
        for parent in parents:
            name = parent.__name__
            key = str(id(parent))
            if parent_data := self._data_dict.get(key):
                parent_str = f"[{name}](`{parent_data['key']}`)"
            else:
                parent_str = (
                    str(parent).replace("<class ", "").replace(">", "").replace("'", "")
                )
            parent_list.append(parent_str)
        out = f"<br>inherits from: {', '.join(parent_list)} \n"
        return out

    def _get_parent_source_block(self, data):
        """Create a parent block with a link."""
        parent = data["key"].removesuffix(f".{data['name']}")
        if parent:
            parent_str = f" of [{parent}](`{parent}`)"
        else:
            parent_str = ""
        origin_txt = f"*{self._data['data_type']}* {parent_str}"
        source_url = self._get_github_source(data)

        if inspect.isclass(data["object"]):
            # add class's ancestor(s)
            origin_txt += self._get_class_parent_string(data)

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
    """Write all the markdown to disk."""
    files_to_delete = set(Path(api_path).rglob("*.qmd"))
    for obj_id, data in data_dict.items():
        # get path and ensure parents exist
        sub_dir = api_path / "/".join(data["key"].split(".")[:-1])
        path = api_path / sub_dir / f"{data['name']}.qmd"

        path.parent.mkdir(exist_ok=True, parents=True)
        # remove path from files to delete
        if path in files_to_delete:
            files_to_delete.remove(path)
        # dont render non-target file if debugging
        if debug and data["name"] != "read":
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


def _clear_empty_directories(parent_path):
    """Recursively delete empty directories."""

    def _dir_empty(path):
        """Return True if directory is empty."""
        contents = (
            x
            for x in path.rglob("*")
            if not (x.name.startswith(".") and len(x.name) < 3)
        )
        if any(contents):
            return False
        return True

    for path in parent_path.rglob("*"):
        if not path.is_dir():
            continue
        if _dir_empty(path):
            os.rmdir(path)


def _map_other_qmd_files(doc_path=DOC_PATH, api_path=API_DOC_PATH):
    """Add all other qmd files, excluding API."""
    out = {}
    parent_doc = doc_path.parent
    api_relative = str(api_path.relative_to(doc_path))
    for path in DOC_PATH.rglob("*.qmd"):
        path_relative = str(path.relative_to(parent_doc))
        # Skip API docs
        if path_relative.startswith(api_relative):
            continue
        value = "/" + str(path.relative_to(doc_path))
        out[path_relative] = value
        # also add key with no qmd extension.
        out[path_relative.split(".")[0]] = value
    return out


def render_project(data_dict, address_dict, api_path=API_DOC_PATH, debug=False):
    """Render the markdown files."""
    # Create and write the qmd files for each function/class/module
    write_api_markdown(data_dict, api_path, address_dict, debug=debug)
    # put the parts together; alias; path to docs
    path_mapping = create_json_mapping(data_dict, address_dict, api_path)
    # Add the other parts of the documentation.
    path_mapping.update(_map_other_qmd_files())
    # dump the json mapping to disk in doc folder
    cross_ref_path = Path(DOC_PATH) / ".cross_ref.json"
    with open(cross_ref_path, "w") as fi:
        json.dump(path_mapping, fi, indent=2)
    # Clear out empty directories
    _clear_empty_directories(api_path)
