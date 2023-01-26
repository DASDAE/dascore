"""
Custom filter that makes dynamic (sphinx-esc) cross links work.

Replaces markdown such as [Patch](`dascore.Patch`) with the path to the markdown
file created from dascore/scripts/build_api_docs.py.
"""
import io
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import Dict


@cache
def get_cross_ref_dict() -> Dict[str, str]:
    """
    Load cross-reference dictionaries.

    Returns
    -------
    """
    out = {}
    path = Path(__file__).absolute()
    while not path.name == "docs":
        path = path.parent
    for cross_ref in path.rglob("cross_ref.json"):
        out.update(json.loads(cross_ref.read_text()))
    return out


def load_stdin(input_stream=None):
    """
    Load input from stdin (json string)
    """
    if input_stream is None:
        input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    return input_stream.read()


def dump_stdout(output_str, out_stream=None):
    """
    Dump modified json string to std out.
    """
    if out_stream is None:
        out_stream = sys.stdout
    out_stream.write(output_str)


def replace_links(json_data, raw_string):
    """Replace the aliases (links) with the cross-references."""

    def _yield_links(element):
        """Yield all link elements."""
        if isinstance(element, list):
            for sub_element in element:
                yield from _yield_links(sub_element)
        elif isinstance(element, (dict)):
            if element.get("t") == "Link":
                yield element
            else:
                yield from _yield_links(element.get("c", []))

    reg_pattern = '(?<="\%60)(.*?)(?=\%60")'
    blocks = json_data["blocks"]

    replace_dict = {}
    for link in _yield_links(blocks):
        str_to_scan = json.dumps(link, separators=(",", ":"), ensure_ascii=False)
        if not str_to_scan in raw_string:
            from remote_pdb import RemotePdb

            RemotePdb("127.0.0.1", 4444).set_trace()

        matches = re.finditer(reg_pattern, str_to_scan)
        for match in matches:
            start, stop = match.span()
            cross_refs = get_cross_ref_dict()
            key = str_to_scan[start:stop].replace("`", "")
            new_value = cross_refs.get(key, key)
            if new_value != key:
                new_sub_str = str_to_scan.replace(f'"%60{key}%60"', f'"{new_value}"')
                replace_dict[str_to_scan] = new_sub_str
    for i, v in replace_dict.items():
        raw_string = raw_string.replace(i, v)
    return raw_string


def test():
    """Function to test filter."""
    here = Path(__file__).parent
    data_path = here / "filter_test_data" / "test_data_2.json"
    with data_path.open("r") as fi:
        data = fi.read()
    input_stream = io.StringIO(data)
    main(input_stream)


def main(raw_data=None):
    """Run filter."""
    raw_str = load_stdin(raw_data)
    json_data = json.loads(raw_str)
    output_data = replace_links(json_data, raw_str)
    dump_stdout(output_data)


if __name__ == "__main__":
    # test()  # uncomment to test.
    main()  # uncomment before running quarto so IO works.
