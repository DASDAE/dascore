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


def _has_link(raw_data, start, stop):
    """Determine if the work 'link' occurs before start index."""
    new_start = max([start - 15, 0])
    return "link" in raw_data[new_start:stop]


def replace_links(data):
    reg_pattern = "(?<=\%60)(.*?)(?=\%60)"
    for match in re.finditer(reg_pattern, data):
        start, stop = match.span()
        # look back for "link" otherwise this may just be an emphasis.
        if not _has_link(data, start, stop):
            continue
        cross_refs = get_cross_ref_dict()
        key = data[start:stop].replace("`", "")
        new_value = cross_refs.get(key, key)
        data = f"{data[:start]}{new_value}{data[stop:]}"
    return data


def test():
    """Function to test filter."""
    here = Path(__file__).parent
    data_path = here / "filter_test_data" / "test.json"
    with data_path.open("r") as fi:
        data = fi.read()
    input_stream = io.StringIO(data)
    main(input_stream)


def main(raw_data=None):
    """Run filter."""
    raw_data = load_stdin(raw_data)
    output_data = replace_links(raw_data)
    dump_stdout(output_data)


if __name__ == "__main__":
    # test()  # uncomment to test.
    main()
