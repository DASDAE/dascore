"""
Panflute filter that embeds wikipedia text

Replaces markdown such as [Stack Overflow](wiki://) with the resulting text.
"""
import fnmatch
import json
from functools import cache
from pathlib import Path

import panflute as pf


@cache
def get_cross_ref_dict():
    """
    Load cross reference dictionaries.

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


def action(elem, doc):
    if isinstance(elem, pf.Link) and fnmatch.fnmatch(elem.url, "%60*60"):
        mapping = get_cross_ref_dict()
        found_str = elem.url[3:-3]
        if found_str not in mapping:
            msg = f"{found_str} is not a listed cross reference! Aborting doc gen."
            raise KeyError(mapping)
        elem.url = mapping[found_str]
        return elem


def main(doc=None):
    return pf.run_filter(action, doc=doc)


if __name__ == "__main__":
    main()
