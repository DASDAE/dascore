"""
Panflute filter that embeds wikipedia text

Replaces markdown such as [Stack Overflow](wiki://) with the resulting text.
"""
import fnmatch
import json
import warnings
from functools import cache
from pathlib import Path

import panflute as pf


@cache
def get_cross_ref_dict():
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


def action(elem, doc):
    """callback to run on each element."""
    if isinstance(elem, pf.Link) and fnmatch.fnmatch(elem.url, "%60*60"):
        mapping = get_cross_ref_dict()
        found_str = elem.url[3:-3]
        if found_str not in mapping:
            msg = f"{found_str} is not a listed cross reference! Aborting doc gen."
            warnings.warn(msg)
            # raise KeyError(msg)
        else:
            elem.url = mapping[found_str]
        return elem


def main(doc=None):
    return pf.run_filter(action, doc=None)


if __name__ == "__main__":
    main()
