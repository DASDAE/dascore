"""
Panflute filter that embeds wikipedia text

Replaces markdown such as [Stack Overflow](wiki://) with the resulting text.
"""

import fnmatch

import panflute as pf


def action(elem, doc):
    # print(elem)
    if isinstance(elem, pf.Link) and fnmatch.fnmatch(elem.url, "%60*60"):
        found_str = elem.url[3:-3]
        new = f"/api/" + "/".join(found_str.split(".")) + ".qmd"
        elem.url = new
        return elem


def main(doc=None):
    return pf.run_filter(action, doc=doc)


if __name__ == "__main__":
    main()
