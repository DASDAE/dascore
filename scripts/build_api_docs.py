"""Script to build the API docs for dascore."""
from __future__ import annotations

from _index_api import get_alias_mapping, parse_project
from _qmd_builder import create_quarto_qmd
from _render_api import render_project
from _validate_links import validate_all_links

import dascore as dc

if __name__ == "__main__":
    print("Building documentation")  # noqa
    print(f"Parsing project {dc.__name__}")  # noqa
    data_dict = parse_project(dc)
    obj_dict = get_alias_mapping(dc)
    print("Generating qmd files")  # noqa
    render_project(data_dict, obj_dict, debug=False)
    # create the quarto info file (needs templating)
    print("creating quarto config")  # noqa
    create_quarto_qmd()
    # validate links
    print("Validating links")  # noqa
    validate_all_links()
