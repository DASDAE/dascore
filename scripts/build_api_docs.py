"""
Script to build the API docs for dascore.
"""
from _index_api import get_alias_mapping, parse_project
from _render_api import render_project

import dascore

if __name__ == "__main__":
    data_dict = parse_project(dascore)
    obj_dict = get_alias_mapping(dascore)
    render_project(data_dict, obj_dict, debug=False)
