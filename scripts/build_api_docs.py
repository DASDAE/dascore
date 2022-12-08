"""
Script to build the API docs for dascore.
"""
import dascore
from _index_api import parse_project, get_alias_mapping
from _render_api import render_project


if __name__ == "__main__":
    data_dict = parse_project(dascore)
    obj_dict = get_alias_mapping(dascore)
    render_project(data_dict, obj_dict, debug=False)
