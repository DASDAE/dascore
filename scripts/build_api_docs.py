"""
Script to build the API docs for dascore.
"""
from pathlib import Path

from _index_api import get_alias_mapping, parse_project
from _render_api import get_template, render_project

import dascore as dc


def create_quarto_qmd():
    """Create the _quarto.yml file."""

    def _get_nice_version_string():
        """Just get a simplified version string."""
        version_str = str(dc.__version__)
        if "dev" not in version_str:
            vstr = version_str
        else:
            vstr = version_str.split("+")[0]
        return f"DASCore ({vstr})"

    temp = get_template("_quarto.yml")
    version_str = _get_nice_version_string()
    out = temp.render(dascore_version_str=version_str)
    path = Path(__file__).parent.parent / "docs" / "_quarto.yml"
    with path.open("w") as fi:
        fi.write(out)


if __name__ == "__main__":
    data_dict = parse_project(dc)
    obj_dict = get_alias_mapping(dc)
    render_project(data_dict, obj_dict, debug=False)
    create_quarto_qmd()
