"""
A script to build quartos main config file.
"""
from pathlib import Path

from _render_api import get_template

import dascore as dc


def build_amp_toc_tree():
    """Build the toc tree for the API."""
    # out = ["- title: 'API'\n contents"]
    # api_path = Path(__file__).absolute().parent.parent / "docs" / "api"
    # sub_dirs = sorted(x for x in api_path.rglob("*") if x.is_dir())
    # for sub_dir in sub_dirs:
    #     level = len(str(sub_dir.relative_to(api_path)).split(os.sep))
    #     print(level)
    # for path in api_path.rglob("*"):
    #     if not path.is_dir():
    #         continue
    #     print(path)


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
    build_amp_toc_tree()
