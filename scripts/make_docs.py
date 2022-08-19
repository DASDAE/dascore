"""
Script to re-make the html docs and publish to gh-pages.
"""
from pathlib import Path
from subprocess import run

import typer

from clean_docs import clean_docs

# Path to top-level sphinx
DOC_PATH = Path(__file__).absolute().parent.parent / "docs"


def make_docs(doc_path=DOC_PATH, timeout=3000) -> str:
    """
    Make the dascore documentation.

    Parameters
    ----------
    doc_path
        The path to the top-level sphinx directory.
    timeout
        Time in seconds allowed for notebook to build.

    Returns
    -------
    Path to created html directory.

    """
    # clean out all the old docs
    clean_docs()
    # execute all the notebooks
    doc_path = Path(doc_path)
    # run auto api-doc
    run(
        "sphinx-apidoc ../dascore -e -M -o api -t _templates/autosummary",
        cwd=doc_path,
        shell=True,
        check=True,
    )
    run("make html", cwd=doc_path, shell=True, check=True)
    # ensure html directory was created, return path to it.
    expected_path: Path = doc_path / "_build" / "html"
    assert expected_path.is_dir(), f"{expected_path} does not exist!"
    return str(expected_path)


if __name__ == "__main__":
    typer.run(make_docs)
