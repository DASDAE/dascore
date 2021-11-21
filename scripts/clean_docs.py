"""
Script to cleanup documentation.
"""
import shutil
from pathlib import Path

import typer


def clean_docs():
    """Clean all the docs"""
    doc_path = Path(__file__).absolute().parent.parent / "docs"
    # first delete all checkpoints
    for checkpoint in doc_path.rglob("*.ipynb_checkpoint"):
        shutil.rmtree(checkpoint)
    # make api documentation
    # cmd = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace"
    api_path = doc_path / "api"
    # clear API documentation
    if api_path.exists():
        shutil.rmtree(api_path)


if __name__ == "__main__":
    typer.run(clean_docs)
