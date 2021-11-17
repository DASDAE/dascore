"""
Script to cleanup documentation.
"""
import shutil
from pathlib import Path
from subprocess import run

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
    # execute all the notebooks
    for note_book_path in doc_path.rglob("*.ipynb"):
        if "ipynb_checkpoints" in str(note_book_path):
            continue
        # result = run(cmd + f" {note_book_path}", shell=True)
        # if result.returncode != 0:
        #     msg = f"failed to execute {note_book_path}!"
        #     raise RuntimeError(msg)


if __name__ == "__main__":
    typer.run(clean_docs)
