"""
Reading and writing the small YAML-or-JSON documents DASCore stores.

An inventory object, an annotation set's attributes and a saved workflow are
all one mapping written as either spelling, so they are read the same way:
the caller decides which parser its own naming rules call for, and this
turns whatever comes back into a mapping or says why it could not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml

from dascore.exceptions import ParameterError
from dascore.utils.misc import _maybe_make_parent_directory
from dascore.utils.paths import quote_path

DocumentFormat = Literal["json", "yaml"]

# Read with the BOM stripped: an editor on Windows writes one, and it is not
# part of the document.
ENCODING = "utf-8-sig"


def read_document(
    path: Path,
    file_format: DocumentFormat,
    *,
    error: type[Exception] = ParameterError,
    holds: str = "describes nothing",
) -> dict[str, Any]:
    """
    Return the mapping a document file holds.

    Which parser to use is the caller's to decide, because each format
    names its files its own way; everything after that is the same.

    Parameters
    ----------
    path
        The file to read.
    file_format
        Either "json" or "yaml".
    error
        The exception raised for a file which cannot be read, cannot be
        parsed, or holds something other than a mapping.
    holds
        How to finish the sentence "<file> holds no mapping, so it ...".
    """
    path = Path(path)
    try:
        text = path.read_text(encoding=ENCODING)
    except (OSError, UnicodeDecodeError) as problem:
        msg = f"Could not read {quote_path(path)}: {problem}."
        raise error(msg) from problem
    document = parse_document(
        text,
        file_format,
        label=quote_path(path),
        error=error,
    )
    if not isinstance(document, Mapping):
        msg = f"{quote_path(path)} holds no mapping, so it {holds}."
        raise error(msg)
    return dict(document)


def dump_document(
    document: Mapping,
    file_format: DocumentFormat,
) -> str:
    """
    Return the text a mapping is written as, in the format named.

    Either spelling keeps the order the document states rather than
    sorting: a document says what it holds, its own order reads better, and
    a set written twice reads the same both times.
    """
    if file_format == "yaml":
        return yaml.safe_dump(dict(document), sort_keys=False)
    return json.dumps(document, indent=2)


def write_document(
    document: Mapping,
    path: Path,
    file_format: DocumentFormat,
) -> Path:
    """
    Write a mapping to a file, in the format named, and return the path.

    Parent directories are made as needed, so saving into a directory which
    is not there yet works.
    """
    text = dump_document(document, file_format)
    return write_text_document(text, path)


def write_text_document(text: str, path: str | Path) -> Path:
    """
    Write a document's text to a file and return the path.

    For a caller which already has the text -- because it hands the same
    string back -- and would otherwise write it and read it again.
    """
    path = Path(path)
    _maybe_make_parent_directory(path)
    path.write_text(text, encoding="utf-8")
    return path


def parse_document(
    text: str,
    file_format: DocumentFormat,
    *,
    label: str,
    error: type[Exception] = ParameterError,
) -> Any:
    """
    Return whatever a document's text holds, naming its source if it cannot.

    Parameters
    ----------
    text
        The document's text.
    file_format
        Either "json" or "yaml".
    label
        What to call the text in an error: a file's name, or a phrase for
        text which came from somewhere else.
    error
        The exception raised for text which does not parse.
    """
    if file_format == "json":
        try:
            return json.loads(text)
        except ValueError as problem:
            msg = f"Could not parse JSON from {label}: {problem}."
            raise error(msg) from problem
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as problem:
        msg = f"Could not parse YAML from {label}: {problem}."
        raise error(msg) from problem
