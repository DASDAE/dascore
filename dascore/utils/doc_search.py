"""Search the installed Markdown corpus with the optional Tantivy engine."""

from __future__ import annotations

import json
import shutil
from importlib.metadata import version
from pathlib import Path

from dascore.utils.doc_cache import documentation_cache
from dascore.utils.doc_corpus import DocumentationError
from dascore.utils.misc import optional_import

_FORMAT_VERSION = 1
_SEARCH_FIELDS = ["title", "aliases", "keywords", "body"]
# Keywords are sparse; compensate for field-length normalization so authored
# topic labels outweigh repeated API qualification in titles and aliases.
_BOOSTS = {"title": 3.0, "aliases": 3.0, "keywords": 200.0}


def _schema(engine):
    """Keep exact tags separate from searchable, stemmed text."""
    builder = engine.SchemaBuilder()
    builder.add_text_field("identifier", stored=True, tokenizer_name="raw")
    builder.add_text_field("kind", stored=True, tokenizer_name="raw")
    builder.add_text_field("tags", stored=True, tokenizer_name="raw")
    for name in _SEARCH_FIELDS:
        builder.add_text_field(name, stored=True, tokenizer_name="en_stem")
    return builder.build()


def _index(engine, root: Path, manifest: dict):
    """Open a matching index or rebuild it while the corpus lock is held."""
    directory = root / "search"
    marker = directory / "identity.json"
    identity = {
        "corpus": manifest["identity"],
        "format": _FORMAT_VERSION,
        "engine": version("tantivy"),
    }
    try:
        if json.loads(marker.read_text(encoding="utf-8")) == identity:
            index = engine.Index.open(str(directory))
            return index
    except (OSError, ValueError):
        pass
    # The index is disposable. Only a completed build receives its identity marker.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()
    schema = _schema(engine)
    index = engine.Index(schema, path=str(directory), reuse=False)
    writer = index.writer(heap_size=32_000_000, num_threads=1)
    for record in manifest["documents"]:
        body = (root / record["path"]).read_text(encoding="utf-8")
        body = body.split("\n---\n", 1)[1].lstrip().partition("\n")[2].lstrip()
        if record["kind"] == "api":
            body = body.partition("\n")[2].lstrip()
        tags = [tag.casefold() for tag in record["keywords"]]
        fields = {
            "identifier": record["id"],
            "title": record["title"],
            "kind": record["kind"],
            "aliases": " ".join(record["aliases"]),
            "keywords": record["keywords"],
            "tags": tags,
            "body": body,
        }
        writer.add_document(engine.Document.from_dict(fields, schema))
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    marker.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    return index


def search_documents(
    query: str = "", *, tag: str | None = None, limit: int = 5
) -> list[dict]:
    """
    Find documentation by text and an optional exact keyword tag.

    Parameters
    ----------
    query
        Tantivy query text. Words must all match by default; quoted phrases
        and Boolean operators are supported.
    tag
        Optional case-insensitive keyword filter. Can be used without text.
    limit
        Maximum number of results, from 1 to 100.

    Returns
    -------
    list[dict]
        Document identifiers, titles, kinds, keywords, and plain-text excerpts.
        Identifiers can be passed directly to `dascore doc`.
    """
    query, tag = query.strip(), (tag or "").strip().casefold()
    if not query and not tag:
        raise DocumentationError("Provide a search query or --tag.")
    if not 1 <= limit <= 100:
        raise DocumentationError("Search limit must be between 1 and 100.")
    engine = optional_import("tantivy", required_for="documentation search")
    with documentation_cache() as (root, manifest):
        # Finish the helper's index/reader lifetime before releasing the cache lock.
        try:
            return _search_index(
                engine, _index(engine, root, manifest), query, tag, limit
            )
        except DocumentationError:
            raise
        except ValueError as exc:
            # Tantivy translates native filesystem/index failures to ValueError.
            raise DocumentationError(f"Documentation search failed: {exc}") from exc


def _search_index(engine, index, query, tag, limit):
    """Query an open index and materialize results while its cache is locked."""
    try:
        parsed = (
            index.parse_query(
                query,
                _SEARCH_FIELDS,
                field_boosts=_BOOSTS,
                conjunction_by_default=True,
            )
            if query
            else engine.Query.all_query()
        )
    except ValueError as exc:
        raise DocumentationError(f"Invalid search query: {exc}") from exc
    if tag:
        exact_tag = engine.Query.term_query(index.schema, "tags", tag)
        parsed = engine.Query.boolean_query(
            [(engine.Occur.Must, parsed), (engine.Occur.Must, exact_tag)]
        )
    searcher = index.searcher()
    snippets = engine.SnippetGenerator.create(searcher, parsed, index.schema, "body")
    snippets.set_max_num_chars(200)
    results = []
    for _score, address in searcher.search(parsed, limit=limit).hits:
        document = searcher.doc(address)
        excerpt = snippets.snippet_from_doc(document).fragment()
        if not excerpt:
            excerpt = document["body"][0][:200]
        results.append(
            {
                "id": document["identifier"][0],
                "title": document["title"][0],
                "kind": document["kind"][0],
                "keywords": document["keywords"],
                "excerpt": " ".join(excerpt.split()),
            }
        )
    return results
