"""Tests for the backend's cached metadata reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import dascore as dc
from dascore.examples import spool_to_directory
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import summaries_to_records


def _records(name, patch=None, **attrs):
    """One source's records holding the example patch with ``attrs``."""
    patch = dc.get_example_patch() if patch is None else patch
    summary = patch.update_attrs(**attrs).summary
    update = {"source_path": name, "source_format": "DASDAE", "source_version": "1"}
    return summaries_to_records([summary.model_copy(update=update)])


def _reads(backend) -> dict:
    """Every cached read, as plain values."""
    return {
        "attr_names": backend.attr_names(),
        "coord_names": backend.coord_names(),
        "coord_dims_map": backend.coord_dims_map(),
        "associated": backend.associated_coord_names(),
        "attr_meta": backend._attr_meta().to_dict("list"),
        "coord_meta": backend.coord_meta().to_dict("list"),
    }


@pytest.fixture
def path(tmp_path):
    """An index file holding one source."""
    out = tmp_path / ".index.sqlite3"
    backend = get_backend(out)
    backend.write_sources(_records("a.h5", station="A"))
    backend.close()
    return out


class TestMetaCache:
    """Cached reads always answer what a fresh read would."""

    def test_own_write(self, path):
        """A write on the same connection is seen at once."""
        backend = get_backend(path)
        before = _reads(backend)
        patch = dc.get_example_patch().update_coords(lat=("distance", np.arange(300.0)))
        backend.write_sources(_records("b.h5", patch=patch, shot=3))
        after = _reads(backend)
        assert "shot" not in before["attr_names"]
        assert "shot" in after["attr_names"]
        assert "lat" in after["coord_dims_map"]
        assert "lat" in after["associated"]
        assert after == _reads(get_backend(path))

    def test_other_connection(self, path):
        """A commit by another connection is seen at once."""
        reader, writer = get_backend(path), get_backend(path)
        assert "shot" not in reader.attr_names()
        writer.write_sources(_records("b.h5", shot=3))
        assert "shot" in reader.attr_names()
        assert _reads(reader) == _reads(writer)

    def test_rollback(self, path):
        """Reads inside a transaction see its writes, and not once rolled back."""
        backend = get_backend(path)
        before = _reads(backend)
        row = ("shot", "num", "shot__num", None)
        with pytest.raises(RuntimeError, match="undo"):
            with backend._transaction():
                assert "shot" not in backend.attr_names()
                # straight on the connection: its changed-row count alone
                # must show the write
                backend._con.execute("INSERT INTO attr_meta VALUES (?, ?, ?, ?)", row)
                assert "shot" in backend.attr_names()
                raise RuntimeError("undo")
        assert _reads(backend) == before

    def test_returned_values_are_copies(self, path):
        """Changing a returned value does not change the next one."""
        backend = get_backend(path)
        backend.attr_names().add("bob")
        backend.coord_names().add("bob")
        backend.associated_coord_names().add("bob")
        backend.coord_dims_map()["bob"] = "bob"
        attrs, coords = backend._attr_meta(), backend.coord_meta()
        attrs.loc[0, "attr_name"] = "bob"
        coords.loc[0, "coord_name"] = "bob"
        assert "bob" not in backend.attr_names() | backend.coord_names()
        assert "bob" not in backend.associated_coord_names()
        assert "bob" not in backend.coord_dims_map()
        assert "bob" not in backend._attr_meta()["attr_name"].tolist()
        assert "bob" not in backend.coord_meta()["coord_name"].tolist()

    def test_coord_meta_by_names(self, path):
        """Each set of names reads its own rows, with no write between."""
        backend = get_backend(path)
        assert set(backend.coord_meta(["time"])["coord_name"]) == {"time"}
        assert set(backend.coord_meta(["distance"])["coord_name"]) == {"distance"}


@pytest.mark.concurrency
class TestThreadedReads:
    """Threads sharing one spool read the metadata cache safely."""

    def test_threaded_selects(self, tmp_path):
        """
        Selections from many threads at once all give one answer.

        This catches a `_cached` without its lock; that each call's frame
        is a copy of its own is `test_returned_values_are_copies`'s to show.
        """
        path = spool_to_directory(dc.get_example_spool("random_das"), tmp_path)
        spool = dc.spool(path).update()
        expected = len(spool.select(distance=(10, 200)))

        def select(_):
            return [len(spool.select(distance=(10, 200))) for _ in range(25)]

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(select, range(8)))
        assert results == [[expected] * 25] * 8
