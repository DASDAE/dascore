"""Opt-in prefetch and executor-backed directory update behavior."""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.config import config_context, get_config
from dascore.examples import inventory_patch_pair
from dascore.exceptions import (
    InvalidSpoolError,
    MissingPatchError,
    ParameterError,
)
from dascore.io.index import indexer as indexer_module


class _RecordingClient:
    """Map-only client recording batches without requiring submit or shutdown."""

    _max_workers = 2

    def __init__(self):
        """Start with no submitted batches."""
        self.batches = []

    def map(self, func, iterable):
        """Record each batch before scanning it synchronously."""
        for batch in iterable:
            self.batches.append(batch)
            yield func(batch)


@pytest.fixture()
def directory(tmp_path):
    """Twelve small, contiguous patches in separate real HDF5 files."""
    path = tmp_path / "data"
    path.mkdir()
    # Leave identity generation to the scan so worker config is observable.
    with config_context(patch_provenance="disabled"):
        spool = dc.get_example_spool(length=12, shape=(3, 20))
        for index, patch in enumerate(spool):
            patch.io.write(path / f"{index:02d}.h5", "DASDAE")
    return path


@pytest.mark.concurrency
class TestSpoolIterate:
    """Prefetch preserves the behavior of the normal spool iterator."""

    @pytest.mark.parametrize("view", ["root", "select", "chunk", "reversed"])
    def test_directory_views(self, directory, view):
        """Selections, merged patches, and presentation order survive prefetch."""
        root = dc.spool(directory).update(progress=False)
        try:
            spool = root
            if view == "select":
                spool = root.select(time=(0.015, 0.345), relative=True)
            elif view == "chunk":
                spool = root.chunk(time=0.16)
            elif view == "reversed":
                spool = root[::-1]
            with closing(spool.iterate()) as iterator:
                for actual, expected in zip(iterator, spool, strict=True):
                    assert actual.equals(expected)
        finally:
            root.indexer.close()

    def test_inventory(self):
        """Prefetch uses the same inventory enrichment as normal iteration."""
        patch, inventory = inventory_patch_pair()
        spool = dc.spool(patch).attach_inventory(inventory).enrich()
        (actual,) = list(spool.iterate())
        assert actual.equals(next(iter(spool)))
        assert actual.attrs.gauge_length == 10

    @pytest.mark.parametrize("length", [0, 1, 5])
    def test_memory_spools(self, length):
        """Empty, single-patch, and longer in-memory spools terminate in order."""
        spool = dc.get_example_spool(length=length, shape=(2, 5))
        actual = list(spool.iterate())
        assert len(actual) == length
        assert all(a is b for a, b in zip(actual, spool, strict=True))

    @pytest.mark.parametrize("limit", [0, 2])
    def test_call_time_config(self, monkeypatch, limit):
        """Threads capture call-time config; zero follows ordinary iteration."""
        spool = dc.get_example_spool(length=3, shape=(2, 5))
        seen = []
        original = dc.Spool.__iter__

        def iterate(self):
            """Record the configuration each patch observes."""
            for patch in original(self):
                seen.append(get_config().display_float_precision)
                yield patch

        monkeypatch.setattr(dc.Spool, "__iter__", iterate)
        ambient = get_config().display_float_precision
        with config_context(display_float_precision=8):
            iterator = spool.iterate(max_in_flight=limit)
        assert seen == []
        assert len(list(iterator)) == 3
        assert seen == [8 if limit else ambient] * 3

    def test_zero_stays_in_caller(self, monkeypatch):
        """A zero window creates no worker and follows ordinary iteration."""
        original = dc.Spool.__iter__
        caller = threading.get_ident()

        def iterate(self):
            """Verify synchronous reads stay in the calling thread."""
            assert threading.get_ident() == caller
            yield from original(self)

        monkeypatch.setattr(dc.Spool, "__iter__", iterate)
        spool = dc.get_example_spool(length=2, shape=(2, 5))
        assert len(list(spool.iterate(max_in_flight=0))) == 2

    @pytest.mark.parametrize("limit", [1, 2, np.int64(3)])
    def test_overlap_and_bound(self, monkeypatch, limit):
        """Loading advances during consumer work, then stops at its bound."""
        original = dc.Spool.__iter__
        full, overflow, closed = threading.Event(), threading.Event(), threading.Event()
        worker_ids = []
        counts = []

        def iterate(self):
            """Signal when loading reaches or exceeds the configured window."""
            worker_ids.append(threading.get_ident())
            try:
                for index, patch in enumerate(original(self)):
                    counts.append(index)
                    if index == limit:
                        full.set()
                    elif index > limit:
                        overflow.set()
                    yield patch
            finally:
                closed.set()

        monkeypatch.setattr(dc.Spool, "__iter__", iterate)
        spool = dc.get_example_spool(length=20, shape=(2, 5))
        iterator = spool.iterate(max_in_flight=limit)
        assert counts == []  # Merely requesting the iterator performs no reads.
        try:
            next(iterator)
            assert full.wait(5)  # No second next() call is needed to load ahead.
            assert not overflow.wait(0.1)
            assert len(counts) == limit + 1  # Yielded patch plus the window.
            assert worker_ids == [worker_ids[0]]
            assert worker_ids[0] != threading.get_ident()
        finally:
            iterator.close()
        assert closed.is_set()
        assert len(counts) == limit + 1
        assert not any(t.ident == worker_ids[0] for t in threading.enumerate())

    def test_close_cancels_pending(self, monkeypatch):
        """Closing waits for the active read without starting queued reads."""
        original = dc.Spool.__iter__
        running, release = threading.Event(), threading.Event()
        cancelled, closed = threading.Event(), threading.Event()
        reads, cancellations, worker_ids = [], [], []
        original_cancel = Future.cancel

        def cancel(future):
            """Signal after both queued reads have been cancelled."""
            result = original_cancel(future)
            if result:
                cancellations.append(future)
                if len(cancellations) == 2:
                    cancelled.set()
            return result

        def iterate(self):
            """Hold the active read until the closing test releases it."""
            worker_ids.append(threading.get_ident())
            try:
                for index, patch in enumerate(original(self)):
                    reads.append(index)
                    if index == 1:
                        running.set()
                        assert release.wait(10)
                    yield patch
            finally:
                closed.set()

        monkeypatch.setattr(Future, "cancel", cancel)
        monkeypatch.setattr(dc.Spool, "__iter__", iterate)
        spool = dc.get_example_spool(length=10, shape=(2, 5))
        iterator = spool.iterate(max_in_flight=3)
        next(iterator)
        assert running.wait(5)
        with ThreadPoolExecutor(1) as closer:
            closing = closer.submit(iterator.close)
            try:
                assert cancelled.wait(5)
                assert not closing.done()
                assert not closed.is_set()
            finally:
                release.set()
                closing.result(timeout=5)
        assert reads == [0, 1]
        assert closed.is_set()
        assert not any(t.ident == worker_ids[0] for t in threading.enumerate())

    def test_read_error_closes(self, monkeypatch):
        """A read failure reaches the consumer and the source is closed."""
        original = dc.Spool.__iter__
        closed = threading.Event()

        def iterate(self):
            """Fail the second read and record source cleanup."""
            try:
                yield next(original(self))
                raise OSError("failed read")
            finally:
                closed.set()

        monkeypatch.setattr(dc.Spool, "__iter__", iterate)
        spool = dc.get_example_spool(length=2, shape=(2, 5))
        iterator = spool.iterate()
        next(iterator)
        with pytest.raises(OSError, match="failed read"):
            next(iterator)
        assert closed.is_set()

    def test_skips_unresolvable(self, monkeypatch):
        """Missing patches retain ordinary iteration's warning-and-skip policy."""
        spool = dc.get_example_spool(length=3, shape=(2, 5))
        spool.get_contents()
        original = spool._catalog.resolve_row
        count = 0

        def resolve(row):
            """Make only the second catalog row unresolvable."""
            nonlocal count
            count += 1
            if count == 2:
                raise MissingPatchError("unavailable")
            return original(row)

        monkeypatch.setattr(spool._catalog, "resolve_row", resolve)
        with pytest.warns(UserWarning, match="Skipping patch"):
            assert len(list(spool.iterate())) == 2


class TestIterateValidation:
    """Invalid windows are rejected before starting a thread."""

    @pytest.mark.parametrize("limit", [-1, 1.5, None, True, False, "2"])
    def test_bad_window(self, limit):
        """Only non-negative integers are accepted, even for an empty spool."""
        with pytest.raises(ParameterError, match="non-negative integer"):
            dc.spool([]).iterate(max_in_flight=limit)


@pytest.mark.concurrency
class TestUpdateExecutor:
    """Use real process/thread executors without changing indexing behavior."""

    @pytest.fixture(scope="class", params=["thread", "process"])
    @classmethod
    def client(cls, request):
        """A caller-owned executor reused across multiple update calls."""
        if request.param == "process":
            pool = ProcessPoolExecutor(
                2,
                mp_context=multiprocessing.get_context("spawn"),
            )
        else:
            pool = ThreadPoolExecutor(2)
        with pool:
            yield pool

    def test_matches_serial(self, directory, tmp_path, client):
        """Full metadata, identities, and loaded data match the serial index."""
        serial = dc.Spool.from_directory(directory, index_path=tmp_path / "serial.db")
        parallel = dc.Spool.from_directory(
            directory, index_path=tmp_path / "parallel.db"
        )
        try:
            serial.update(progress=False)
            parallel.update(progress=False, client=client)
            pd.testing.assert_frame_equal(
                serial.get_contents(), parallel.get_contents()
            )
            for actual, expected in zip(parallel, serial, strict=True):
                assert actual.equals(expected)
            assert list(client.map(abs, [-2, -1])) == [2, 1]
        finally:
            serial.indexer.close()
            parallel.indexer.close()

    def test_config_and_lifecycle(self, directory, client):
        """Workers inherit scoped config; changed and deleted files refresh."""
        root = dc.spool(directory)
        try:
            with config_context(patch_provenance="disabled"):
                root.update(progress=False, client=client)
            assert "patch_id" not in root.get_contents()
            assert all(x.attrs.patch_id for x in dc.scan(directory, progress=False))
            extra = dc.get_example_patch(shape=(3, 20), tag="new")
            extra.io.write(directory / "extra.h5", "DASDAE")
            (directory / "00.h5").unlink()
            # The DASDAE writer appends groups, so replace the file to test
            # an actual source rewrite rather than a newly appended patch.
            (directory / "01.h5").unlink()
            extra.update_attrs(tag="modified").io.write(directory / "01.h5", "DASDAE")
            root.update(progress=False, client=client)
            contents = root.get_contents()
            assert len(contents) == 12
            assert len(contents[contents["tag"] == "new"]) == 1
            assert len(contents[contents["tag"] == "modified"]) == 1
            before = contents.copy()
            root.update(progress=False, client=client)
            pd.testing.assert_frame_equal(before, root.get_contents())
        finally:
            root.indexer.close()


class TestUpdateBatches:
    """Batching and error policies are independent of executor implementation."""

    def test_batches_and_no_change(self, directory):
        """Changed paths arrive in several batches; no-op updates submit none."""
        client = _RecordingClient()
        root = dc.spool(directory)
        try:
            root.update(progress=False, client=client)
            assert len(client.batches) > 1
            assert all(len(batch) > 1 for batch in client.batches)
            assert set(path for batch in client.batches for path in batch) == set(
                directory.glob("*.h5")
            )
            client.batches.clear()
            root.update(progress=False, client=client)
            assert client.batches == []
            (directory / "00.h5").rename(directory / "renamed.h5")
            root.update(progress=False, client=client)
            assert client.batches == []
            assert len(root) == 12
        finally:
            root.indexer.close()

    def test_map_only_client(self, directory):
        """An executor need only implement the existing ordered-map protocol."""

        class Client:
            """No worker-count, submit, or shutdown API."""

            def map(self, func, iterable):
                """Return ordered results without an executor lifecycle API."""
                return map(func, iterable)

        root = dc.spool(directory)
        try:
            assert len(root.update(progress=False, client=Client())) == 12
        finally:
            root.indexer.close()

    def test_interrupted_collection_closes_results(self, directory, monkeypatch):
        """An interrupted collection closes results while its traceback lives."""
        closed = threading.Event()

        class Client:
            def map(self, func, iterable):
                """Record closure while yielding actual worker results."""
                try:
                    for batch in iterable:
                        yield func(batch)
                finally:
                    closed.set()

        def interrupt(results, *args, **kwargs):
            """Interrupt progress reporting with the source iterator suspended."""
            next(results)
            raise KeyboardInterrupt

        monkeypatch.setattr(indexer_module, "track", interrupt)
        root = dc.spool(directory)
        try:
            with pytest.raises(KeyboardInterrupt) as exc_info:
                root.update(client=Client())
            assert exc_info.traceback
            assert closed.is_set()
        finally:
            root.indexer.close()

    def test_derived_refused(self, directory):
        """An executor does not permit updating a derived spool."""
        root = dc.spool(directory).update(progress=False)
        client = _RecordingClient()
        try:
            with pytest.raises(InvalidSpoolError, match="root spool"):
                root[:1].update(progress=False, client=client)
            assert client.batches == []
        finally:
            root.indexer.close()

    def test_invalid_progress(self, directory):
        """Invalid progress is rejected before the executor receives work."""
        root = dc.spool(directory)
        client = _RecordingClient()
        try:
            with pytest.raises(ParameterError, match="progress"):
                root.update(progress="wrong", client=client)
            assert client.batches == []
        finally:
            root.indexer.close()
