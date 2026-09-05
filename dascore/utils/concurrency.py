"""Small concurrency helpers for spool operations."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, cast

from dascore.config import config_context, get_config

T = TypeVar("T")


def _prefetch(iterable: Iterable[T], max_in_flight: int) -> Generator[T, None, None]:
    """Capture caller config and return an iterator with one loading thread."""
    config = get_config()

    def iterate() -> Generator[T, None, None]:
        """Keep a bounded set of next calls on the same worker thread."""
        if not max_in_flight:
            yield from iterable
            return
        iterator = iter(iterable)
        sentinel = object()

        def load():
            """Apply the caller's configuration for every resumed read."""
            with config_context(config):
                return next(iterator, sentinel)

        def close():
            """Finish the source iterator on the thread which advanced it."""
            with config_context(config):
                if closer := getattr(iterator, "close", None):
                    closer()

        with ThreadPoolExecutor(1, thread_name_prefix="dascore-prefetch") as pool:
            pending = deque()
            try:
                for _ in range(max_in_flight):
                    pending.append(pool.submit(load))
                while pending:
                    item = pending.popleft().result()
                    if item is sentinel:
                        break
                    # Refill before yielding so even a one-patch window
                    # overlaps the caller's processing with the next read.
                    pending.append(pool.submit(load))
                    yield cast(T, item)
                    del item
            finally:
                for future in pending:
                    future.cancel()
                # Queued reads are cancelled; a running read finishes before
                # closing its generator. The executor then joins its thread.
                pool.submit(close).result()

    return iterate()
