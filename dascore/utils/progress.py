"""Simple interface for progress markers."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sized
from contextlib import suppress

import rich.progress as prog

from dascore.compat import Progress
from dascore.config import get_config
from dascore.constants import PROGRESS_LEVELS


def get_progress_instance(progress: PROGRESS_LEVELS | Progress = "standard"):
    """
    Get the Rich progress bar instance based on complexity level.
    """
    # If a progress class is passed in, just use it.
    if isinstance(progress, Progress):
        return progress
    progress_list = [
        prog.SpinnerColumn(),
        prog.TextColumn("[progress.description]{task.description}"),
        prog.BarColumn(bar_width=30),
        prog.TaskProgressColumn(),
        prog.TimeRemainingColumn(),
        prog.TimeElapsedColumn(),
        prog.MofNCompleteColumn(),
    ]
    if progress == "basic":
        # set the refresh rate very low and eliminate the spinner
        return Progress(
            *progress_list[1:],
            refresh_per_second=get_config().progress_basic_refresh_per_second,
        )
    return Progress(*progress_list)


def track(
    sequence: Iterable,
    description: str,
    progress: PROGRESS_LEVELS | Progress = "standard",
    length: int | None = None,
    min_length: int = 1,
):
    """
    A simple iterator for tracking updates.

    Parameters
    ----------
    sequence
        A sequence or generator to trace the iteration over.
    description
        A string describing the operation
    progress
        options are
            None- disable progress bar,
            "basic" reduced refresh rate,
            "standard" - the normal progress bar
        can also accept a subclass of rich.progress.Progress.
    min_length
        The minimum length to emmit a progress bar.
    """
    # A generator has no length, so only measure when none was passed in.
    if not length and isinstance(sequence, Sized):
        # Being Sized is not a promise: len() of a 0-d array still raises.
        with suppress(TypeError, ValueError):
            length = len(sequence)
    # An unsized iterable with no length given gets no progress bar.
    if length is None or length < min_length:
        length = 0
    # This is a dirty hack to allow debugging while running tests.
    # Otherwise, pdb doesn't work in any tracking scope.
    # See: https://github.com/Textualize/rich/issues/1053
    # Rich progress requires a refresh thread, which WebAssembly can't start.
    no_threads = sys.platform in ("emscripten", "wasi")
    if get_config().debug or not length or progress is None or no_threads:
        yield from sequence
        return
    update = 1.0 if isinstance(progress, str) and progress == "standard" else 5.0
    progress = get_progress_instance(progress)
    with progress:
        yield from progress.track(
            sequence,
            total=length,
            description=description,
            update_period=update,
        )
