"""Simple interface for progress markers."""

from __future__ import annotations

from collections.abc import Generator, Sized
from contextlib import suppress

import rich.progress as prog

import dascore as dc
from dascore.compat import Progress
from dascore.constants import PROGRESS_LEVELS


def get_progress_instance(progress: PROGRESS_LEVELS | Progress = "standard"):
    """
    Get the Rich progress bar instance based on complexity level.
    """
    # If a progress class is passed in, just use it.
    if isinstance(progress, Progress):
        return progress
    kwargs = {}
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
        kwargs["refresh_per_second"] = 0.25
        progress_list = progress_list[1:]
    return Progress(*progress_list, **kwargs)


def track(
    sequence: Sized | Generator,
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
    # In the case of a generator we need to make sure this just exists
    guess_len = length if length is not None else 0
    with suppress(TypeError, ValueError):
        length = len(sequence) if not guess_len else guess_len
    if length < min_length:
        length = 0
    # This is a dirty hack to allow debugging while running tests.
    # Otherwise, pdb doesn't work in any tracking scope.
    # See: https://github.com/Textualize/rich/issues/1053
    if dc._debug or not length or progress is None:
        yield from sequence
        return
    update = 1.0 if isinstance(progress, str) and progress == "standard" else 5.0
    progress = get_progress_instance(progress)
    with progress:
        yield from progress.track(
            sequence,
            total=length or len(sequence),
            description=description,
            update_period=update,
        )
