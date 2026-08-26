"""Simple interface for progress markers."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing import Literal, get_args

import rich.progress as prog

from dascore.compat import Progress
from dascore.config import get_config
from dascore.constants import PROGRESS_LEVELS
from dascore.exceptions import ParameterError

#: The levels which actually produce a bar; the others turn it off.
BAR_LEVELS = Literal["standard", "basic"]

#: The levels safe to compare by equality. False is accepted, but only by
#: identity: it is equal to 0, so leaving it here would accept any zero as
#: a level and hand it on to be turned into a standard bar.
_EQUATABLE_LEVELS = tuple(x for x in get_args(PROGRESS_LEVELS) if x is not False)


def get_progress_instance(progress: BAR_LEVELS | Progress = "standard"):
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


def get_track_length(sequence: Iterable, length: int | None, min_length: int) -> int:
    """
    Return the total to report to the progress bar.

    A length of 0 means no bar: either the sequence is too short to be
    worth one, or it is an iterable whose length cannot be known.
    """
    # A generator has no length, so only measure when none was passed in.
    if not length and isinstance(sequence, Sized):
        # Being Sized is not a promise: len() of a 0-d array still raises.
        with suppress(TypeError, ValueError):
            length = len(sequence)
    if length is None or length < min_length:
        return 0
    return length


def validate_progress_level(progress):
    """
    Return a supported progress level, or the caller's own bar.

    `False` is the obvious way to ask for no bar, so it is accepted and
    returned as None. Callers must use the returned value: anything else
    falls through to the standard bar, so a `False` passed on unchanged
    would turn one on rather than off. `True` is not accepted, having no
    one obvious meaning where there are two bars to choose between.
    """
    if progress is False:
        return None
    if isinstance(progress, Progress) or progress in _EQUATABLE_LEVELS:
        return progress
    msg = (
        f"progress must be one of {get_args(PROGRESS_LEVELS)} or a "
        f"rich Progress, got {progress!r}."
    )
    raise ParameterError(msg)


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
            None (or False) - disable progress bar,
            "basic" reduced refresh rate,
            "standard" - the normal progress bar
        can also accept a subclass of rich.progress.Progress.
    length
        The number of items, for sequences which cannot report their own.
    min_length
        The minimum length to emit a progress bar.
    """
    progress = validate_progress_level(progress)
    total = get_track_length(sequence, length, min_length)
    # This is a dirty hack to allow debugging while running tests.
    # Otherwise, pdb doesn't work in any tracking scope.
    # See: https://github.com/Textualize/rich/issues/1053
    # Rich progress requires a refresh thread, which WebAssembly can't start.
    no_threads = sys.platform in ("emscripten", "wasi")
    if get_config().debug or not total or progress is None or no_threads:
        yield from sequence
        return
    update = 1.0 if isinstance(progress, str) and progress == "standard" else 5.0
    progress = get_progress_instance(progress)
    with progress:
        yield from progress.track(
            sequence,
            total=total,
            description=description,
            update_period=update,
        )
