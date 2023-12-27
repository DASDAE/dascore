"""Simple interface for progress markers."""
from __future__ import annotations

import rich.progress as prog

import dascore as dc
from dascore.constants import PROGRESS_LEVELS


def get_progress_instance(progress: PROGRESS_LEVELS = "standard"):
    """
    Get the Rich progress bar instance based on complexity level.
    """
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
    return prog.Progress(*progress_list, **kwargs)


def track(sequence, description, progress: PROGRESS_LEVELS = "standard"):
    """A simple iterator for tracking updates."""
    # This is a dirty hack to allow debugging while running tests.
    # Otherwise, pdb doesn't work in any tracking scope.
    # See: https://github.com/Textualize/rich/issues/1053
    if dc._debug or not len(sequence) or progress is None:
        yield from sequence
        return
    update = 1.0 if progress == "standard" else 5.0
    progress = get_progress_instance(progress)
    with progress:
        yield from progress.track(
            sequence,
            total=len(sequence),
            description=description,
            update_period=update,
        )
