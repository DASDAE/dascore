"""Simple interface for progress markers."""
from __future__ import annotations

import rich.progress as prog

import dascore as dc


def track(sequence, description):
    """A simple iterator for tracking updates."""
    # This is a dirty hack to allow debugging while running tests.
    # Otherwise, pdb doesn't work in any tracking scope.
    # See: https://github.com/Textualize/rich/issues/1053
    if dc._debug or not len(sequence):
        yield from sequence
        return
    # Normal progress bar behavior
    progress = prog.Progress(
        # prog.SpinnerColumn(),
        prog.TextColumn("[progress.description]{task.description}"),
        prog.BarColumn(bar_width=30),
        prog.TaskProgressColumn(),
        prog.TimeRemainingColumn(),
        prog.TimeElapsedColumn(),
        prog.MofNCompleteColumn(),
        refresh_per_second=1,
    )
    # Note: we rest the refresh-rate low because of latency issues with
    # remote servers. It also helped to remove the spinner so the bar didnt
    # look so jumpy. See #305.
    total = len(sequence)
    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=1.0
        )
