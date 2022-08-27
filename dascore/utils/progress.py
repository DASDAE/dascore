"""
Simple interface for progress markers.
"""
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

import dascore as dc


def track(sequence, description):
    """
    A simple iterator for tracking updates.
    """
    # This is a dirty hack to allow debugging while running tests.
    # Otherwise, pdb doesn't work in any tracking scope.
    # See: https://github.com/Textualize/rich/issues/1053
    if getattr(dc, "_debug") or not len(sequence):
        yield from sequence
        return
    # Normal progress bar behavior
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    )
    total = len(sequence)
    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=1.0
        )
