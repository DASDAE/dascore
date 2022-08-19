"""
Simple interface for progress markers.
"""

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


def track(sequence, description):
    """
    A simple iterator for tracking updates.
    """
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )
    total = len(sequence)
    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=1.0
        )


track_index_update = track


def dummy_track(iterable, *args, **kwargs):
    """A dummy tracker for debugging"""
    return iterable
