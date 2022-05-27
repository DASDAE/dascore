"""
Simple interface for progress markers.
"""

from rich.progress import track


track_index_update = track


def dummy_track(iterable, *args, **kwargs):
    """A dummy tracker for debugging"""
    return iterable
