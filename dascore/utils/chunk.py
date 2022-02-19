"""
Utilities for chunking dataframe.
"""
import numpy as np
import pandas as pd


def get_intervals(
    start,
    stop,
    step,
    overlap=None,
):
    """
    Create a range of values with optional overlaps.

    Parameters
    ----------
    start
        The start of the interval.
    stop
        The end of the interval.
    step
        Step size.
    overlap
        The overlap of the start of each interval with the end
        of the previous interval.

    Returns
    -------
    A 2D array where first column is start and second column is end.
    """
    # get variable and perform checks
    overlap = step * 0 if not overlap else overlap
    assert overlap < step, "Overlap must be less than step"
    assert (stop - start) > step, "Range must be greater than step"
    # reference with no overlap
    new_step = step - overlap
    reference = np.arange(start, stop + new_step, step=new_step)
    ends = reference[:-1] + step
    starts = reference[:-1]
    # trim end to not surpass stop
    if ends[-1] > stop:
        ends, starts = ends[:-1], starts[:-1]
    return np.stack([starts, ends]).T


def chunk(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Chunk a datafarme based on columns of the dataframe.

    Parameters
    ----------
    df
        Input dataframe to chunk.
    **kwargs
        Used to specify the dimensions to chunk. The name of the kwarg
        should be a column

    """
