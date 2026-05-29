"""Module for visualizing spool file statistics"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats


def _find_outliers(gap, method, tolerance_percent):
    """
    Identify outliers in file interval data.

    Parameters
    ----------
    gap : array-like
        Array of time intervals between consecutive files, in seconds.
    method : {"mode", "mean", "median"}
        Method used to identify outliers.

        - ``"mode"``: values which differ from the most common interval.
        - ``"mean"``: values outside ``tolerance_percent`` of the mean.
        - ``"median"``: values outside ``tolerance_percent`` of the median.
    tolerance_percent : float
        Percentage deviation from the reference value used to classify
        outliers for the ``"mean"`` and ``"median"`` methods.

    Returns
    -------
    numpy.ndarray
        Array containing the indices of detected outliers.
    """

    def _is_not_within_tolerance(value, reference, tolerance_percent):
        """Calculate the absolute difference as a fraction of the reference"""
        diff_percent = abs((value - reference) / reference)

        # Returns True if the difference is tolerance_percent  or less
        return diff_percent > tolerance_percent / 100

    if method == "mode":
        # most common value; outliers are exact not matches!
        value = stats.mode(gap).mode
        outlier_index = np.where(gap != value)[0]

    elif method == "mean":
        reference_value = np.mean(gap)
        outlier_index = np.where(
            _is_not_within_tolerance(gap, reference_value, tolerance_percent)
        )[0]

    elif method == "median":
        reference_value = np.median(gap)
        outlier_index = np.where(
            _is_not_within_tolerance(gap, reference_value, tolerance_percent)
        )[0]

    return outlier_index


# %%
def viz_spool(spool, method="mode", tolerance_percent=20):
    """
    Visualize spool timing statistics.

    Creates two plots:

    1. The interval between consecutive files.
    2. The sample rate evolution through time.

    Detected outliers in file intervals are highlighted and annotated on
    the interval plot.

    Parameters
    ----------
    spool
        A DASCore spool or a pandas DataFrame containing spool contents.
        The contents must include ``time_min`` and ``time_step`` columns.
    method : {"mode", "mean", "median"}, optional
        Method used to identify outlier file intervals.

        - ``"mode"``: values which differ from the most common interval.
        - ``"mean"``: values outside ``tolerance_percent`` of the mean.
        - ``"median"``: values outside ``tolerance_percent`` of the median.
    tolerance_percent : float, optional
        Percentage deviation from the reference interval used to classify
        outliers for the ``"mean"`` and ``"median"`` methods.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis showing sample rate evolution.
    interval : numpy.ndarray
        The interval between consecutive file start times, in seconds.
    outlier_index : numpy.ndarray
        Indices of detected interval outliers.

    Raises
    ------
    TypeError
        If ``spool`` is not a DASCore spool or pandas DataFrame.
    ValueError
        If ``method`` is not a supported outlier detection method.

    Notes
    -----
    File intervals are computed from consecutive values in
    ``time_min``:

    ``gap = np.diff(time_min)``

    Therefore the reported intervals represent differences between file
    start times
    """
    if hasattr(spool, "get_contents"):
        df = spool.get_contents()
    elif isinstance(spool, pd.DataFrame):
        df = spool
    else:
        raise TypeError(
            f"Expected a DASCore spool or DataFrame, got {type(spool).__name__}"
        )

    valid_options = ["mode", "mean", "median", "ransac"]
    if method.lower() not in valid_options:
        raise ValueError(
            f"Unknown outlier detection method {method!r}. "
            f"Allowed values are {valid_options}."
        )
    method = method.lower()

    filestart = df["time_min"].to_numpy()
    dt = df["time_step"].to_numpy()
    fsamp = np.timedelta64(1, "s") / dt
    interval = np.diff(filestart) / np.timedelta64(1, "s")

    outlier_index = _find_outliers(interval, method, tolerance_percent)

    # plotting
    tick_map = [
        [1, "1 sec"],
        [10, "10 sec"],
        [60, "1 min"],
        [10 * 60, "10 min"],
        [3600, "1 hour"],
        [6 * 3600, "6 hours"],
        [86400, "1 day"],
        [7 * 86400, "1 week"],
        [30 * 86400, "1 month"],
    ]
    ticks = [x[0] for x in tick_map]
    ticklabels = [x[1] for x in tick_map]
    # % Plot
    _, axs = plt.subplots(2, 1, figsize=(12, 10), layout="constrained")

    # Plot file-time differences
    ax = axs[0]
    ax.semilogy(filestart[:-1], interval, ".")
    ax.set_yticks(
        ticks,
        ticklabels,
    )
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid("on")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("Filetime Difference")

    for i in outlier_index:
        txt = str(np.timedelta64(int(interval[i]), "s").item())
        txt = "  " + str(filestart[i])[:19]
        ax.text(filestart[i], interval[i], txt, rotation=90, ha="center", fontsize=8)

    # add start label
    txt = str(filestart[0])[:19].replace("T", "\n")
    ax.text(
        filestart[0],
        interval[0],
        txt,
        rotation=90,
        va="center",
        ha="right",
        fontsize=8,
        color="darkgreen",
    )
    # add end label
    txt = str(filestart[-1])[:19].replace("T", "\n")
    ax.text(
        filestart[-1],
        interval[-1],
        txt,
        rotation=90,
        va="center",
        ha="left",
        fontsize=8,
        color="red",
    )
    ax.set_title("Filetime Difference")

    # plor sampling rate evolution
    ax = axs[1]
    ax.plot(filestart, fsamp, "-", lw=3)
    # ax.set_yticks(ticks, ticklabels, )
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid("on")
    # ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("Sample Rate [Hz]")
    ax.set_title("Sample Rate Evolution")

    return ax, interval, outlier_index


# %%
if __name__ == "__main__":
    from dascore.utils.hdf5 import HDFPatchIndexManager

    tmpfile = Path(
        r"O:\Staff\andreasw\Dev\FibreEyes\Aurland\_dascore_index_Aurland.hdf5"
    )
    tmpfile = Path(
        r"C:\Users\andreasw\Downloads\Spool_Visualisation\_dascore_index_Aurland.hdf5"
    )

    df = HDFPatchIndexManager(tmpfile).get_index(time_min="2026-01-16T14:00:00")

    dummy = viz_spool(df, method="mode", tolerance_percent=20)


"""
    if 1:
        ax, duration, idx = viz_spool(df)
        without_gaps = np.delete(duration, idx)
        if np.allclose(without_gaps, without_gaps[0]):
            print(f"All files have identical duration of {duration[0]} seconds")
        else:
            without_gaps = np.random.normal(10, 0.2, len(duration) - len(idx))

            import matplotlib.ticker as mticker

            plt.hist(without_gaps, bins=25, log=True, edgecolor="black")
            plt.xlabel("File Duration [s]")
            plt.ylabel("Number of Files")
            plt.grid("on")
            plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            plt.title(tmpfile.stem)
"""
