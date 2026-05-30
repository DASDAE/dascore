"""Module for visualizing spool file statistics"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats


def _find_outliers(gap, duration, method, tolerance_percent):
    """
    Identify outliers in file gap data.

    Parameters
    ----------
    gap : array-like
        Array of time gaps between consecutive files, in seconds.
    method : {"mode", "mean", "median"}
        Method used to identify outliers.

        - ``"mode"``: values which differ from the most common gap.
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
    if method == "mode":
        reference_value = stats.mode(duration).mode

    elif method == "mean":
        reference_value = np.mean(duration)

    elif method == "median":
        reference_value = np.median(duration)

    threshold = reference_value * (1 + tolerance_percent / 100)
    outlier_index = np.where(gap > threshold)[0]

    return outlier_index


# %%
def viz_spool(spool, method="mode", tolerance_percent=20, annotate_gaps=True):
    """
    Visualize spool timing statistics.

    Creates two plots:

    1. The gap between consecutive files.
    2. The sample rate evolution through time.

    Detected outliers in file gaps are highlighted and annotated on
    the gap plot.

    Parameters
    ----------
    spool
        A DASCore spool or a pandas DataFrame containing spool contents.
        The contents must include ``time_min`` and ``time_step`` columns.
    method : {"mode", "mean", "median"}, optional
        Method used to identify outlier file gaps.

        - ``"mode"``: values which differ from the most common gap.
        - ``"mean"``: values outside ``tolerance_percent`` of the mean.
        - ``"median"``: values outside ``tolerance_percent`` of the median.
    tolerance_percent : float, optional
        Percentage deviation from the reference gap used to classify
        outliers for the ``"mean"`` and ``"median"`` methods.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis showing sample rate evolution.
    gap : numpy.ndarray
        The gap between consecutive file start times, in seconds.
    outlier_index : numpy.ndarray
        Indices of detected gap outliers.

    Raises
    ------
    TypeError
        If ``spool`` is not a DASCore spool or pandas DataFrame.
    ValueError
        If ``method`` is not a supported outlier detection method.

    Notes
    -----
    File gaps are computed from the difference of between the end-time of one file
    and the start of the next file (minus 1 sample interval)
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

        # %%
    method = "mean"
    tolerance_percent = 20
    method = method.lower()

    filestart = df["time_min"].to_numpy()
    fileend = df["time_max"].to_numpy()

    duration = (fileend - filestart) / np.timedelta64(1, "s")
    gap = (filestart[1:] - fileend[:-1]) / np.timedelta64(1, "s")
    outlier_index = _find_outliers(gap, duration, method, tolerance_percent)

    # %% Plot

    tick_map = [
        [0.000001, "1 μs"],
        [0.00001, "10 μs"],
        [0.0001, "100 μs"],
        [0.001, "1 ms"],
        [0.01, "10 ms"],
        [0.1, "100 ms"],
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
    ticks = np.asarray([x[0] for x in tick_map])
    ticklabels = np.asarray([x[1] for x in tick_map])

    mosaic = [["top", "top"], ["middle", "middle"], ["bottom_left", "bottom_right"]]

    _, axs = plt.subplot_mosaic(mosaic, layout="constrained", figsize=(10, 12))

    ############################
    # Plot file-time differences
    ax = axs["top"]
    ax.semilogy(filestart[:-1], gap, ".")

    if (len(outlier_index) < 50) & annotate_gaps:
        for i in outlier_index:
            txt = str(np.timedelta64(int(gap[i]), "s").item())
            txt = "  " + str(filestart[i])[:19]
            ax.text(
                filestart[i],
                gap[i],
                txt,
                rotation=45,
                ha="left",
                va="center",
                fontsize=8,
                rotation_mode="anchor",
            )
    else:
        ax.semilogy(filestart[outlier_index], gap[outlier_index], "r.")
        pass

    ax.set_yticks(ticks, labels=ticklabels)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid("on")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("Gap Between Files")
    ax.set_title("File Gaps")

    ####################################
    # %% Plot file duration
    ax = axs["middle"]
    ax.semilogy(filestart, duration, ".")

    minlim = 10 ** np.floor(np.log10(duration.min()))
    maxlim = 10 ** (np.ceil(np.log10(duration.max())) + 1)
    use = (minlim <= ticks) & (ticks <= maxlim)
    ax.set_yticks(ticks[use], labels=ticklabels[use])

    ax.tick_params(axis="x", labelrotation=90)
    ax.grid("on")
    ax.set_ylabel("Filetime Difference")
    ax.tick_params(axis="x", labelrotation=90)
    # ax.grid("on")
    ax.set_ylabel("File Duration")
    ax.set_title("File Durations")

    ####################################
    # %% Plot gap histogram
    # refine resolution
    gap = gap[gap > 0]
    minlim = 10 ** (np.floor(np.log10(gap.min())) - 1)
    maxlim = 10 ** (np.ceil(np.log10(gap.max())) + 1)
    use = (minlim <= ticks) & (ticks <= maxlim)
    fine_ticks = np.geomspace(minlim, maxlim, num=31)

    ax = axs["bottom_left"]
    ax.hist(gap, bins=fine_ticks, edgecolor="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks[use], labels=ticklabels[use])
    ax.set_xlim((minlim, maxlim))
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title("File Gaps")

    ####################################
    # % Plot file duration histogram
    minlim = 10 ** (np.floor(np.log10(duration.min())))
    maxlim = 10 ** (np.ceil(np.log10(duration.max())) + 1)
    use = (minlim <= ticks) & (ticks <= maxlim)
    fine_ticks = np.geomspace(minlim, maxlim, num=31)

    ax = axs["bottom_right"]
    ax.hist(duration, bins=fine_ticks, edgecolor="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks[use], labels=ticklabels[use])
    ax.set_xlim((minlim, maxlim))
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title("File Duration")

    # %%
    return axs, duration, gap, outlier_index


# %%
if __name__ == "__main__":
    from dascore.utils.hdf5 import HDFPatchIndexManager

    tmpfile = Path(
        r"O:\Staff\andreasw\Dev\FibreEyes\Aurland\_dascore_index_Aurland.hdf5"
    )

    tmpfile = Path(
        r"C:\Users\andreasw\Downloads\Spool_Visualisation\_dascore_index_Hoyanger.hdf5"
    )

    tmpfile = Path(
        r"C:\Users\andreasw\Downloads\Spool_Visualisation\_dascore_index_Aurland.hdf5"
    )

    df = HDFPatchIndexManager(tmpfile).get_index(time_min="2025-01-16T14:00:00")
    dummy = viz_spool(df, method="mode", tolerance_percent=20, annotate_gaps=True)


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
