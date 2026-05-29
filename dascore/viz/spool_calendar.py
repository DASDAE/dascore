"""Module for visualizing spool availability as calendars"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


# %%
def _build_calendar_matrix(df, starttime=None, endtime=None, method="gap"):
    """
    Build a calendar matrix from spool availability.

    Parameters
    ----------
    spool
        A  pandas DataFrame containing spool contents.
    starttime
        The first day to include in the calendar. If None, use the first day
        with available data.
    endtime
        The day after the last day to include in the calendar. If None, use the
        last day with available data.
    method : {"gap", "percent", "number"}, optional
        Method used to calculate each calendar entry.

        - ``"gap"``: total missing data per day, in seconds.
        - ``"percent"``: percentage of the day with available data.
        - ``"number"``: number of files overlapping each day.

    Returns
    -------
    calendar_matrix : numpy.ndarray
        A 2D array with one row per month and one column per day of month.
        Entries without corresponding dates are NaN.
    first_day : datetime.date
        The first day represented by the calendar matrix.

    Raises
    ------
    TypeError
        If ``spool`` is not a DASCore spool or pandas DataFrame.
    ValueError
        If ``method`` is not one of ``"gap"``, ``"percent"``, or ``"number"``.
    """
    first_data_day = df["time_min"].min().date()
    last_data_day = df["time_max"].max().date()

    if starttime is None:
        first_day = first_data_day
    else:
        first_day = np.datetime64(starttime, "D").item()

    if endtime is None:
        last_day = last_data_day
    else:
        last_day = np.datetime64(endtime, "D").item()

    days = np.arange(first_day, last_day, dtype="datetime64[D]")
    days = pd.to_datetime(days)

    filestarts = df["time_min"].dt.floor("ms")
    fileends = df["time_max"].dt.floor("ms")
    step = df["time_step"].dt.floor("ms")

    # allocate calendar matrix size
    n_months = (
        (last_day.year - first_day.year) * 12 + (last_day.month - first_day.month) + 1
    )

    calendar_matrix = np.full((n_months, 31), np.nan)

    current_month = None
    day_in_ns = np.timedelta64(np.timedelta64(1, "D"), "ns")
    row = -1
    for i, this_day in enumerate(days):
        if (this_day.date() < first_data_day) | (this_day.date() > last_data_day):
            continue

        month_key = (this_day.year, this_day.month)
        if month_key != current_month:
            current_month = month_key
            row += 1
        col = this_day.day - 1

        next_day = this_day + pd.Timedelta(days=1)

        # idx = (filestarts >= this_day) & (filestarts < next_day)

        # this syntax captures overlaps
        use = (df["time_min"] < next_day) & (df["time_max"] > this_day)

        t0 = np.clip((filestarts[use] - this_day).to_numpy(), min=0)
        t1 = np.clip((fileends[use] - this_day).to_numpy(), max=day_in_ns)
        dt = np.clip((t1 - t0), min=0, max=day_in_ns)

        seconds_with_data = (np.sum(dt) + np.sum(step[use].iloc[:-1])).total_seconds()

        if method.upper() == "PERCENT":  # percent of day
            calendar_matrix[row, col] = seconds_with_data / 86400 * 100

        elif method.upper() == "GAP":  # missing seconds
            calendar_matrix[row, col] = np.max((86400 - seconds_with_data, 0.1))

        elif method.upper() == "NUMBER":  # of files
            calendar_matrix[row, col] = sum(use)

    return calendar_matrix, first_day


# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def calendar(spool, starttime=None, endtime=None, method="gap"):
    """
    Plot spool availability as a calendar.

    The calendar contains one row per month and one column per day of month.
    Each cell summarizes the data availability for one day according to
    ``method``.

    Parameters
    ----------
    spool
        A DASCore spool or a pandas DataFrame containing spool contents. The
        contents must include ``time_min``, ``time_max``, and ``time_step``
        columns.
    starttime
        The first day to include in the calendar. If None, use the first day
        with available data.
    endtime
        The day after the last day to include in the calendar. If None, use the
        last day with available data.
    method : {"gap", "percent", "number"}, optional
        Method used to calculate each calendar entry.

        - ``"gap"``: total missing data per day, in seconds.
        - ``"percent"``: percentage of the day with available data.
        - ``"number"``: number of files overlapping each day.

    Returns
    -------
        Axes object
    """
    if hasattr(spool, "get_contents"):
        df = spool.get_contents()
    elif isinstance(spool, pd.DataFrame):
        df = spool
    else:
        raise TypeError(
            f"Expected a DASCore spool or DataFrame, got {type(spool).__name__}"
        )

    valid_options = ["gap", "percent", "number"]
    if method.lower() not in valid_options:
        raise ValueError(f"Unknown METHOD: '{method}'. Options are {valid_options}")

    calendar_matrix, first_day = _build_calendar_matrix(df, starttime, endtime, method)

    if method.upper() == "PERCENT":  # percent of day
        cmap = "RdYlGn"
        cbar_label_str = "Availability [%]"
        norm = None

    elif method.upper() == "GAP":  # missing seconds
        cbar_label_str = "Total Data Gap [sec]"
        cmap = "RdYlGn_r"
        norm = "log"

    elif method.upper() == "NUMBER":  # of files
        cmap = "YlGn"
        cbar_label_str = "Number of Files"
        norm = None

    cmap = plt.get_cmap(cmap)
    cmap.set_bad("gray")
    cmap.set_under("white")
    cmap.set_over("red")

    lim = [
        np.nanmin(calendar_matrix[calendar_matrix > 0]),
        np.nanmax(calendar_matrix[calendar_matrix > 0]),
    ]

    nan_mask = np.isnan(calendar_matrix)
    zero_mask = calendar_matrix == 0

    # Replace zeros temporarily with tiny positive number
    matrix = calendar_matrix.copy()
    matrix[zero_mask] = 0.1

    # Mask NaNs only
    masked_array = np.ma.array(matrix, mask=nan_mask)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    him = ax.pcolormesh(masked_array, cmap=cmap, norm=norm, vmin=lim[0], vmax=lim[1])
    ax.grid("on", color="silver", lw=0.3)
    ax.set_xlabel("Day of Month")

    # beautify x- and y-ticks to be in the middle of each cell
    labels = []
    for m in range(calendar_matrix.shape[0]):
        txt = (first_day.replace(day=1) + relativedelta(months=m)).strftime("%Y-%b")
        labels.append(txt)

    xticks = np.arange(0, 32)
    yticks = np.arange(0, masked_array.shape[0] + 1)

    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))

    # 2. Hide major tick labels
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # 3. Set minor ticks at the midpoints (e.g., 0.5, 1.5...)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(xticks + 0.5))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(yticks + 0.5))

    # 4. Set the actual labels on the minor ticks
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(xticks + 1))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(labels))

    # 5. Remove the actual "dash" mark for minor ticks so only text shows
    ax.tick_params(axis="both", which="minor", length=0)

    ax.yaxis.set_inverted(True)
    cb = fig.colorbar(him)  # , extend='min')
    cb.ax.set_ylabel(cbar_label_str)

    if method.upper() == "GAP":
        him.set_clim([0.1, 86400])  # full-day gaps are thusly in white
        tick_map = [
            [1, "1 sec"],
            [10, "10 sec"],
            [60, "1 min"],
            [10 * 60, "10 min"],
            [3600, "1 hour"],
            [6 * 3600, "6 hours"],
            [86400, "24 hours"],
        ]

        tick_values = np.array([x[0] for x in tick_map])
        tick_labels = [x[1] for x in tick_map]

        cb.set_ticks(tick_values)
        cb.set_ticklabels(tick_labels)
        cb.minorticks_off()

        cmap.set_under("green")
        cmap.set_over("white")

    return ax


# %%
if __name__ == "__main__":
    from dascore.utils.hdf5 import HDFPatchIndexManager

    tmpfile = Path(
        r"O:\Staff\andreasw\Dev\FibreEyes\Aurland\_dascore_index_Aurland.hdf5"
    )
    tmpfile = Path(
        r"C:\Users\andreasw\Downloads\Spool_Visualisation\_dascore_index_Aurland.hdf5"
    )
    tmpfile = Path(
        r"C:\Users\andreasw\Downloads\Spool_Visualisation\_dascore_index_Hoyanger.hdf5"
    )

    df = HDFPatchIndexManager(tmpfile).get_index()  # time_min="2026-01-16T14:00:00")
    df = df.sort_values(by="time_min")

    calendar(df, starttime="2025-11-27", endtime=None)


# %%
"""
    filestart = df["time_min"].to_numpy()
    fileend = df["time_max"].to_numpy()

    target_day = np.datetime64('2026-05-25')

    # 3. Find all timestamps on the target day
    mask = (dates >= target_day) & (dates < target_day + np.timedelta64(1, 'D'))
    filtered_dates = dates[mask]


    #%%
    dt = df["time_step"].to_numpy()
    fsamp = np.timedelta64(1, "s") / dt
    gap = np.diff(filestart) / np.timedelta64(1, "s")
"""
