"""
Module for filtering patches.
"""

# import warnings
#
# from scipy.signal import iirfilter, sosfilt, zpk2sos
#
# import fios
# from fios.utils.patch import patch_function
#
#
# @patch_function()
# def pass_filter(
#     patch: "fios.Patch",
# ) -> "fios.Patch":
#     """
#
#     Parameters
#     ----------
#     patch
#
#     Returns
#     -------
#
#     """
#
#
# def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
#     """
#     Butterworth-Bandpass Filter.
#
#     Filter data from ``freqmin`` to ``freqmax`` using ``corners``
#     corners.
#     The filter uses :func:`scipy.signal.iirfilter` (for design)
#     and :func:`scipy.signal.sosfilt` (for applying the filter).
#
#     :type data: numpy.ndarray
#     :param data: Data to filter.
#     :param freqmin: Pass band low corner frequency.
#     :param freqmax: Pass band high corner frequency.
#     :param df: Sampling rate in Hz.
#     :param corners: Filter corners / order.
#     :param zerophase: If True, apply filter once forwards and once backwards.
#         This results in twice the filter order but zero phase shift in
#         the resulting filtered trace.
#     :return: Filtered data.
#     """
#     fe = 0.5 * df
#     low = freqmin / fe
#     high = freqmax / fe
#     # raise for some bad scenarios
#     if high - 1.0 > -1e-6:
#         msg = (
#             "Selected high corner frequency ({}) of bandpass is at or "
#             "above Nyquist ({}). Applying a high-pass instead."
#         ).format(freqmax, fe)
#         warnings.warn(msg)
#         return highpass(data, freq=freqmin, df=df,
# corners=corners, zerophase=zerophase)
#     if low > 1:
#         msg = "Selected low corner frequency is above Nyquist."
#         raise ValueError(msg)
#     z, p, k = iirfilter(
#         corners, [low, high], btype="band", ftype="butter", output="zpk"
#     )
#     sos = zpk2sos(z, p, k)
#     if zerophase:
#         firstpass = sosfilt(sos, data)
#         return sosfilt(sos, firstpass[::-1])[::-1]
#     else:
#         return sosfilt(sos, data)
#
#
# def stop_filter(patch: "fios.Patch") -> "fios.Patch":
#     """"""
