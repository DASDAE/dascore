"""
QT compatibility layer.

Simply uses qtpy for now, but still important to funnel all qt stuff
through this module to make it easier to change in the future.
"""
from qtpy import QtCore as qc  # NOQA
from qtpy import QtGui as qg  # NOQA
from qtpy import QtWidgets as qw  # NOQA
