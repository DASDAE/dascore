"""
QT compatibility layer.

Simply uses qtpy for now, but still important to funnel all qt stuff
through this module to make it easier to change in the future.
"""
import qtpy

from qtpy import QtCore as qc
from qtpy import QtWidgets as qw
from qtpy import QtGui as qg
