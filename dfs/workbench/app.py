"""
Simple module for handling Qt global app.
"""
import sys
from pathlib import Path

APP = {}
icon_path = Path(__file__).parent / "icons"
from .qcompat import qw


def get_app():
    """
    Gets a reusable app instance for DFS.
    """
    if "app" not in APP:
        import locale

        locale.setlocale(locale.LC_ALL, "C")
        APP["app"] = qw.QApplication(sys.argv)

    return APP["app"]
