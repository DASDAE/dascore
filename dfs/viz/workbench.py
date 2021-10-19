"""
Main window for WorkBench.
"""
from typing import Optional

import xarray as xr
import pyqtgraph as pg

from dfs.viz.qcompat import qg, qc, qw
from dfs.viz.utils import ShowOnCall, QTBase
from dfs.utils.downloader import fetch
from .app import get_app
from pyqtgraph.dockarea import DockArea


class WorkBench(qw.QMainWindow, ShowOnCall, QTBase):
    """Main Window."""

    def __init__(self, data: Optional[xr.DataArray] = None, parent=None):
        """Initializer."""
        get_app()  # ensure app is created before creating widgets
        super().__init__(parent)
        self._size_window()
        # create menus
        self._menus = {}  # dict containing dict of menus
        self._create_menus()
        self._dock_area = DockArea()
        self.setCentralWidget(self._dock_area)

    def _size_window(self):
        hammer_path = fetch("hammer_icon")
        self.setWindowIcon(qg.QIcon(str(hammer_path)))
        self.setMinimumSize(500, 700)
        self.resize(1000, 700)
        self.setWindowTitle("DFS WorkBench")

    def _create_menus(self):
        """Create the menus."""
        self._menu_bar = self.menuBar()
        self._menu_bar.setMinimumHeight(30)
        self._create_file_menu()
        self._menus["tools"] = qw.QMenu("&Tools", self)

        for name, menu in self._menus.items():
            self._menu_bar.addMenu(menu)

    def _create_file_menu(self):
        """Create the options under the file menu."""
        file_menu = qw.QMenu("&File", self)
        action = qw.QAction("&Quit", file_menu)
        file_menu.addAction(action)
        action.triggered.connect(self.close)
        self._menus["file"] = file_menu

    def _add_tools(self):
        """Add a tool menu to menu bar."""

    def register_tool(self, tool_class):
        """Pass"""

    def remove_tool(self, name):
        """Remove a tool."""


class BenchTool:
    """
    Base class for implementing tools in DFS.
    """

    def __init__(self, workbench: WorkBench):
        pass

    def create_toolbar(self) -> qw.QToolBar:
        """When called this should return the tool bar."""

    @property
    def model(self):
        """Return the data model."""


class Time:
    pass


if __name__ == "__main__":
    win = WorkBench()
    win()
    breakpoint()
