"""
Main window for WorkBench.
"""

from pyqtgraph.dockarea import DockArea

from dascore.utils.downloader import fetch
from dascore.workbench.qcompat import qg, qw
from dascore.workbench.utils import QTBase, ShowOnCall

from .app import get_app


class WorkBench(qw.QMainWindow, ShowOnCall, QTBase):
    """Main Window."""

    def __init__(self, data=None, parent=None):
        """Initializer."""
        get_app()  # ensure app is created before creating widgets
        super().__init__(parent)
        self._prep_window()
        # self.model =
        # create menus
        self._menus = {}  # dict containing dict of menus
        self._create_menus()
        self._dock_area = DockArea()
        self.setCentralWidget(self._dock_area)

    def _prep_window(self):
        hammer_path = fetch("hammer_icon")
        self.setWindowIcon(qg.QIcon(str(hammer_path)))
        self.setMinimumSize(500, 700)
        self.resize(1000, 700)
        self.setWindowTitle("dascore WorkBench")

    def _create_menus(self):
        """Create the menus."""
        self._menu_bar = self.menuBar()
        self._menu_bar.setMinimumHeight(30)
        font = self._menu_bar.font()
        font.setPointSize(14)
        self._menu_bar.setFont(font)
        self._create_file_menu()
        self._menus["tools"] = qw.QMenu("&Tools", self)

        for name, menu in self._menus.items():
            self._menu_bar.addMenu(menu)

    def _create_file_menu(self):
        """Create the options under the file menu."""
        file_menu = qw.QMenu("&File", self)
        font = file_menu.font()
        font.setPointSize(14)
        file_menu.setFont(font)
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
    Base class for implementing tools in dascore.
    """

    def __init__(self, workbench: WorkBench):
        pass

    def create_toolbar(self) -> qw.QToolBar:
        """When called this should return the tool bar."""

    @property
    def model(self):
        """Return the data model."""


if __name__ == "__main__":
    win = WorkBench()
    win()
