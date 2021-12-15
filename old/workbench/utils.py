"""
Utilties for visualizations
"""
from dascore.workbench import get_app


class QTBase:
    """Basse class for QT stuff in workbench"""

    _showing = False


class ShowOnCall:
    """
    A qt subclass which calls the show method with __call__
    """

    def __call__(self, show=True, exec=True):
        """Show the class's content."""
        try:
            app = get_app()
            if show:
                self._showing = True
                self.show()
            if exec:
                # If the application is not yet active
                if app.applicationState() != 4:
                    app.exec()
            return self
        except Exception:
            if getattr(self, "_debug", False):
                from PyQt5.Qt import pyqtRemoveInputHook, pyqtRestoreInputHook

                pyqtRemoveInputHook()
                breakpoint()  # NOQA
                pyqtRestoreInputHook()
