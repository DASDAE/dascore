"""
Utilties for vizualizations
"""
from fios.workbench import get_app


class QTBase:
    """Basse class for QT stuff in workbench"""

    _showing = False


class ShowOnCall:
    def __call__(self, show=True, exec=True):
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
                breakpoint()
