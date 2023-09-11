"""Core module for rsf file format support."""
from dascore.io.core import FiberIO


class rsfV1(FiberIO):
    """An IO class supporting version 1 of the rsf format."""

    # you must specify the format name using the name attribute
    name = "rsf"
    # you can also define which file extensions are expected like so.
    # this will speed up DASCore's automatic file format determination.
    preferred_extensions = ("rsf",)
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named JingleV2.
    version = "1"

    def read(self, path, rsf_param=1, **kwargs):
        """
        Read should take a path and return a patch or sequence of patches.

        It can also define its own optional parameters, and should always
        accept kwargs. If the format supports partial reads, these should
        be implemented as well.
        """

    def get_format(self, path):
        """
        Used to determine if path is a supported jingle file.

        Returns a tuple of (format_name, file_version) if the file is a
        supported jingle file, else return False or raise a
        dascore.exceptions.UnknownFiberFormat exception.
        """

    def scan(self, path):
        """
        Used to get metadata about a file without reading the whole file.

        This should return a list of
        [`PatchAttrs`](`dascore.core.attrs.PatchAttrs`) objects
        from the [dascore.core.attrs](`dascore.core.attrs`) module, or a
        format-specific subclass.
        """

    def write(self, patch, path, **kwargs):
        """Write a patch or spool back to disk in the jingle format."""
