# Adding Support for a File Format

Several steps are required to add support for a new file format. To demonstrate, imagine adding support for a
format called `jingle` which conventionally uses a file extension of `jgl`.

## Adding a New IO Module

First, we create a new io module called 'jingle' in DASCore's io module (dascore/io/jingle).
Make sure there is a __init__.py file in this module whose docstring describes the format.

contents of `dascore/io/jingle/__init__.py`
```python
"""
Jingle format support module.

Jingle is a really cool new format which supports all the "bells" and whistles
people need for working with das data.

Examples
--------
import dascore as dc

jingle = dc.read('path_to_file.jgl')
"""
```

Next, create a `core.py` file in the new module (dascore/io/jingle/core.py). Start by creating
a class called `JingleIO`,  which subclasses `dascore.io.core.FiberIO, and implement the supported
methods.

Contents of `dascore/io/jingle/core.py`

```python
"""
Core module for jingle file format support.
"""
import dascore.exceptions
from dascore.io.core import FiberIO


class JingleIO(FiberIO):
    """
    An IO class supporting the jingle format.
    """
    # you must specify the format name using the name attribute
    name = 'jingle'
    # you can also define which file extensions are expected like so.
    # this will speed up determining the formats of files if not specified.
    preferred_extensions = ('jgl',)

    def read(self, path, jingle_param=1, **kwargs):
        """
        Read should take, at least, a path and return a patch or sequence of patches.

        It can also define its own parameters, and should always accept kwargs.
        """

    def is_format(self, path):
        """
        Used to determine if path is a supported jingle file.

        Returns a tuple of (format_name, version_number) if the file is a
        supported jingle file, else return False or raise a
        dascore.exceptions.UnknownFiberFormat.
        """

    def scan(self):
        """
        Used to gather metadata about a file without reading in the whole file.

        This should return a list of dictionaries with the keys/types specified
        in dascore.constants.SUMMARY_KEYS.
        """

    def write(self, patch, path, **kwargs):
        """
        Write a patch or spool back to disk in the jingle format.
        """
```

All of the 4 methods are optional; some formats will only support reading,
others only writing. `is_format` is fairly important otherwise the format
wont be auto-detectable.


## Writing Tests

Next we need to write tests for the format (you weren't thinking of skipping
this step were you!?). The tests should go in: `test/test_io/test_jingle.py`.
The hardest part is finding a *small* (typically no more than 1-2 mb) file to
include in the test suite. Assuming you have found, or manufactured, such
as file, it should go into [dasdae's data repo](https://github.com/DASDAE/test_data).
Simply clone the repo, add you file format, and push back to master or open a
PR on a separate branch.

Next, add your file to dascore's data registry (dascore/data_registry.txt).
You will have to get the sha256 hash of your test file, for that you can simply
use [Pooch's hash_file function](https://www.fatiando.org/pooch/latest/api/generated/pooch.file_hash.html),
and you can create the proper download url using the other entries as examples.

The name, hash, and url might look  something like this:

```
jingle_test_file.jgl
12e087d2c1cd08c9afd18334e17e21787be0b646151b39802541ee11a516976a
https://github.com/dasdae/test_data/raw/master/das/jingle_test_file.jgl
```

Then your test file might look like this:

contents of tests/test_io/test_jingle.py
```python

import pytest

import dascore
from dascore.utils.downloader import fetch

class TestJingleIO:
    """Tests for jingle IO format."""
    @pytest.fixture(scope='class')
    def jingle_file_path(self):
        """Return the path to the test jingle file."""
        path = fetch("jingle_test_file.jgl")
        return path

    @pytest.fixture(scope='class')
    def jingle_patch(self, jingle_file_path):
        """Read the jingle test data"""
        return dascore.read(jingle_file_path, format='jingle')

    def test_read(self, jingle_file_path):
        """Ensure the test file can be read."""
        # fetch will download the file (if not already downloaded) and
        # test read without specifying format
        out1 = dascore.read(jingle_file_path)
        # next assert things about the output, maybe len, attributes etc.
        assert len(out1) == 1
        # test read specifying format
        out2 = dascore.read(jingle_file_path, 'jingle')
        ...

    def test_write(self, tmp_path_factory, jingle_patch):
        """Ensure jingle format can be written."""
        out_path = tmp_path_factory.mktemp('jingle') / 'jingle_test.jgl'
        jingle_patch.io.write(jingle_patch, out_path)
        # make sure the new file exists and has a size
        assert out_path.exists()
        assert out_path.stat().size > 0

    def test_scan(self, jingle_file_path):
        """Tests for scanning a jingle file"""
        scan = dascore.scan_file(jingle_file_path)
        assert len(scan) == 1

    def test_is_format(self, jingle_file_path):
        """Tests for automatically determining a jingle format."""
        out = dascore.get_format(jingle_file_path)
        assert out[0].lower() == 'jingle'

```

## Register Plugin

Now that the Jingle format support is implemented and tested, the final step is to
register jingle as an entry point in the setup.cfg. This is done under
options.entry_points in the dascore.plugin.fiber_io namespace. For example,
after adding jingle the setup.cfg section might look like this:

```
[options.entry_points]
dascore.plugin.fiber_io =
    TERRA15 = dascore.io.terra15.core:Terra15Formatter
    PICKLE = dascore.io.pickle.core:PickleIO
    WAV = dascore.io.wav.core:WavIO
    JINGLE = dascore.io.jingle.core:JingleIO
```
