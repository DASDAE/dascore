"""
RSF format support module.

Notes
-----
- output has been tested on Madagascar version 3.1-git


Examples
--------
import dascore as dc

# get the path to a random DAS file.
patch = dc.get_example_patch()
patch.io.write("test_out.rsf","rsf",data_path="test_out.rsf")

"""

from .core import RSFV1
