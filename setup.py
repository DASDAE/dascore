"""
Setup.py patch
"""

import site
import sys

import setuptools

if __name__ == "__main__":
    # This is needed for editable install on CI, see
    # https://github.com/pypa/pip/issues/7953
    site.ENABLE_USER_SITE = "--user" in sys.argv[1:]
    setuptools.setup()
