
## Building the documentation

The documentation can be built using the script called "make_docs.py" in the scripts directory.
If you have followed the [development installation instructions](dev_install.md), all the required
dependencies should be installed. You will also need to install `pandoc` using conda or the
[offical installation](https://pandoc.org/installing.html).

<!--pytest-codeblocks:skip-->
```bash
python scripts/make_docs.py
```

The docs can then be accessed by double-clicking on the newly created html index at docs/_build/html/index.html.

## Contributing to the documentation

The documentation is primarily done in markdown but easily converted to jupyter notebooks using
[jupytext](https://github.com/mwouts/jupytext).

Markdown files should go into docs/markdown. If the markdown file contains python code which should be
tested (most python code should be tested), the file should go in docs/markdown/executable. The
executable code uses the [myst markdown flavor](https://myst-parser.readthedocs.io/en/latest/).
For an example see the [intro page source code](https://github.com/DASDAE/dascore/blob/master/docs/markdown/executable/intro.md).
