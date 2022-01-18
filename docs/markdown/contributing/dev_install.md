
# Installing DASCore for Development

The following steps are needed to set up `DASCore` for development:

## 1. Clone DASCore

<!--pytest-codeblocks:skip-->
```bash
git clone https://github.com/dasdae/dascore
cd dascore
```

## 2. Pull tags

Make sure to pull all of the latest git tags.

**NOTE:** You may need to do this periodically to keep tags in sync.

<!--pytest-codeblocks:skip-->
```bash
git pull origin master --tags
```

## 3. Create a virtual environment (optional)

Create and activate a virtual environment so DASCore will not mess with the base (or system) python installation.

If you are using [Anaconda](https://www.anaconda.com/), simply use the environment provided:

<!--pytest-codeblocks:skip-->
```bash
conda env create -f environment.yml
conda activate dascore
```

## 4. Install DASCore in development mode

<!--pytest-codeblocks:skip-->
```bash
pip install -e .[test,docs]
```

## 5. Setup pre-commit hooks

dascore uses several [pre-commit](https://pre-commit.com/) hooks to ensure the code stays tidy. Please install and use them!

<!--pytest-codeblocks:skip-->
```bash
pre-commit install -f
```
