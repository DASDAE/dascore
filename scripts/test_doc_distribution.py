"""Check the documentation contract in built source and wheel distributions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    """Build an isolated source copy containing representative generated pollution."""
    work = tmp_path_factory.mktemp("doc-distribution")
    source = work / "source"
    source.mkdir()
    shutil.copytree(
        ROOT / "dascore",
        source / "dascore",
        ignore=shutil.ignore_patterns("__pycache__", "_site", ".quarto", "api"),
    )
    for name in ("pyproject.toml", "MANIFEST.in", "readme.md"):
        shutil.copy2(ROOT / name, source / name)
    for name in (
        "api/generated.qmd",
        "_site/generated.html",
        "tutorial/page_files/figure.png",
        "tutorial/export.ipynb",
    ):
        path = source / "dascore/docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated output must not ship")
    env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DASCORE": "0.0.1"}
    dist = work / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist), str(source)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    sdist = next(dist.glob("*.tar.gz"))
    unpacked = work / "unpacked"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dist),
            str(next(unpacked.iterdir())),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return sdist, next(dist.glob("*.whl")), work


class TestDocumentationDistribution:
    """Built artifacts contain the authored corpus and an executable lookup."""

    def test_contents(self, distributions):
        """Both artifact types include prose and exclude generated documentation."""
        sdist, wheel, _ = distributions
        with tarfile.open(sdist) as archive:
            source_names = {name.split("/", 1)[-1] for name in archive.getnames()}
        with ZipFile(wheel) as archive:
            wheel_names = set(archive.namelist())
        for names in (source_names, wheel_names):
            assert "dascore/docs/tutorial/patch.qmd" in names
            assert "dascore/docs/recipes/tunnel_inventory.qmd" in names
            assert not any("/docs/_static/" in name for name in names)
            assert not any(
                name.endswith((".png", ".svg", ".jpg", ".jpeg"))
                for name in names
                if "/docs/" in name
            )
            assert not any(
                "generated." in name or "page_files" in name or name.endswith(".ipynb")
                for name in names
            )
        assert "dascore/cli.py" in wheel_names

    def test_wheel_lookup(self, distributions):
        """The wheel's own sources and metadata drive the CLI outside a checkout."""
        _, wheel, work = distributions
        installed = work / "wheel"
        with ZipFile(wheel) as archive:
            archive.extractall(installed)
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(installed)!r}); "
            "import dascore as dc; "
            f"assert dc.__file__.startswith({str(installed)!r}); "
            "assert dc.__version__ == '0.0.1'; "
            f"dc.set_config(docs_cache_dir={str(work / 'cache')!r}); "
            "from dascore.cli import main; "
            "assert main(['doc']) == 0; "
            "assert main(['doc', 'Patch.viz.waterfall']) == 0; "
            "assert main(['doc', 'recipes/tunnel_inventory']) == 0; "
            "assert main(['doc', 'index']) == 0"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "waterfall(" in result.stdout
        assert "DASCore" in result.stdout
        assert (
            "https://raw.githubusercontent.com/DASDAE/dascore/v0.0.1/"
            "dascore/docs/_static/tunnel_deployment.svg"
        ) in result.stdout
        manifest = json.loads((work / "cache/0.0.1/manifest.json").read_text())
        assert manifest["identity"]["version"] == "0.0.1"
