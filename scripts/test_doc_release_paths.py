"""Keep release-code restoration compatible with package-local docs."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(root, *args):
    """Run Git inside an isolated test repository."""
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


class TestReleaseDocumentation:
    """Restoring release code preserves the checkout's authored documentation."""

    def test_restore(self, tmp_path):
        """The actual workflow command restores code without deleting moved docs."""
        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.name", "Test")
        _git(tmp_path, "config", "user.email", "test@example.com")
        package = tmp_path / "dascore"
        package.mkdir()
        code = package / "__init__.py"
        code.write_text("release")
        _git(tmp_path, "add", "dascore")
        _git(tmp_path, "commit", "-m", "Release code")
        release = _git(tmp_path, "rev-parse", "HEAD")
        code.write_text("development")
        page = package / "docs/index.qmd"
        page.parent.mkdir()
        page.write_text("Current authored docs")
        _git(tmp_path, "add", "dascore")
        _git(tmp_path, "commit", "-m", "Package documentation")
        workflow = (
            ROOT / ".github/workflows/build_deploy_stable_docs.yaml"
        ).read_text()
        command = next(
            line.strip()
            for line in workflow.splitlines()
            if line.strip().startswith("git restore")
        )
        args = [part.replace("$tag", release) for part in shlex.split(command)]
        _git(tmp_path, *args[1:])
        assert code.read_text() == "release"
        assert page.read_text() == "Current authored docs"
