"""Regression guard for editable-repo discovery in the deployment snapshotter.

`taaf.deploy._discover_editable_repos` converts each editable distribution's
`direct_url.json` URL into a filesystem path. It used to slice `file://` off
the front, which is wrong on Windows: `file:///D:/x` becomes `/D:/x`, which
`Path` treats as a *drive-relative* path and resolves against the cwd. Every
editable repo then walked up to whichever repo the cwd happened to sit inside
and deduped to a single entry -- silently shipping a Kaggle/Slurm source
bundle with `tufa-arc-agi-framework` missing, so the worker could not import
`taaf` to unpickle the benchmark.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import taaf.deploy


def test_both_editable_repos_are_discovered():
    """The launcher venv has two editable repos; both must be snapshotted."""
    discovered = {path.name for path in taaf.deploy._discover_editable_repos()}
    assert "ARC3-Inference" in discovered
    assert "tufa-arc-agi-framework" in discovered


def test_discovered_repo_paths_exist_and_hold_a_pyproject():
    """A repo root that does not exist means the URL conversion is broken."""
    for repo in taaf.deploy._discover_editable_repos():
        assert repo.is_dir(), f"discovered repo root does not exist: {repo}"
        assert (repo / "pyproject.toml").is_file(), f"no pyproject.toml in {repo}"


def test_file_url_conversion_survives_a_windows_drive_letter():
    """`url2pathname` keeps the drive absolute where a naive slice does not.

    This is the exact conversion `_discover_editable_repos` performs. The old
    `url[len("file://"):]` form produced a drive-relative path on Windows.
    """
    url = "file:///D:/Projects/duck-harness/tufa-arc-agi-framework"
    converted = Path(url2pathname(urlparse(url).path))
    naive = Path(url[len("file://") :])
    assert converted.is_absolute()
    if converted.drive:
        # Windows: the naive form drops the separator after the drive colon,
        # which is what made it resolve against the cwd.
        assert str(naive) != str(converted)
