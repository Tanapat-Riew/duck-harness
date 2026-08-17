"""Deployment pickles must survive being read on a different platform.

A concrete `pathlib` path pickles by its flavour class, so a Kaggle bundle
built on Windows embeds `WindowsPath`. Loading that on Kaggle's Linux worker
raises `NotImplementedError: cannot instantiate 'WindowsPath' on your system`
before the run can start -- which is exactly what happened in production.
"""
from __future__ import annotations

import io
import pathlib
import pickle

import taaf.support


def _round_trip(obj):
    buffer = io.BytesIO()
    taaf.support.portable_pickle_dump(obj, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def test_a_concrete_path_pickles_as_the_flavour_neutral_factory():
    """The stream must name `pathlib.Path`, never `WindowsPath`/`PosixPath`."""
    payload = _round_trip({"job_dir": pathlib.Path("runs") / "abc"})
    assert b"WindowsPath" not in payload
    assert b"PosixPath" not in payload
    assert b"Path" in payload


def test_a_path_survives_the_round_trip_as_a_usable_path():
    restored = pickle.loads(_round_trip(pathlib.Path("runs") / "abc" / "d.txt"))
    assert isinstance(restored, pathlib.Path)
    # A concrete path on either platform, so callers can still use it.
    assert restored.name == "d.txt"
    assert restored.as_posix() == "runs/abc/d.txt"


class _Target:
    """Module level so pickle can resolve it by reference, like a real target."""

    def __init__(self):
        self.repos = [pathlib.Path("a") / "b"]
        self.mapping = {"out": pathlib.Path("c")}


def test_paths_nested_in_containers_and_objects_are_rewritten():
    payload = _round_trip(_Target())
    assert b"WindowsPath" not in payload
    assert b"PosixPath" not in payload
    restored = pickle.loads(payload)
    assert restored.repos[0].as_posix() == "a/b"
    assert restored.mapping["out"].as_posix() == "c"


def test_pure_paths_are_also_normalised():
    payload = _round_trip(pathlib.PureWindowsPath("D:/x/y"))
    assert b"PureWindowsPath" not in payload
    assert pickle.loads(payload).as_posix().endswith("x/y")


def test_non_portable_dump_still_embeds_the_local_flavour():
    """The default path is unchanged -- portability is opt-in per call site."""
    payload = pickle.dumps(pathlib.Path("x"), protocol=pickle.HIGHEST_PROTOCOL)
    assert b"WindowsPath" in payload or b"PosixPath" in payload
