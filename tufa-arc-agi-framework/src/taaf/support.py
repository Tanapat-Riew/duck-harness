"""Generic support utilities for TAAF."""

import json
import os
import pickle
from pathlib import Path, PurePath
from typing import IO, Any


def atomic_json_dump(obj: Any, path: Path) -> None:
    """Write ``obj`` as JSON to ``path`` via tempfile + os.replace.

    A crash mid-write leaves either the previous file or nothing — never a
    partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


class _PortablePathPickler(pickle.Pickler):
    """Pickler that writes every ``PurePath`` as a platform-neutral ``Path``.

    A concrete path pickles by its flavour class, so a bundle built on Windows
    embeds ``WindowsPath`` and unpickling it on Linux raises
    ``NotImplementedError: cannot instantiate 'WindowsPath' on your system``
    (and vice versa). Rewriting the reduction to ``pathlib.Path`` with a POSIX
    string means the value reconstructs as whatever flavour the *reader* runs
    -- still a concrete, usable path on both sides.
    """

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, PurePath):
            return (Path, (obj.as_posix(),))
        return NotImplemented


def portable_pickle_dump(obj: Any, file: IO[bytes]) -> None:
    """Pickle ``obj`` to ``file`` so embedded paths cross platforms."""
    _PortablePathPickler(file, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)


def atomic_pickle_dump(obj: Any, path: Path, *, portable_paths: bool = False) -> None:
    """Write ``obj`` as a pickle to ``path`` via tempfile + os.replace.

    Same crash-safety guarantee as ``atomic_json_dump``.

    Set ``portable_paths`` when the pickle will be read on another platform --
    a deployment bundle, for instance. See :class:`_PortablePathPickler`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        if portable_paths:
            portable_pickle_dump(obj, f)
        else:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
