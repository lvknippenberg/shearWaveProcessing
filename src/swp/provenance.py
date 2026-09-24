"""Provenance stamp for every output: which code, which zea, which configuration produced it.

Why: on 2026-09-17 the Linux server reproduced a folder with displacement correlation 0.21 against
Windows, and the cause turned out to be the **zea commit**, not the backend. Nothing in the output
files recorded either. Every HDF5 / CSV this package writes now carries:

* ``swp_git_commit`` / ``swp_git_branch`` / ``swp_git_dirty`` - this repository's state
  (``dirty`` counts and lists the modified tracked files, and ``swp_git_diff_sha256`` hashes the
  uncommitted diff, so an uncommitted run is visible and its exact state identifiable);
* ``zea_version`` / ``zea_git_commit`` / ``zea_git_dirty`` - read from the installed distribution
  metadata and, for an editable install, from that checkout's git - **without importing zea**
  (importing it initialises keras/torch, ~10 s);
* ``config_sha256`` + ``config_json`` - a canonical (sorted-key) JSON of the configuration, so two
  outputs made with the same settings have the same hash whatever the YAML formatting;
* ``created_utc``, ``host``, ``python``, ``numpy``, ``scipy``, ``keras_backend``, ``command``.

Use :func:`stamp_h5` on an open ``h5py.File`` (or a path), or :func:`stamp_text` for CSV/log
headers. :func:`read_h5` reads a stamp back.
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUP = "provenance"


def _git(repo: str, *args: str, strip: bool = True) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() if strip else r.stdout


def _git_state(repo: str, prefix: str) -> dict:
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return {f"{prefix}_git_commit": "unknown"}
    # porcelain lines are "XY path": do not strip, the leading status column can be a space
    status = _git(repo, "status", "--porcelain", "--untracked-files=no", strip=False) or ""
    modified = [ln[3:] for ln in status.splitlines() if ln.strip()]
    out = {
        f"{prefix}_git_commit": commit,
        f"{prefix}_git_branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "",
        f"{prefix}_git_dirty": "",
    }
    if modified:
        shown = ", ".join(modified[:10]) + (f", ... (+{len(modified) - 10})" if len(modified) > 10 else "")
        out[f"{prefix}_git_dirty"] = f"{len(modified)} modified: {shown}"
        # identifies the exact uncommitted state: same hash = same working-tree diff
        diff = _git(repo, "diff", "HEAD", strip=False) or ""
        out[f"{prefix}_git_diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return out


@functools.lru_cache(maxsize=1)
def _static() -> dict:
    """The parts that cannot change within one process (computed once)."""
    out = {"swp_version": _swp_version()}
    out.update(_git_state(_REPO, "swp"))
    out.update(_zea_state())
    import numpy
    out["numpy"] = numpy.__version__
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except ImportError:
        pass
    out["python"] = platform.python_version()
    out["host"] = platform.node()
    return out


def _swp_version() -> str:
    try:
        from swp import __version__
        return __version__
    except ImportError:
        return "unknown"


def _zea_state() -> dict:
    import importlib.metadata as md
    try:
        dist = md.distribution("zea")
    except md.PackageNotFoundError:
        return {"zea_version": "not installed"}
    out = {"zea_version": dist.version}
    try:
        url = json.loads(dist.read_text("direct_url.json") or "{}").get("url", "")
    except (json.JSONDecodeError, TypeError):
        url = ""
    if url.startswith("file:///"):
        from urllib.parse import unquote
        path = unquote(url[len("file:///"):])
        out["zea_source"] = path
        state = _git_state(path, "zea")
        if state.get("zea_git_commit") != "unknown":
            out.update(state)
    return out


def config_hash(config: Any) -> tuple[str, str]:
    """(sha256, canonical JSON) of a configuration (dict/list/dataclass/scalars)."""
    txt = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest(), txt


def _jsonable(x: Any) -> Any:
    import dataclasses
    import numpy as np
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def provenance(config: Any = None, extra: Optional[dict] = None) -> dict:
    """The full stamp as a flat ``{str: str}`` dict."""
    out = dict(_static())
    out["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    out["keras_backend"] = os.environ.get("KERAS_BACKEND", "")
    out["command"] = " ".join([os.path.basename(sys.argv[0] or "python")] + sys.argv[1:])
    if config is not None:
        out["config_sha256"], out["config_json"] = config_hash(config)
    for k, v in (extra or {}).items():
        out[str(k)] = v if isinstance(v, str) else json.dumps(_jsonable(v))
    return {k: str(v) for k, v in out.items()}


def stamp_h5(target, config: Any = None, extra: Optional[dict] = None, group: str = GROUP) -> dict:
    """Write the stamp as string attributes of ``/<group>`` in an HDF5 file.

    ``target`` is an open ``h5py.File``/``Group`` or a path (opened in append mode). An existing
    stamp group is replaced, so re-stamping after an in-place update is safe.
    """
    import h5py
    stamp = provenance(config, extra)
    if isinstance(target, (str, os.PathLike)):
        with h5py.File(target, "a") as f:
            _write(f, group, stamp)
    else:
        _write(target, group, stamp)
    return stamp


def _write(f, group, stamp):
    if group in f:
        del f[group]
    g = f.create_group(group)
    for k, v in stamp.items():
        g.attrs[k] = v


def read_h5(path, group: str = GROUP) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        if group not in f:
            return {}
        return {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in f[group].attrs.items()}


def stamp_text(config: Any = None, extra: Optional[dict] = None, prefix: str = "# ") -> str:
    """The stamp as comment lines for the top of a CSV or log (``config_json`` left out - the
    hash identifies it and the JSON can be long)."""
    stamp = provenance(config, extra)
    stamp.pop("config_json", None)
    return "".join(f"{prefix}{k}: {v}\n" for k, v in stamp.items())
