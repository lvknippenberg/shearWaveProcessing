"""Files, hashes and the per-folder state of the manual passive study.

Everything lives in ``<folder>/output/swp_passive_manual/``; the earlier passive outputs
(``swp_passive/``, ``mlines/``) are only ever read (to pre-load old lines).

Each file has exactly ONE writer, so the interactive session and any number of workers can run at
the same time without clobbering each other:

=====================  ==========  ============================================================
file                   written by  content
=====================  ==========  ============================================================
general.json/.npz      GUI         general M-line (buffer-4 coordinates) + how it was drawn
windows.json           worker      detected event windows, keyed by the general line's hash
events.json            GUI         per-window M-line records, keyed by the windows' hash
event<i>_mline.npz     GUI         the line used for window i (buffer-4 coordinates)
processed.json         worker      per window: hash of the line it was computed from + st hash
st_win<i>.npz          worker      the five space-times of window i (+ B-mode thumbnail)
slopes.json            GUI         per window: the hand slope, keyed by the space-time hash
log.jsonl              GUI         append-only record of every accept / skip (audit trail)
=====================  ==========  ============================================================

Staleness propagates through the hashes: a redrawn general line changes ``general.hash`` ->
windows are re-detected -> ``windows.hash`` changes -> old event lines no longer count -> their
space-times and slopes no longer count. Nothing is deleted; replaced files are archived.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CONFIG = str(REPO / "configs" / "passive_manual.yaml")
OUTDIR = "swp_passive_manual"
N_SAMPLES = 250

RUNTIME_MAT = "AcquisitionParametersAndECG.mat"
COMBINED_MAT = "CombinedData.mat"


# ------------------------------------------------------------------ paths
class Paths:
    def __init__(self, folder):
        self.folder = str(folder)
        self.output = os.path.join(self.folder, "output")
        self.dir = os.path.join(self.output, OUTDIR)
        j = lambda n: os.path.join(self.dir, n)             # noqa: E731
        self.general_json, self.general_npz = j("general.json"), j("general_mline.npz")
        self.windows_json, self.events_json = j("windows.json"), j("events.json")
        self.processed_json, self.slopes_json = j("processed.json"), j("slopes.json")
        self.log = j("log.jsonl")
        self.lock = j("worker.lock")

    def event_npz(self, i):
        return os.path.join(self.dir, f"event{i}_mline.npz")

    def st_npz(self, i):
        return os.path.join(self.dir, f"st_win{i}.npz")

    def bmode(self, buffer):
        return os.path.join(self.output, f"CombinedData_buffer{buffer}_iq.hdf5")

    # the earlier (September) passive run, read only
    def legacy_general(self):
        return os.path.join(self.output, "mlines", "passive_general_mline.npz")

    def legacy_windows(self):
        return os.path.join(self.output, "swp_passive", "passive_windows.json")

    def legacy_event(self, i):
        return os.path.join(self.output, "mlines", f"passive_win{i}_mline.npz")


# ------------------------------------------------------------------ io
def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    for attempt in range(5):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            if attempt == 4:
                raise
            time.sleep(0.3)                  # a writer's os.replace in flight on the share


def write_json(path, obj):
    """Atomic write. Retries os.replace: on the network share it can briefly hit a reader."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def append_log(p: Paths, record: dict):
    os.makedirs(p.dir, exist_ok=True)
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(), **record)
    with open(p.log, "a") as f:
        f.write(json.dumps(rec, default=_jsonable) + "\n")


def save_line(path, points_m):
    """Atomic, like write_json: a session killed mid-write leaves the previous line intact."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, points=np.asarray(points_m, float), n_samples=N_SAMPLES)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def load_points(path):
    return np.asarray(np.load(path)["points"], float)


def points_hash(points) -> str:
    return hashlib.sha1(np.round(np.asarray(points, float), 7).tobytes()).hexdigest()[:12]


def windows_hash(windows) -> str:
    return hashlib.sha1(json.dumps([round(w["t_peak"], 5) for w in windows]).encode()).hexdigest()[:12]


def archive(p: Paths, names, tag):
    """Move the named files of the manual dir into ``archive_<time>_<tag>/`` (never delete)."""
    present = [n for n in names if os.path.exists(os.path.join(p.dir, n))]
    if not present:
        return None
    dest = os.path.join(p.dir, f"archive_{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    os.makedirs(dest, exist_ok=True)
    for n in present:
        shutil.move(os.path.join(p.dir, n), os.path.join(dest, n))
    return dest


def downstream_files(p: Paths):
    """Everything that depends on the general line (all but general.* and the log)."""
    if not os.path.isdir(p.dir):
        return []
    keep = {"general.json", "general_mline.npz", "general.png", "log.jsonl", "worker.lock"}
    return [n for n in os.listdir(p.dir)
            if n not in keep and not n.startswith("archive_") and os.path.isfile(os.path.join(p.dir, n))]


# ------------------------------------------------------------------ folders
def find_folders(root=None, folders=(), subject=None):
    """Measurement folders, the ones with September passive lines first (then sorted)."""
    found = [Path(f) for f in folders]
    if root:
        pattern = f"{subject}/*" if subject else "*/*"
        # Strain_data acquisitions carry the same runtime .mat but no passive buffer; listing
        # them only makes every queue refresh poll ~500 never-ready folders on the share
        found += [q for q in Path(root).glob(pattern)
                  if q.is_dir() and "strain_data" not in q.name.lower()
                  and ((q / RUNTIME_MAT).is_file() or (q / COMBINED_MAT).is_file())]
    seen, out = set(), []
    for q in sorted(found, key=lambda q: str(q).lower()):
        k = str(q.resolve()).lower()
        if k not in seen:
            seen.add(k)
            out.append(str(q))
    legacy = [f for f in out if os.path.exists(Paths(f).legacy_general())]
    return legacy + [f for f in out if f not in legacy]


def ready(p: Paths) -> bool:
    """Beamforming finished for this folder: the batch writes the buffer-4 GIF after buffers
    1, 3 and 4 are complete."""
    return (os.path.exists(p.bmode(4))
            and os.path.exists(os.path.join(p.output, "CombinedData_buffer4_iq.gif")))


# ------------------------------------------------------------------ state
def state(folder) -> dict:
    """Where a folder stands. Keys: stage, and the window indices still needing each step.

    stage: not-ready | need-general | skipped | detecting | error | no-windows | need-events |
           processing | need-slopes | done
    (a window whose processing failed is listed in ``proc_failed`` and does not hold the folder;
    a window below the detection screen without a line is listed in ``screened`` and does not
    hold it either)
    """
    p = Paths(folder)
    s = dict(folder=str(folder), stage="not-ready", n_windows=0, need_events=[], need_proc=[],
             need_slopes=[], slopes_done=[], slopes_skipped=[], lines_skipped=[], proc_failed=[],
             screened=[])
    if not ready(p):
        return s
    gen = read_json(p.general_json)
    if gen is None:
        s["stage"] = "need-general"
        return s
    if gen.get("skipped"):
        s["stage"] = "skipped"
        return s
    errs = read_json(os.path.join(p.dir, "worker_errors.json"), {})
    win = read_json(p.windows_json)
    if win is None or win.get("key", {}).get("general_hash") != gen["hash"]:
        failed = errs.get("detect", {}).get("key") == gen["hash"]
        s["stage"] = "error" if failed else "detecting"
        s["error"] = errs.get("detect", {}).get("error") if failed else None
        return s
    windows = win["windows"]
    s["n_windows"] = len(windows)
    if not windows:
        s["stage"] = "no-windows"
        return s
    s["windows"] = windows
    ev = read_json(p.events_json) or {}
    events = ev.get("events", {}) if ev.get("windows_hash") == win["hash"] else {}
    proc = read_json(p.processed_json) or {}
    slopes = read_json(p.slopes_json) or {}
    for i in range(len(windows)):
        k = str(i)
        e = events.get(k)
        if e is None:
            # below the general-line screen (swp.passive_screen): no line asked unless requested
            (s["screened"] if windows[i].get("screened") else s["need_events"]).append(i)
            continue
        if e.get("skipped"):
            s["lines_skipped"].append(i)
            continue
        pr = proc.get(k)
        if pr is None or pr.get("line_hash") != e["hash"] or not os.path.exists(p.st_npz(i)):
            if errs.get(f"process{i}", {}).get("key") == e["hash"]:
                s["proc_failed"].append(i)
            else:
                s["need_proc"].append(i)
            continue
        sl = slopes.get(k)
        if sl is None or sl.get("st_hash") != pr["st_hash"]:
            s["need_slopes"].append(i)
        elif sl.get("skipped"):
            s["slopes_skipped"].append(i)
        else:
            s["slopes_done"].append(i)
    s["stage"] = ("need-events" if s["need_events"] else "processing" if s["need_proc"]
                  else "need-slopes" if s["need_slopes"] else "done")
    return s


# ------------------------------------------------------------------ worker lock
def _lock_owner_dead(path) -> bool:
    """True when the lock was written on THIS host by a process that no longer runs (a session
    killed mid-processing). Another host's claim is only released by age."""
    try:
        with open(path) as fh:
            host, pid = fh.read().split()[:2]
        pid = int(pid)
        mtime = os.path.getmtime(path)
    except (OSError, ValueError):
        return False
    if host != socket.gethostname():
        return False
    try:
        import psutil
    except ImportError:                                   # pragma: no cover - age rule only
        return False
    try:                                                  # a reused PID started after the claim
        return psutil.Process(pid).create_time() > mtime + 1
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:                                  # pragma: no cover - access denied etc.
        return False


@contextmanager
def folder_lock(p: Paths, stale_s=3 * 3600):
    """Exclusive claim of a folder by one worker (yields False if another worker has it).

    A claim is released when it is older than ``stale_s`` or, on the same host, as soon as its
    process is gone - so a session killed mid-processing does not hold the folder for hours."""
    os.makedirs(p.dir, exist_ok=True)
    got = False
    for _ in range(2):
        try:
            fd = os.open(p.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.ctime()}".encode())
            os.close(fd)
            got = True
            break
        except FileExistsError:
            try:
                if (time.time() - os.path.getmtime(p.lock) > stale_s
                        or _lock_owner_dead(p.lock)):
                    os.remove(p.lock)                     # a crashed worker's claim
                    continue
            except FileNotFoundError:
                continue
            break
    try:
        yield got
    finally:
        if got:
            try:
                os.remove(p.lock)
            except FileNotFoundError:
                pass
