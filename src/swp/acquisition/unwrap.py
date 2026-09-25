"""Make buffer 3 (focused live loop, a circular receive buffer) chronological.

Why this exists
---------------
Buffer 3 is recorded as a live loop: frame i of every pass is written into receive-buffer slot i,
and the loop is left (UI button -> ``set&Run``) wherever the hardware happens to be. The stored
slots are therefore ROTATED: the oldest frame is not slot 0. VSX would report the head in
``Resource.RcvBuffer(3).lastFrame``, and zea's converter uses it to unwrap
(``VerasonicsFile.get_raw_data_order``) - but VSX only sets that field "when a script freezes or
exits" (Vantage Sequence Programming Manual p57, Tutorial p64), while ``SaveRFData.m`` copies the
buffers mid-sequence. The saved value is always ``numFrames``, so no unwrapping happened.
Buffers 1, 2, 4, 5, 6 are written in one complete pass and are chronological.

The frame triggers and R-peaks in the trigger log are correct, so the head can be recovered. In
order of preference (calibration: ``study/analysis/buffer3_unwrap_calibration.py``,
``docs/buffer3_unwrap.md``):

``trigger-count``  EXACT, when the trigger log still holds the start of the live run (~43 % of the
                   study; the log is circular and loses the start of long runs). With L triggers in
                   the run the oldest stored slot is ``(L - 1) mod N``: the loop is always left
                   mid-frame, after that frame's trigger but before its transfer, so the last
                   trigger has no stored frame (``shift`` -1 for the frame times).
``buffer1``        every stored frame gets a trigger time and cardiac phase under each candidate
                   head and is compared with the buffer-1 frame (chronological, correctly timed)
                   at the same phase; the head with the most similar pairs wins. Needs a
                   trustworthy R-peak record. Against the trigger count (145 folders): 89 % exact,
                   and at margin >= MIN_MARGIN_BUFFER1 (75 % of folders) 98.9 % exact, the one
                   miss being one slot.
otherwise          'ambiguous': stored order kept, ``buffer_unwrapped`` False.

Frame continuity (the least similar cyclic neighbour pair is the wrap) was tested on the IQ and on
beamforming-free RF and rejected: 33-56 % exact, with large errors at high margins.

The result is written back into BOTH the converted RF file and the beamformed IQ file (atomically:
a reordered copy replaces the original), with flags so it is never applied twice:

    custom/buffer_unwrapped      True once the frames are chronological
    custom/unwrap_status         'unwrapped' | 'chronological' (head already at slot 0) |
                                 'ambiguous' (left in stored order) | 'no-data'
    custom/unwrap_first_frame    stored (.mat) slot that was acquired first; -1 if unknown
    custom/unwrap_shift_frames   trigger-to-frame assignment offset used for the frame times
    custom/unwrap_method, unwrap_details (JSON), unwrap_time, unwrap_version
    custom/frame_time_ms         chronological frame CENTRE times on the trigger-log clock
                                 (trigger fires on the first of 73 lines; centre = + half a frame)
    custom/frame_rpeak_phase_ms  time since the preceding logged R-peak, per frame (NaN if unknown)

and the IQ ``timestamps`` become the chronological frame-centre times relative to frame 0.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin                                            # noqa: F401 - zea's lzf/blosc filters
except ImportError:                                              # pragma: no cover
    pass

VERSION = 1
BUFFER = 3                     # MATLAB buffer number of the focused live loop
_RAW = "tracks/track_0/data/raw_data"
_BF = "tracks/track_0/data/beamformed_data"
GX = np.arange(-25, 25.01, 0.8)          # common anatomy grid (mm): where the heart is in PLAX
GZ = np.arange(20, 100.01, 0.8)

# Decision threshold (study/analysis/buffer3_unwrap_calibration.py, docs/buffer3_unwrap.md): of the
# buffer-1 heads that disagreed with the exact trigger count, all but one (off by one slot) had a
# margin <= 0.005.
MIN_MARGIN_BUFFER1 = 0.006     # buffer-1 score gap to the best non-adjacent head
SHIFT = -1                     # the loop is left mid-frame: the last trigger has no stored frame
DEFAULT_SHIFT = SHIFT


# ------------------------------------------------------------------ files
def buffer_files(output_dir, buffer=BUFFER):
    """(converted RF path or None, IQ path or None) of one buffer in an ``output`` folder."""
    out = Path(output_dir)
    conv = sorted((out / "converted").glob(f"*_buffer{buffer}.hdf5"))
    iq = sorted(out.glob(f"*_buffer{buffer}_iq.hdf5"))
    return (conv[0] if conv else None), (iq[0] if iq else None)


def read_flag(path):
    """The unwrap record stored in ``path`` (dict) or None when the file was never unwrapped."""
    if path is None or not Path(path).exists():
        return None
    with h5py.File(path, "r") as f:
        if "custom/unwrap_status" not in f:
            return None
        g = f["custom"]
        rec = {k[len("unwrap_"):] if k.startswith("unwrap_") else k: _read(g[k])
               for k in g if k.startswith("unwrap_") or k == "buffer_unwrapped"}
    return rec


def _read(ds):
    v = ds[()]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.generic):
        v = v.item()
    return v


# ------------------------------------------------------------------ anatomy images
def _anatomy(env, x_mm, z_mm):
    from ..mline.transfer import anatomy, resample
    a = anatomy(resample(env, x_mm, z_mm, GX, GZ), 0.8)
    a = a - a.mean()
    return (a / (np.linalg.norm(a) + 1e-12)).ravel()


def anatomy_stack(iq_path):
    """Per-frame anatomy vectors of a beamformed IQ file (stored frame order)."""
    with h5py.File(iq_path, "r") as f:
        v = np.asarray(f[f"{_BF}/values"], np.float32)
        c = np.asarray(f[f"{_BF}/coordinates"])
    x, z = c[0, :, 0] * 1e3, c[:, 0, 2] * 1e3
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    if x[0] > x[-1]:
        x, env = x[::-1], env[:, :, ::-1]
    if z[0] > z[-1]:
        z, env = z[::-1], env[:, ::-1]
    return np.stack([_anatomy(e, x, z) for e in env])


# ------------------------------------------------------------------ estimators
def continuity_scores(A):
    """Score per candidate head (lower = more likely): the lag-1 and lag-2 correlations crossing
    the break before that slot, each relative to the median of its lag."""
    n = len(A)
    M = A @ A.T
    lag = {L: np.array([M[i, (i + L) % n] for i in range(n)]) for L in (1, 2)}
    med = {L: float(np.median(v)) for L, v in lag.items()}
    score = np.empty(n)
    for first in range(n):
        b = (first - 1) % n
        score[first] = 0.5 * (lag[1][b] / med[1] + np.mean([lag[2][(b - 1) % n], lag[2][b]]) / med[2])
    return score


def _phase(t_ms, r_peaks):
    """Time since the preceding R-peak for each time in ``t_ms`` (1-D array; NaN before the first)."""
    t_ms = np.atleast_1d(np.asarray(t_ms, float))
    out = np.full(t_ms.shape, np.nan)
    for i, t in enumerate(t_ms):
        prev = r_peaks[r_peaks <= t]
        if prev.size:
            out[i] = t - prev.max()
    return out


def chronological_times(folder, n, shift):
    """Frame-centre times (ms, trigger-log clock) and R-peak phases of chronological frames 0..n-1,
    or (None, None) when the trigger log cannot place buffer 3."""
    from .triggerlog import buffer_timing
    b3 = buffer_timing(str(folder), BUFFER)
    if b3 is None or b3.n_frames != n:
        return None, None
    t = b3.frame_times() + b3.frame_ms / 2 + shift * b3.frame_ms
    return t, _phase(t, np.asarray(b3.r_peaks_ms))


def buffer1_scores(folder, A3, iq1_path, max_dphase_ms=10.0):
    """{shift: score per candidate head} (higher = better), or None without usable timing."""
    from .triggerlog import buffer_timing
    b1 = buffer_timing(str(folder), 1)
    b3 = buffer_timing(str(folder), BUFFER)
    n = len(A3)
    if b1 is None or b3 is None or b3.n_frames != n or not Path(iq1_path).exists():
        return None
    A1 = anatomy_stack(iq1_path)
    if len(A1) != b1.n_frames:
        return None
    rp = np.asarray(b3.r_peaks_ms)
    ph1 = _phase(b1.frame_times() + b1.frame_ms / 2, rp)
    ok1 = np.isfinite(ph1)
    M = A3 @ A1.T
    out = {}
    for shift in (-1, 0, 1):
        _, ph_c = chronological_times(folder, n, shift)
        sc = np.full(n, np.nan)
        for first in range(n):
            vals = []
            for q in range(n):
                ph = ph_c[(q - first) % n]
                if not np.isfinite(ph):
                    continue
                d = np.where(ok1, np.abs(ph1 - ph), np.inf)
                j = int(np.argmin(d))
                if d[j] < max_dphase_ms:
                    vals.append(M[q, j])
            if len(vals) >= n // 2:
                sc[first] = float(np.mean(vals))
        out[shift] = sc
    return out


def trigger_count_head(folder):
    """Exact head from the trigger log, or None when the start of the live run is not in the log.

    The loop writes slots 1..N cyclically, one trigger per frame (on its first transmit). With L
    triggers in the run and the last one belonging to the frame being acquired when the loop was
    left (aborted mid-frame, never transferred), the oldest stored slot is c = (L - 1) mod N.
    The log is circular (1500/2000 triggers, shared with every buffer), so a long live run loses
    its start: then the count is unknown and None is returned.
    """
    from .triggerlog import _runs, read_log
    log = read_log(str(folder))
    if log is None or BUFFER not in log[2]:
        return None
    _, trig, params = log
    n, fms = params[BUFFER]["n"], 1000.0 / params[BUFFER]["fps"]
    runs = [(i, j) for i, j in _runs(trig, fms, max(1.5, 0.08 * fms)) if j - i + 1 >= n]
    if not runs:
        return None
    i, j = runs[-1]
    if i == 0:                                       # run starts at the start of the log: truncated
        return None
    L = j - i + 1
    return dict(run_length=int(L), n=int(n), c=int((L - 1) % n),
                gap_before_ms=float(trig[i] - trig[i - 1]))


def resolve_trigger_count(tc, A3):
    """(first, 'aborted'|'clean', local margin): is slot c the OLDEST frame (loop left mid-frame,
    the usual case) or the NEWEST (left between frames, head c + 1)? The break is on the side of
    slot c with the lower neighbour correlation."""
    n, c = len(A3), tc["c"]
    before = float(A3[(c - 1) % n] @ A3[c])
    after = float(A3[c] @ A3[(c + 1) % n])
    if before <= after:
        return c, "aborted", after - before
    return (c + 1) % n, "clean", before - after


def _ecg_ok(folder):
    from .rrcheck import assess_rr
    return bool(assess_rr(str(folder)).trustworthy)


def best_and_margin(values, lower_is_better):
    """(best index, gap to the best candidate that is not adjacent to it, cyclically)."""
    v = np.asarray(values, float)
    n = len(v)
    s = np.where(np.isfinite(v), v if lower_is_better else -v, np.inf)
    i = int(np.argmin(s))
    j = min((k for k in range(n) if min((k - i) % n, (i - k) % n) >= 2), key=lambda k: s[k])
    return i, float(s[j] - s[i])


@dataclass
class Estimate:
    status: str                      # unwrapped | chronological | ambiguous | no-data
    first: int = -1                  # stored slot acquired first
    shift: int = DEFAULT_SHIFT
    method: str = ""
    details: dict = field(default_factory=dict)

    @property
    def resolved(self):
        return self.status in ("unwrapped", "chronological")


def estimate(folder, output_dir=None, cross_check=True):
    """Find the head of buffer 3 for one measurement folder (read-only).

    1. trigger count (exact) when the log holds the start of the live run;
    2. otherwise buffer-1 similarity (shift fixed at -1), accepted at margin >= MIN_MARGIN_BUFFER1;
    3. otherwise 'ambiguous'.
    With ``cross_check`` the buffer-1 method also runs when the trigger count is available, and its
    agreement is recorded in the details (a running check of both on every folder).
    """
    output_dir = Path(output_dir) if output_dir else Path(folder) / "output"
    _, iq3 = buffer_files(output_dir)
    if iq3 is None:
        return Estimate("no-data", details=dict(reason="no buffer-3 IQ"))
    A3 = anatomy_stack(iq3)
    n = len(A3)
    det = dict(n_frames=n)
    tc = trigger_count_head(folder)
    if tc is not None:
        det.update(trigger_run_length=tc["run_length"], trigger_first=tc["c"])
    try:
        ecg_ok = _ecg_ok(folder)
    except Exception:                                                  # noqa: BLE001
        ecg_ok = False
    det["ecg_trustworthy"] = ecg_ok
    b_first = None
    if ecg_ok and (tc is None or cross_check):
        _, iq1 = buffer_files(output_dir, buffer=1)
        sc = buffer1_scores(folder, A3, iq1) if iq1 is not None else None
        if sc is not None and np.isfinite(sc[SHIFT]).any():
            b_first, b_margin = best_and_margin(sc[SHIFT], lower_is_better=False)
            det.update(buffer1_first=b_first, buffer1_margin=round(b_margin, 4),
                       buffer1_score=round(float(sc[SHIFT][b_first]), 4),
                       buffer1_score_stored_order=round(float(sc[SHIFT][0]), 4))
    if tc is not None:
        first, method = tc["c"], "trigger-count"
        if b_first is not None:
            det["buffer1_agrees"] = bool(b_first == first)
    elif b_first is not None and det["buffer1_margin"] >= MIN_MARGIN_BUFFER1:
        first, method = b_first, "buffer1"
    else:
        det["reason"] = ("no trigger count; " + ("buffer-1 margin below threshold" if b_first is not None
                                                 else "no trustworthy ECG / buffer 1"))
        return Estimate("ambiguous", method="none", details=det)
    return Estimate("chronological" if first == 0 else "unwrapped", int(first), SHIFT, method, det)


def _cyc(a, b, n):
    return min((a - b) % n, (b - a) % n)


# ------------------------------------------------------------------ rewriting files
def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _rewrite(path, order, per_frame, extra):
    """Write a copy of ``path`` with the datasets in ``per_frame`` reordered along axis 0 by
    ``order`` (new frame c = old frame order[c]) and the ``extra`` custom/ datasets set, then
    atomically replace the original. ``extra`` values of None delete that dataset."""
    path = Path(path)
    tmp = path.with_name(path.name + ".unwrap_tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        _copy_attrs(src, dst)

        def copy_group(sg, dg):
            for name, obj in sg.items():
                full = obj.name.lstrip("/")
                if isinstance(obj, h5py.Group):
                    copy_group(obj, dg.require_group(name))
                    _copy_attrs(obj, dg[name])
                elif full in per_frame:
                    d = dg.create_dataset(name, shape=obj.shape, dtype=obj.dtype, chunks=obj.chunks,
                                          compression=obj.compression, compression_opts=obj.compression_opts,
                                          shuffle=obj.shuffle, fletcher32=obj.fletcher32)
                    _copy_attrs(obj, d)
                    new = per_frame[full]
                    if new is not None:                    # replacement values (e.g. timestamps)
                        d[...] = new
                    else:
                        for c, k in enumerate(order):
                            d[c] = obj[k]
                elif not (full.startswith("custom/") and full in extra):
                    sg.copy(obj, dg, name=name)

        copy_group(src, dst)
        g = dst.require_group("custom")
        for k, v in extra.items():
            name = k.split("/", 1)[1]
            if name in g:
                del g[name]
            if v is not None:
                g.create_dataset(name, data=v)
    os.replace(tmp, path)


def _write_inplace(path, extra, replace=None):
    """Flags (and small replacement datasets) written in place - when no frame moves."""
    with h5py.File(path, "a") as f:
        g = f.require_group("custom")
        for k, v in extra.items():
            name = k.split("/", 1)[1]
            if name in g:
                del g[name]
            if v is not None:
                g.create_dataset(name, data=v)
        for k, v in (replace or {}).items():
            if v is not None and k in f:
                f[k][...] = v


def _extra(est, times, phases, when):
    return {
        "custom/buffer_unwrapped": bool(est.resolved),
        "custom/unwrap_status": est.status,
        "custom/unwrap_first_frame": int(est.first),
        "custom/unwrap_shift_frames": int(est.shift),
        "custom/unwrap_method": est.method,
        "custom/unwrap_details": json.dumps(est.details, default=float),
        "custom/unwrap_time": when,
        "custom/unwrap_version": VERSION,
        "custom/frame_time_ms": None if times is None else np.asarray(times, float),
        "custom/frame_rpeak_phase_ms": None if phases is None else np.asarray(phases, float),
    }


def unwrap_buffer3(folder, output_dir=None, dry_run=False, gif=True, log=print):
    """Unwrap buffer 3 of one folder: the converted RF and IQ files, then every other buffer-3 IQ
    variant (REFoCUS, incoherent, ... reconstructions: ``*_buffer3_<variant>_iq.hdf5``) with the
    same head. Returns the Estimate (or the stored record) of the main files."""
    res = _unwrap_main(folder, output_dir, dry_run=dry_run, gif=gif, log=log)
    if not dry_run:
        unwrap_variants(folder, output_dir, gif=gif, log=log)
    return res


def _record_estimate(rec):
    return Estimate(rec["status"], int(rec["first_frame"]), int(rec["shift_frames"]), rec["method"],
                    json.loads(rec["details"]))


def unwrap_variants(folder, output_dir=None, gif=True, log=print):
    """Apply the head stored in the main buffer-3 file to the other buffer-3 IQ files of the folder.

    Reconstruction variants are beamformed from the same (stored-order) RF, so they carry the same
    rotation; they are reordered with the stored head, never re-estimated. Idempotent. Returns the
    list of files changed."""
    output_dir = Path(output_dir) if output_dir else Path(folder) / "output"
    conv, iq = buffer_files(output_dir)
    rec = read_flag(iq) or read_flag(conv)
    if rec is None:
        return []
    est = _record_estimate(rec)
    changed = []
    for path in sorted(output_dir.glob(f"*_buffer{BUFFER}_*_iq.hdf5")):
        if iq is not None and path.samefile(iq) or read_flag(path):
            continue
        with h5py.File(path, "r") as f:
            if f"{_BF}/values" not in f:
                continue
            n = f[f"{_BF}/values"].shape[0]
        if est.resolved and n != int(est.details.get("n_frames", n)):
            log(f"  {path.name}: {n} frames, main file {est.details.get('n_frames')} - left alone")
            continue
        times, phases = chronological_times(folder, n, est.shift) if est.resolved else (None, None)
        extra = _extra(est, times, phases, time.strftime("%Y-%m-%d %H:%M:%S"))
        ts = ((times - times[0]) * 1e-3).astype(np.float32) if times is not None else None
        moves = est.resolved and est.first != 0
        if moves:
            per = {f"{_BF}/values": None}
            if ts is not None:
                per[f"{_BF}/timestamps"] = ts
            _rewrite(path, (np.arange(n) + est.first) % n, per, extra)
        else:
            _write_inplace(path, extra, replace={f"{_BF}/timestamps": ts})
        from ..provenance import stamp_h5
        stamp_h5(str(path), extra={"stage": "unwrap", "buffer": BUFFER}, group="provenance_unwrap")
        changed.append(path)
        log(f"  {path.name}: {'reordered' if moves else 'flagged'} ({est.status}, head {est.first})")
        if gif and moves:
            try:
                from .gifs import gif_for_file
                gif_for_file(Path(path))
            except Exception as exc:                                   # noqa: BLE001
                log(f"  {path.name}: GIF not re-rendered ({exc})")
    return changed


def _unwrap_main(folder, output_dir=None, dry_run=False, gif=True, log=print):
    """Estimate the head of buffer 3 and rewrite its converted RF and IQ files chronologically.

    Safe to re-run: a file already carrying ``custom/unwrap_status`` is never permuted again. If one
    file is flagged and the other is not (e.g. buffer 3 re-beamformed from an unwrapped converted
    file, or reconverted from the .mat), the stored head is reused - frames already chronological
    are not moved, frames in .mat order are. Returns the Estimate (or the stored record).
    """
    output_dir = Path(output_dir) if output_dir else Path(folder) / "output"
    conv, iq = buffer_files(output_dir)
    fc, fi = read_flag(conv), read_flag(iq)
    if (conv is None or fc) and (iq is None or fi):
        log(f"  buffer 3: already {(fc or fi or {}).get('status', 'absent')} - nothing to do")
        return fc or fi
    ref = fc or fi
    if ref is not None:
        est = _record_estimate(ref)
        # a file carrying no flag is in .mat order unless it was produced from the flagged file
        # after the unwrap: the IQ re-beamformed from an unwrapped converted file
        iq_from_unwrapped = (fc and not fi and iq is not None
                             and os.path.getmtime(iq) > time.mktime(time.strptime(fc["time"], "%Y-%m-%d %H:%M:%S")))
        log(f"  buffer 3: reusing stored head (slot {est.first}, {est.status})")
    else:
        est = estimate(folder, output_dir)
        iq_from_unwrapped = False
        log(f"  buffer 3: {est.status} - first slot {est.first}, shift {est.shift:+d} ({est.method}; "
            + ", ".join(f"{k} {v}" for k, v in est.details.items() if "first" in k or "margin" in k) + ")")
    if dry_run or est.status == "no-data":
        return est
    n = None
    for p in (conv, iq):
        if p is not None:
            with h5py.File(p, "r") as f:
                n = f[_RAW].shape[0] if _RAW in f else f[f"{_BF}/values"].shape[0]
            break
    order = (np.arange(n) + (est.first if est.resolved else 0)) % n
    times, phases = chronological_times(folder, n, est.shift) if est.resolved else (None, None)
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    extra = _extra(est, times, phases, when)
    from ..provenance import stamp_h5
    moves = est.resolved and est.first != 0
    if conv is not None and not fc:
        if moves:
            _rewrite(conv, order, {_RAW: None}, extra)
        else:
            _write_inplace(conv, extra)
        stamp_h5(str(conv), extra={"stage": "unwrap", "buffer": BUFFER}, group="provenance_unwrap")
    if iq is not None and not fi:
        ts = None
        if times is not None:
            ts = ((times - times[0]) * 1e-3).astype(np.float32)
        iq_moves = moves and not iq_from_unwrapped
        if iq_moves:
            per = {f"{_BF}/values": None}
            if ts is not None:
                per[f"{_BF}/timestamps"] = ts
            _rewrite(iq, order, per, extra)
        else:
            _write_inplace(iq, extra, replace={f"{_BF}/timestamps": ts})
        stamp_h5(str(iq), extra={"stage": "unwrap", "buffer": BUFFER}, group="provenance_unwrap")
        if gif and iq_moves:
            try:
                from .gifs import gif_for_file
                gif_for_file(Path(iq))
            except Exception as exc:                                   # noqa: BLE001
                log(f"  buffer 3: GIF not re-rendered ({exc})")
    return est


def summary(est):
    return dict(status=est.status, first=est.first, shift=est.shift, method=est.method,
                **{k: v for k, v in est.details.items()}) if isinstance(est, Estimate) else dict(est or {})
