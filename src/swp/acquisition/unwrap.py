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
``combined``       (needs a trustworthy ECG) z(buffer-1 similarity: every stored frame vs the
                   buffer-1 frame at the same cardiac phase under each candidate head)
                   + z(expected motion: buffer 3's own frame-similarity matrix vs buffer 1's at
                   the same phases) + 0.5 z(continuity); 98 % exact on the 210-folder trigger-count
                   reference, 99 % at margin >= MIN_MARGIN_COMBINED (96 % of folders).
``continuity``     the least similar cyclic neighbour pair is the wrap; image-only, so the one
                   method without ECG, accepted only at a high margin. It works when the wrap
                   joins different cardiac phases (92-95 % exact for a > 300 ms phase jump).
otherwise          'ambiguous': stored order kept, ``buffer_unwrapped`` False.

VERSION 2 (2026-09-28): the loop's triggers come from the parsed acquisition sequence
(triggerlog.parse_sequence) - VERSION 1 counted buffer 1's first trigger as a loop frame in 227 of
724 folders, one slot off - and the combined estimator replaces buffer-1-only. Files flagged by
VERSION 1 are re-estimated and rotated from their current head (docs/buffer3_unwrap.md).

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

VERSION = 2                     # 2 (2026-09-28): parsed trigger log + combined estimator; v1 files are redone
BUFFER = 3                     # MATLAB buffer number of the focused live loop
_RAW = "tracks/track_0/data/raw_data"
_BF = "tracks/track_0/data/beamformed_data"
GX = np.arange(-25, 25.01, 0.8)          # common anatomy grid (mm): where the heart is in PLAX
GZ = np.arange(20, 100.01, 0.8)

# Decision threshold (study/analysis/buffer3_unwrap_calibration.py, docs/buffer3_unwrap.md): of the
# buffer-1 heads that disagreed with the exact trigger count, all but one (off by one slot) had a
# margin <= 0.005.
MIN_MARGIN_BUFFER1 = 0.006     # (VERSION 1) buffer-1 score gap to the best non-adjacent head
# VERSION 2 (study/analysis/unwrap_combined_calibration.py, corrected trigger-count reference):
MIN_MARGIN_COMBINED = 1.3      # combined z-score gap: 99 % exact on 210 reference folders
MIN_MARGIN_CONTINUITY = 0.055  # continuity alone (no ECG): its 99 % point
W_CONTINUITY = 0.5             # weight of continuity in the combined score (0.5 beat 1.0 and a wrap-jump weight)
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

    The run is taken from the parsed acquisition sequence (triggerlog.parse_sequence), which ends
    it before buffer 1's first trigger; a log that does not fit the sequence gives None.
    """
    from .triggerlog import parse_sequence, read_log
    log = read_log(str(folder))
    if log is None or BUFFER not in log[2]:
        return None
    seq = parse_sequence(log)
    if seq is None or seq.loop_truncated:
        return None
    _, trig, params = log
    n = params[BUFFER]["n"]
    i, j = seq.loop
    if j - i + 1 < n:
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


def _zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def buffer1_curve(A1, A3, ph1, ph3, rr_ms, tol_ms=10.0):
    """Buffer-1 similarity per candidate head (higher = better). Each stored buffer-3 frame is paired
    with the buffer-1 frame at the same cardiac phase (circular, mod RR, +-tol_ms); correlations are
    z-scored per buffer-3 frame across buffer-1 frames first, which removes a frame's own baseline."""
    M = A3 @ A1.T
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-12)
    n = len(A3)
    ok1 = np.isfinite(ph1)
    sc = np.full(n, np.nan)
    for first in range(n):
        vals = []
        for q in range(n):
            ph = ph3[(q - first) % n]
            if not np.isfinite(ph):
                continue
            dd = np.abs(ph1 - ph)
            if np.isfinite(rr_ms):
                dd = np.minimum(dd, rr_ms - dd)
            dd = np.where(ok1, dd, np.inf)
            j = int(np.argmin(dd))
            if dd[j] < tol_ms:
                vals.append(M[q, j])
        if len(vals) >= n // 2:
            sc[first] = np.mean(vals)
    return sc


def expected_motion_curve(A1, A3, ph1, ph3, frame_ms):
    """Per candidate head (higher = better): correlation between buffer 3's own frame-to-frame
    similarity matrix and the one buffer 1 predicts for the same phases. Each buffer is compared only
    with itself, so the widebeam-vs-focused image difference cancels. None without enough phases."""
    n = len(A3)
    ok1 = np.flatnonzero(np.isfinite(ph1))
    jk = np.full(n, -1)
    for k in range(n):
        if np.isfinite(ph3[k]) and ok1.size:
            dd = np.abs(ph1[ok1] - ph3[k])
            if dd.min() < frame_ms / 2:
                jk[k] = ok1[np.argmin(dd)]
    valid = np.flatnonzero(jk >= 0)
    if valid.size < n // 2:
        return None
    S1 = A1[jk[valid]] @ A1[jk[valid]].T
    S3 = A3 @ A3.T
    iu = np.triu_indices(valid.size, 1)
    p = S1[iu]
    sc = np.full(n, np.nan)
    for first in range(n):
        q = (valid + first) % n
        sc[first] = np.corrcoef(S3[np.ix_(q, q)][iu], p)[0, 1]
    return sc


def estimate(folder, output_dir=None, current_head=0):
    """Find the head of buffer 3 for one measurement folder (read-only).

    ``current_head``: the stored slot at frame 0 of the buffer-3 IQ file as it is now (0 = stored
    order; an earlier unwrap's head otherwise) - the features are rotated back to stored order.

    Decision (VERSION 2, calibration: study/analysis/unwrap_combined_calibration.py):
    1. trigger count (exact) when the parsed log holds the start of the live run;
    2. otherwise, with a trustworthy ECG, the COMBINED score
       z(buffer-1 similarity) + z(expected motion) + 0.5 z(continuity), accepted at margin >=
       MIN_MARGIN_COMBINED (99 % exact on the 210-folder trigger-count reference);
    3. otherwise continuity alone at margin >= MIN_MARGIN_CONTINUITY (no ECG needed; 99 % point);
    4. otherwise 'ambiguous'.
    The combined score also runs where the trigger count exists and its agreement is recorded.
    """
    from .triggerlog import buffer_timing, median_rr_ms
    output_dir = Path(output_dir) if output_dir else Path(folder) / "output"
    _, iq3 = buffer_files(output_dir)
    if iq3 is None:
        return Estimate("no-data", details=dict(reason="no buffer-3 IQ"))
    A3 = anatomy_stack(iq3)
    n = len(A3)
    A3 = A3[(np.arange(n) - current_head) % n]                 # back to stored slot order
    det = dict(n_frames=n)
    tc = trigger_count_head(folder)
    if tc is not None:
        det.update(trigger_run_length=tc["run_length"], trigger_first=tc["c"])
    try:
        ecg_ok = _ecg_ok(folder)
    except Exception:                                                  # noqa: BLE001
        ecg_ok = False
    det["ecg_trustworthy"] = ecg_ok
    cont = -continuity_scores(A3)                                      # higher = more likely
    c_first, c_margin = best_and_margin(cont, lower_is_better=False)
    det.update(continuity_first=c_first, continuity_margin=round(c_margin, 4))
    comb = None
    b1, b3 = buffer_timing(str(folder), 1), buffer_timing(str(folder), BUFFER)
    _, iq1 = buffer_files(output_dir, buffer=1)
    if ecg_ok and b1 is not None and b3 is not None and b3.n_frames == n and iq1 is not None:
        A1 = anatomy_stack(iq1)
        if len(A1) == b1.n_frames:
            rp = np.asarray(b3.r_peaks_ms)
            rr = median_rr_ms(rp)
            ph1 = _phase(b1.frame_times() + b1.frame_ms / 2, rp)
            _, ph3 = chronological_times(folder, n, SHIFT)
            s_b1 = buffer1_curve(A1, A3, ph1, ph3, rr)
            s_em = expected_motion_curve(A1, A3, ph1, ph3, b3.frame_ms)
            if np.isfinite(s_b1).any():
                comb = _zs(s_b1) + (_zs(s_em) if s_em is not None else 0.0) + W_CONTINUITY * _zs(cont)
                comb = np.where(np.isfinite(comb), comb, -np.inf)
                m_first, m_margin = best_and_margin(comb, lower_is_better=False)
                det.update(combined_first=m_first, combined_margin=round(m_margin, 3),
                           buffer1_first=int(np.nanargmax(s_b1)),
                           expected_motion_first=int(np.nanargmax(s_em)) if s_em is not None else -1)
    if tc is not None:
        first, method = tc["c"], "trigger-count"
        if comb is not None:
            det["combined_agrees"] = bool(det["combined_first"] == first)
        det["continuity_agrees"] = bool(c_first == first)
    elif comb is not None and det["combined_margin"] >= MIN_MARGIN_COMBINED:
        first, method = det["combined_first"], "combined"
    elif c_margin >= MIN_MARGIN_CONTINUITY:
        first, method = c_first, "continuity"
    else:
        det["reason"] = "no trigger count; " + (
            "combined margin below threshold" if comb is not None else "no trustworthy ECG / buffer 1") + (
            " and continuity margin below threshold")
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


def is_current(rec):
    """True when an unwrap record was written by this VERSION (older ones are redone)."""
    return bool(rec) and int(rec.get("version", 1)) >= VERSION


def file_head(rec):
    """Stored slot at frame 0 of a file as it is now: an earlier unwrap's head, else 0 (stored order)."""
    return int(rec["first_frame"]) if rec and rec.get("buffer_unwrapped") and int(rec["first_frame"]) >= 0 else 0


def _apply(path, est, head_now, n, per_key, folder, log, gif):
    """Rotate one file from ``head_now`` to the estimate's head (stored order when unresolved) and
    write the flags. Returns True when frames moved."""
    target = est.first if est.resolved else 0
    rot = (target - head_now) % n
    times, phases = chronological_times(folder, n, est.shift) if est.resolved else (None, None)
    extra = _extra(est, times, phases, time.strftime("%Y-%m-%d %H:%M:%S"))
    ts = ((times - times[0]) * 1e-3).astype(np.float32) if times is not None else None
    is_iq = per_key != _RAW
    if rot:
        per = {per_key: None}
        if is_iq and ts is not None:
            per[f"{_BF}/timestamps"] = ts
        _rewrite(path, (np.arange(n) + rot) % n, per, extra)
    else:
        _write_inplace(path, extra, replace={f"{_BF}/timestamps": ts} if is_iq else None)
    from ..provenance import stamp_h5
    stamp_h5(str(path), extra={"stage": "unwrap", "buffer": BUFFER, "version": VERSION}, group="provenance_unwrap")
    if gif and rot and is_iq:
        try:
            from .gifs import gif_for_file
            gif_for_file(Path(path))
        except Exception as exc:                                       # noqa: BLE001
            log(f"  {Path(path).name}: GIF not re-rendered ({exc})")
    return bool(rot)


def unwrap_variants(folder, output_dir=None, gif=True, log=print):
    """Apply the head stored in the main buffer-3 file to the other buffer-3 IQ files of the folder.

    Reconstruction variants are beamformed from the same (stored-order) RF, so they carry the same
    rotation; they are reordered with the stored head, never re-estimated. Idempotent. Returns the
    list of files changed."""
    output_dir = Path(output_dir) if output_dir else Path(folder) / "output"
    conv, iq = buffer_files(output_dir)
    rec = read_flag(iq) or read_flag(conv)
    if not is_current(rec):
        return []                                  # the main file is unwrapped (by this VERSION) first
    est = _record_estimate(rec)
    changed = []
    for path in sorted(output_dir.glob(f"*_buffer{BUFFER}_*_iq.hdf5")):
        vrec = read_flag(path)
        if iq is not None and path.samefile(iq) or is_current(vrec):
            continue
        with h5py.File(path, "r") as f:
            if f"{_BF}/values" not in f:
                continue
            n = f[f"{_BF}/values"].shape[0]
        if n != int(est.details.get("n_frames", n)):
            log(f"  {path.name}: {n} frames, main file {est.details.get('n_frames')} - left alone")
            continue
        moved = _apply(path, est, file_head(vrec), n, f"{_BF}/values", folder, log, gif)
        changed.append(path)
        log(f"  {path.name}: {'reordered' if moved else 'flagged'} ({est.status}, head {est.first})")
    return changed


def _unwrap_main(folder, output_dir=None, dry_run=False, gif=True, log=print):
    """Estimate the head of buffer 3 and rewrite its converted RF and IQ files chronologically.

    Safe to re-run: a file flagged by this VERSION is never permuted again. A file flagged by an
    older VERSION is re-estimated and rotated from its current head to the new one. If one file is
    flagged (this VERSION) and the other is not (e.g. reconverted from the .mat), the stored head is
    reused. Each file is rotated from where it is now (``file_head``) to the target head.
    Returns the Estimate (or the stored record).
    """
    output_dir = Path(output_dir) if output_dir else Path(folder) / "output"
    conv, iq = buffer_files(output_dir)
    fc, fi = read_flag(conv), read_flag(iq)
    if (conv is None or is_current(fc)) and (iq is None or is_current(fi)):
        log(f"  buffer 3: already {(fc or fi or {}).get('status', 'absent')} (v{VERSION}) - nothing to do")
        return fc or fi
    # where each file is now; an unflagged IQ written after the converted file was unwrapped was
    # re-beamformed from it and carries the converted file's head
    head_conv = file_head(fc)
    iq_from_unwrapped = bool(fc and not fi and iq is not None and fc.get("buffer_unwrapped")
                             and os.path.getmtime(iq) > time.mktime(time.strptime(fc["time"], "%Y-%m-%d %H:%M:%S")))
    head_iq = head_conv if iq_from_unwrapped else file_head(fi)
    cur = fc if is_current(fc) else fi if is_current(fi) else None
    if cur is not None:
        est = _record_estimate(cur)
        log(f"  buffer 3: reusing stored head (slot {est.first}, {est.status})")
    else:
        est = estimate(folder, output_dir, current_head=head_iq)
        old = fi or fc
        log(f"  buffer 3: {est.status} - first slot {est.first}, shift {est.shift:+d} ({est.method}; "
            + ", ".join(f"{k} {v}" for k, v in est.details.items() if "first" in k or "margin" in k) + ")"
            + (f"  [was v{old.get('version', 1)} {old.get('status')} slot {old.get('first_frame')}]" if old else ""))
    if dry_run or est.status == "no-data":
        return est
    if conv is not None and not is_current(fc):
        with h5py.File(conv, "r") as f:
            n = f[_RAW].shape[0]
        _apply(conv, est, head_conv, n, _RAW, folder, log, gif)
    if iq is not None and not is_current(fi):
        with h5py.File(iq, "r") as f:
            n = f[f"{_BF}/values"].shape[0]
        _apply(iq, est, head_iq, n, f"{_BF}/values", folder, log, gif)
    return est


def summary(est):
    return dict(status=est.status, first=est.first, shift=est.shift, method=est.method,
                **{k: v for k, v in est.details.items()}) if isinstance(est, Estimate) else dict(est or {})
