"""Stage B / anatomical M-line: displacement along a hand-drawn line over time.

A shear wave is only interpretable once the 2-D displacement movie is reduced to a
**space-time plot** `D(s, t)` - displacement as a function of position `s` along an
anatomical line (e.g. down the interventricular septum from the valve) and slow time `t`.
A propagating wave then shows up as a **tilted band**, whose slope `ds/dt` is the shear-wave
speed (the Radon/slope-fit step, later).

This module builds that line and samples it:

1. :func:`select_mline` shows a B-mode frame and lets the user click points along the
   anatomy **in any order** - they are ordered along the line's principal axis, and the
   fitted spline is drawn live and can be fine-tuned by dragging points (see
   :class:`MLineSelector`). For **passive** (buffer 4) the B-mode is that same buffer; for
   **active** (buffer 2) it is the co-registered buffer-5 frame. With >=2 points a spline
   is fit *through* the points (a straight line for exactly two).
2. :func:`sample_along_line` interpolates the displacement movie onto the spline for every
   slow-time frame, giving `D(s, t)`. Because every zea file stores per-pixel
   ``coordinates`` in metres, the line (in metres) maps into *any* displacement grid - so a
   line drawn on the wide buffer-5 image samples correctly into the smaller buffer-2 ROI,
   with points outside that ROI returned as NaN.
3. :func:`spacetime_plot` renders `D(s, t)` with a real time axis, optionally cropped to a
   time window (e.g. around mitral or aortic valve closure).

:func:`process_mline` chains these into the intended order: pick the M-line -> sample the
full-measurement displacement along it (overview) -> auto-detect strong bursts along the
line and place a fixed window around each (:func:`detect_line_bursts`) -> **re-filter the
IQ inside each short window** (:func:`reprocess_window`; short-window clutter + band-pass
differ from filtering the whole record) -> a space-time plot per window.

The line is held **fixed** across each window: a valve-closure event lasts only tens of ms,
over which tissue barely moves, so a static line is a good approximation. Selection is
interactive (matplotlib), but every function also accepts pre-supplied points, so the whole
path is scriptable and testable head-less.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from zea import File


@dataclass
class BurstWindow:
    """A candidate valve-closure event: a time window around a displacement energy burst."""

    t_peak: float            # time of the energy peak, seconds
    t0: float                # window start, seconds (burst half-max left edge, padded)
    t1: float                # window end, seconds
    score: float             # peak energy (relative burst strength)
    label: str = ""          # optional user label, e.g. "mitral" / "aortic"

    @property
    def window(self) -> tuple[float, float]:
        return (self.t0, self.t1)


@dataclass
class MLine:
    """A sampled anatomical line in the imaging plane (metres)."""

    x: np.ndarray            # (n_s,) lateral, metres
    z: np.ndarray            # (n_s,) axial/depth, metres
    s: np.ndarray            # (n_s,) arc length from the start point, metres
    points: np.ndarray       # (n_pts, 2) original clicked [x, z], metres

    @property
    def length(self) -> float:
        return float(self.s[-1])


# ---------------------------------------------------------------------------
def fit_spline(points_xz, n_samples: int = 250, smoothing: float = 0.0) -> MLine:
    """Fit a spline through clicked ``(x, z)`` points (metres) and sample it uniformly.

    ``smoothing=0`` interpolates *through* every point (the requested behaviour); the
    spline degree is ``min(3, n_points-1)`` so two points give a straight line. Returns an
    :class:`MLine` with cumulative arc length ``s``.
    """
    pts = np.asarray(points_xz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("Need at least two (x, z) points in metres to fit an M-line.")
    k = min(3, pts.shape[0] - 1)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], k=k, s=smoothing)
    u = np.linspace(0.0, 1.0, int(n_samples))
    x, z = splev(u, tck)
    x, z = np.asarray(x), np.asarray(z)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    return MLine(x=x, z=z, s=s, points=pts)


def _grid_axes(coords: np.ndarray):
    """Return the regular ``(xs, zs)`` axis vectors (metres) of a ``(z, x, 3)`` grid."""
    coords = np.asarray(coords)
    xs = coords[0, :, 0]        # lateral varies along axis 1
    zs = coords[:, 0, 2]        # depth varies along axis 0
    return xs, zs


def _to_fractional_index(values, axis):
    """Map physical positions to fractional pixel indices on a regular ``axis``.

    Robust to axis direction/spacing. Positions outside the axis range map outside
    ``[0, n-1]`` so the caller can mask them.
    """
    n = len(axis)
    span = axis[-1] - axis[0]
    if span == 0:
        raise ValueError("Degenerate grid axis (zero span).")
    return (np.asarray(values) - axis[0]) / span * (n - 1)


def sample_along_line(field_txz: np.ndarray, coords: np.ndarray, mline: MLine,
                      order: int = 1) -> np.ndarray:
    """Sample a ``(T, z, x)`` field along ``mline`` for every frame -> ``D(s, t)``.

    Points that fall outside the field's grid (e.g. a buffer-5 line reaching past the
    buffer-2 ROI) are returned as ``NaN``. Bilinear interpolation by default.
    """
    xs, zs = _grid_axes(coords)
    cols = _to_fractional_index(mline.x, xs)
    rows = _to_fractional_index(mline.z, zs)
    inside = ((rows >= 0) & (rows <= len(zs) - 1) & (cols >= 0) & (cols <= len(xs) - 1))

    T = field_txz.shape[0]
    coordinates = np.stack([rows, cols])           # (2, n_s)
    out = np.empty((mline.s.size, T), dtype=np.float32)
    for t in range(T):
        out[:, t] = map_coordinates(field_txz[t], coordinates, order=order,
                                    mode="nearest", prefilter=(order > 1))
    out[~inside, :] = np.nan
    return out


def _line_normals(mline: MLine):
    """Unit normal ``(nx, nz)`` per M-line point (tangent rotated 90 deg).

    Works for any orientation - horizontal, 45 deg or vertical - so a perpendicular offset
    always samples *neighbouring* tissue rather than sliding along the line.
    """
    tx = np.gradient(np.asarray(mline.x, dtype=float))
    tz = np.gradient(np.asarray(mline.z, dtype=float))
    norm = np.hypot(tx, tz) + 1e-20
    return (-tz / norm, tx / norm)          # rotate tangent by 90 deg


def sample_along_line_median(field_txz, coords, mline: MLine, n_lines: int = 11,
                             spacing_px: float = 1.0, order: int = 1,
                             spacing_m: float | None = None) -> np.ndarray:
    """``D(s, t)`` = **median over ``n_lines`` copies of the M-line offset perpendicular to it**.

    The line is duplicated ``n_lines`` times, each shifted along its **local normal** by a
    multiple of the spacing (centred on the original), sampled, and the median taken per
    ``(s, t)``. Offsetting perpendicular (not just in depth) means it works for an M-line at
    any angle - horizontal, diagonal or vertical. Averaging this thin parallel band
    suppresses per-line speckle while keeping the wavefront. ``n_lines <= 1`` = single line.

    ``spacing_m`` sets the physical line spacing (metres); if ``None`` it is ``spacing_px``
    times the axial pixel pitch (backwards-compatible).
    """
    if n_lines <= 1:
        return sample_along_line(field_txz, coords, mline, order=order)
    if spacing_m is None:
        dz = float(np.median(np.abs(np.diff(np.asarray(coords)[:, 0, 2]))))
        spacing_m = float(spacing_px) * dz
    nx, nz = _line_normals(mline)
    offsets = (np.arange(n_lines) - n_lines // 2) * float(spacing_m)
    stack = np.stack([
        sample_along_line(field_txz, coords,
                          dataclasses.replace(mline, x=mline.x + off * nx, z=mline.z + off * nz),
                          order=order)
        for off in offsets
    ])
    return np.nanmedian(stack, axis=0).astype(np.float32)


# =============== M-line tracking through a B-mode cine (per-frame adaptation) ==============
def _index_to_physical(idx, axis):
    """Fractional pixel index -> physical position (m) on a regular ``axis`` (inverse of
    :func:`_to_fractional_index`)."""
    axis = np.asarray(axis, float)
    return np.interp(np.asarray(idx, float), np.arange(axis.size), axis)


def _extract_template(img, row, col, pr):
    """Zero-mean ``(2*pr+1)`` patch centred at ``(row, col)`` + its L2 norm (clipped to bounds)."""
    nz, nx = img.shape
    row = int(np.clip(round(row), pr, nz - 1 - pr))
    col = int(np.clip(round(col), pr, nx - 1 - pr))
    t = img[row - pr:row + pr + 1, col - pr:col + pr + 1].astype(np.float64)
    t = t - t.mean()
    return t, float(np.sqrt((t * t).sum()) + 1e-12)


def _ncc_match(tmpl, tn, cur, row, col, pr, sr):
    """Find where a **fixed template** ``tmpl`` best matches (NCC) in ``cur``, searching
    ``+/-sr`` px around ``(row, col)``. Returns ``(row2, col2, ncc_peak)`` with parabolic
    sub-pixel refinement. Vectorised over the search window (``sliding_window_view``).

    A *fixed* template (from the manual frame) - rather than one updated each step - avoids
    template drift: sequential re-templating lets a match that lands slightly off-centre lock
    onto a wall edge and accumulate, which is why only the seed frame and its neighbours stayed
    centred. Centring the *search* on the predicted position still allows large cumulative motion.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    nz, nx = cur.shape
    row = int(np.clip(round(row), pr, nz - 1 - pr))
    col = int(np.clip(round(col), pr, nx - 1 - pr))
    r_lo, r_hi = max(row - sr, pr), min(row + sr, nz - 1 - pr)
    c_lo, c_hi = max(col - sr, pr), min(col + sr, nx - 1 - pr)
    reg = cur[r_lo - pr:r_hi + pr + 1, c_lo - pr:c_hi + pr + 1].astype(np.float64)
    win = sliding_window_view(reg, (2 * pr + 1, 2 * pr + 1))
    wc = win - win.mean(axis=(2, 3), keepdims=True)
    num = (wc * tmpl).sum(axis=(2, 3))
    den = np.sqrt((wc * wc).sum(axis=(2, 3))) * tn + 1e-12
    ncc = num / den
    ia, ib = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    peak = float(ncc[ia, ib])

    def _sub(i, arr):
        if 0 < i < arr.size - 1:
            y0, y1, y2 = arr[i - 1], arr[i], arr[i + 1]
            d = y0 - 2 * y1 + y2
            if abs(d) > 1e-9:
                return i + 0.5 * (y0 - y2) / d
        return float(i)

    return r_lo + _sub(ia, ncc[:, ib]), c_lo + _sub(ib, ncc[ia, :]), peak


def track_mline_cine(bmode_iq_path, mline: MLine, frame0: int, patch_mm: float = 3.0,
                     search_mm: float = 6.0, min_ncc: float = 0.3, smooth: int = 3,
                     n_samples: int | None = None, dynamic_range=(-50, 0),
                     snap: bool = True, snap_window_mm: float = 6.0, snap_n_nodes: int = 21,
                     snap_min_prom: float = 0.2):
    """Adapt an M-line to **every frame of a B-mode cine** by tracking its control points.

    The active buffer-2 pushes are interleaved with buffer-5 B-mode frames (frame ``m`` <-> push
    ``m``), spread over ~1 cardiac cycle, so the wall - and hence the M-line - moves between
    pushes. Starting from a single manual line on frame ``frame0``, each control point is tracked
    **outward** (``frame0 -> +/-1 -> ...``) by NCC block matching (:func:`_ncc_match`) of a
    **fixed template taken from ``frame0``** (the wall-centre appearance the user drew - avoids
    the edge-locking drift of a per-step-updated template) on the log-compressed B-mode, with the
    search centre following the predicted position. Only the **line-normal component** of each
    match is kept (the wall is self-similar along its length -> along-wall matches are ambiguous).
    Trajectories are then cleaned **image-only**: points below ``min_ncc`` are dropped and linearly
    interpolated over frame index, then lightly smoothed (median-3 + moving-average ``smooth``) -
    gentle enough to preserve the real (periodic) cardiac motion. A spline is re-fit per frame.

    Args:
        bmode_iq_path: the co-registered B-mode cine (buffer-5 ``*_iq.hdf5``).
        mline: the manual seed line (control points in metres).
        frame0: the cine frame the seed line was drawn on.
        patch_mm/search_mm: NCC template size / search half-window (mm). ``search_mm`` should
            exceed the max inter-frame wall motion (~1 cm covers fast systole).
        min_ncc: below this match quality a frame's point is treated as lost (interpolated).
        smooth: moving-average length over frames (light temporal regularisation).
        snap: after NCC coarse tracking, **snap each frame's line onto the septum's intensity
            centre** (:func:`snap_to_band`) so the line sits mid-band consistently for every push
            - NCC aligns speckle texture, not the bright band, so its offset from the band centre
            drifts with cardiac phase; the snap removes that (per-push offset ~0 vs +/-4 mm).
        snap_window_mm/snap_n_nodes/snap_min_prom: band-snap search half-window, number of nodes,
            and minimum band prominence (fraction of local max) to accept a snap.

    Returns:
        ``(lines, quality, coords)`` - ``lines`` a ``{frame_index: MLine}`` dict, ``quality`` a
        ``(n_frames, n_pts)`` NCC-confidence array, ``coords`` the cine grid ``(z, x, 3)`` (m).
    """
    from scipy.ndimage import median_filter, uniform_filter1d

    with File(str(bmode_iq_path)) as f:
        bd = f.data.beamformed_data
        vals = np.asarray(bd.values[:])
        coords = np.asarray(bd.coordinates[:], dtype=np.float32)
    env = np.sqrt(vals[..., 0] ** 2 + vals[..., 1] ** 2).astype(np.float32)
    peak = env.max(axis=(1, 2), keepdims=True)
    peak[peak == 0] = 1.0
    db = np.clip(20.0 * np.log10(env / peak + 1e-6), dynamic_range[0], dynamic_range[1])

    n = db.shape[0]
    xs, zs = _grid_axes(coords)
    dz = float(np.median(np.abs(np.diff(zs))))          # metres
    dx = float(np.median(np.abs(np.diff(xs))))          # metres
    pr = max(2, int(round(patch_mm * 1e-3 / (2.0 * dz))))   # mm -> px radius
    sr = max(2, int(round(search_mm * 1e-3 / dz)))
    frame0 = int(np.clip(frame0, 0, n - 1))

    pts = np.asarray(mline.points, dtype=float)                 # (npts, 2) [x, z] m
    npts = pts.shape[0]
    # Per-control-point unit NORMAL, in pixel space. The wall is self-similar along its length
    # (aperture problem: NCC slides freely along the wall), so we keep only the well-constrained
    # ACROSS-wall (normal) component of each match and discard the ambiguous along-wall slide -
    # exactly the motion that carries the line off the wall.
    tx = np.gradient(pts[:, 0]); tz = np.gradient(pts[:, 1])     # tangent (x, z) metres
    n_col, n_row = -tz, tx                                       # rotate 90 deg -> normal (x, z)
    n_col, n_row = n_col / dx, n_row / dz                        # -> pixel-space direction
    nn = np.hypot(n_col, n_row) + 1e-20
    n_col, n_row = n_col / nn, n_row / nn

    rows = np.full((n, npts), np.nan)
    cols = np.full((n, npts), np.nan)
    qual = np.zeros((n, npts))
    r0 = _to_fractional_index(pts[:, 1], zs)
    c0 = _to_fractional_index(pts[:, 0], xs)
    rows[frame0], cols[frame0], qual[frame0] = r0, c0, 1.0

    # Fixed templates from the seed frame (the wall-centre appearance the user drew) - matched
    # into every frame to avoid drift; only the search *centre* follows the predicted position.
    templ = [_extract_template(db[frame0], r0[i], c0[i], pr) for i in range(npts)]

    for direction in (1, -1):                                   # walk outward from frame0
        rp, cp = r0.copy(), c0.copy()
        f = frame0 + direction
        while 0 <= f < n:
            for i in range(npts):
                tm, tn = templ[i]
                rr, cc, q = _ncc_match(tm, tn, db[f], rp[i], cp[i], pr, sr)
                proj = (rr - rp[i]) * n_row[i] + (cc - cp[i]) * n_col[i]   # normal component only
                rows[f, i] = rp[i] + proj * n_row[i]
                cols[f, i] = cp[i] + proj * n_col[i]
                qual[f, i] = q
            rp, cp = rows[f].copy(), cols[f].copy()              # search centre = predicted position
            f += direction

    X = _index_to_physical(cols, xs)                            # (n, npts) metres
    Z = _index_to_physical(rows, zs)
    idx = np.arange(n)
    for i in range(npts):
        good = qual[:, i] >= min_ncc
        if good.sum() >= 2:
            X[:, i] = np.interp(idx, idx[good], X[good, i])
            Z[:, i] = np.interp(idx, idx[good], Z[good, i])
        if smooth and smooth > 1:
            X[:, i] = uniform_filter1d(median_filter(X[:, i], 3, mode="nearest"),
                                       int(smooth), mode="nearest")
            Z[:, i] = uniform_filter1d(median_filter(Z[:, i], 3, mode="nearest"),
                                       int(smooth), mode="nearest")

    ns = n_samples or int(mline.s.size)
    lines = {fr: fit_spline(np.column_stack([X[fr], Z[fr]]), n_samples=ns) for fr in range(n)}
    if snap:
        # NCC gave the right *region* (which structure, coarse position); now centre each line on
        # the septum's bright band along the normal, for a consistent, phase-invariant placement.
        lines = {fr: snap_to_band(env[fr], lines[fr], coords, window_mm=snap_window_mm,
                                  n_nodes=snap_n_nodes, min_prom_frac=snap_min_prom)
                 for fr in range(n)}
    return lines, qual, coords


# ---------------------------------------------------------------------------
def _band_center_offset(prof, offsets, min_prom_frac=0.2):
    """Signed offset (m, same units as ``offsets``) from 0 to the **FWHM-midpoint of the dominant
    bright band** in an along-normal intensity profile, or ``None`` if no prominent band.

    ``offsets`` is the (monotonic) normal-offset axis the profile was sampled on (0 = current
    node). The band is the main peak; its half-maximum crossings bracket the wall, and their
    midpoint is the wall centre - robust to one edge being brighter than the other (unlike the
    peak) and to background (unlike a plain windowed centroid).
    """
    from scipy.ndimage import uniform_filter1d

    p = uniform_filter1d(np.asarray(prof, float), 3)
    if p.size < 5 or p.max() <= 0:
        return None
    base, pk = float(p.min()), int(np.argmax(p))
    prom = p[pk] - base
    if prom < min_prom_frac * float(p.max()):
        return None
    half = base + 0.5 * prom
    a = pk
    while a > 0 and p[a] > half:
        a -= 1
    b = pk
    while b < p.size - 1 and p[b] > half:
        b += 1
    mid = 0.5 * (a + b)
    return float(np.interp(mid, np.arange(p.size), np.asarray(offsets, float)))


def snap_to_band(bmode_frame, mline: MLine, coords, window_mm: float = 6.0, n_nodes: int = 21,
                 min_prom_frac: float = 0.2, order: int = 1) -> MLine:
    """Re-centre an M-line onto the **septum's bright band** (intensity centre), per column.

    At ``n_nodes`` points along ``mline`` the (linear-envelope) ``bmode_frame`` is sampled along
    the **line normal** over ``+/-window_mm``; each node is moved to the FWHM-midpoint of the
    dominant band there (:func:`_band_center_offset`). Nodes with no prominent band keep their
    position. The node offsets are lightly smoothed along the line and a spline is re-fit. This
    turns "centre of the septum" into an explicit, appearance-based, phase-invariant definition,
    fixing the phase-dependent bias of pure speckle (NCC) tracking. The incoming ``mline`` (from
    NCC) is the prior that says *where to look* - the snap only refines perpendicular placement
    within ``window_mm``, so it cannot jump to a different structure.
    """
    from scipy.ndimage import median_filter, uniform_filter1d

    xs, zs = _grid_axes(coords)
    dz = float(np.median(np.abs(np.diff(zs))))
    nx_u, nz_u = _line_normals(mline)                       # per-sample unit normal (x, z)
    ns = mline.s.size
    idx = np.unique(np.linspace(0, ns - 1, min(n_nodes, ns)).round().astype(int))
    offs = np.arange(-window_mm * 1e-3, window_mm * 1e-3 + dz, dz)

    nx_node, nz_node = [], []
    for i in idx:
        x0, z0, nxv, nzv = mline.x[i], mline.z[i], nx_u[i], nz_u[i]
        xp, zp = x0 + offs * nxv, z0 + offs * nzv
        cols = _to_fractional_index(xp, xs)
        rows = _to_fractional_index(zp, zs)
        prof = map_coordinates(bmode_frame, np.stack([rows, cols]), order=order, mode="nearest")
        off = _band_center_offset(prof, offs, min_prom_frac)
        off = 0.0 if off is None else off
        nx_node.append(x0 + off * nxv)
        nz_node.append(z0 + off * nzv)

    nx_node = uniform_filter1d(median_filter(np.array(nx_node), 3, mode="nearest"), 3, mode="nearest")
    nz_node = uniform_filter1d(median_filter(np.array(nz_node), 3, mode="nearest"), 3, mode="nearest")
    return fit_spline(np.column_stack([nx_node, nz_node]), n_samples=ns)


# ---------------------------------------------------------------------------
def _bmode_from_iq_frame(values_iq: np.ndarray, dynamic_range=(-50, 0)) -> np.ndarray:
    """Single-frame ``(z, x, 2=[I,Q])`` -> 8-bit B-mode ``(z, x)`` for display."""
    from zea.display import to_8bit

    env = np.sqrt(values_iq[..., 0] ** 2 + values_iq[..., 1] ** 2).astype(np.float32)
    peak = env.max()
    if peak > 0:
        env /= peak
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(env + 1e-12)
    return to_8bit(db, dynamic_range, pillow=False)


def load_bmode_frame(bmode_iq_path, frame: int):
    """Read one B-mode frame + its grid coords from a beamformed IQ file (lazy slice)."""
    with File(str(bmode_iq_path)) as f:
        bdata = f.data.beamformed_data
        n = bdata.values.shape[0]
        frame = int(np.clip(frame, 0, n - 1))
        vals = np.asarray(bdata.values[frame])          # (z, x, 2), single-frame read
        coords = np.asarray(bdata.coordinates[:], dtype=np.float32)
    return _bmode_from_iq_frame(vals), coords, n


# Backends that cannot pop up a window for clicking (everything else - TkAgg, QtAgg,
# GTK*Agg, macosx, ... - can, despite the "agg" suffix; the old check wrongly excluded them).
_NON_GUI_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}


def _is_gui_backend(name: str) -> bool:
    """True if the matplotlib backend can display an interactive window for ``ginput``."""
    b = name.lower()
    if b in _NON_GUI_BACKENDS:
        return False
    if b.startswith("module://") and "inline" in b:   # notebook inline: no ginput
        return False
    return True


def _ensure_gui_backend() -> str:
    """Make sure an interactive backend is active, switching to one if needed.

    Returns the active backend name. Raises if none can be used (truly head-less), with a
    message pointing at the scriptable ``points`` alternative.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if _is_gui_backend(matplotlib.get_backend()):
        return matplotlib.get_backend()
    for cand in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            plt.switch_backend(cand)
            return matplotlib.get_backend()
        except Exception:
            continue
    raise RuntimeError(
        f"No interactive matplotlib backend available (current: {matplotlib.get_backend()}). "
        "Install a GUI toolkit (e.g. tkinter) and run locally, or skip the picker by passing "
        "pre-chosen points via MLineConfig.points / fit_spline()."
    )


def _order_points(points_xz, anchor=None) -> np.ndarray:
    """Order scattered points into a smooth open path (so click order does not matter).

    Points are sorted by their projection onto the **principal axis** (PCA) of the set -
    the dominant direction the line runs - so clicking e.g. left, right, then centre still
    yields left -> centre -> right. If ``anchor`` (an ``(x, z)`` point) is given, the path
    is oriented to *start* at the end nearest the anchor, which keeps ``s = 0`` at, e.g.,
    the valve (taken as the first clicked point).
    """
    p = np.asarray(points_xz, dtype=float)
    if p.shape[0] <= 2:
        ordered = p.copy()
    else:
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        ordered = p[np.argsort(centred @ vt[0])]
    if anchor is not None and ordered.shape[0] >= 2:
        a = np.asarray(anchor, dtype=float)
        if np.hypot(*(ordered[0] - a)) > np.hypot(*(ordered[-1] - a)):
            ordered = ordered[::-1]
    return ordered


class MLineSelector:
    """Interactive B-mode point editor with a live M-line preview.

    Click points in **any order** (they are ordered along the line automatically); the
    fitted spline is drawn and updated live once there are >= 2 points, so it can be
    fine-tuned. Interactions: left-click adds a point; left-drag on a point moves it;
    right-click deletes the nearest; Backspace removes the last; **Enter** finishes.
    """

    def __init__(self, ax, min_points: int = 2, n_samples: int = 250):
        self.ax = ax
        self.fig = ax.figure
        self.min_points = min_points
        self.n_samples = n_samples
        self.points: list[tuple[float, float]] = []   # (x_mm, z_mm), insertion order; [0] anchors s=0
        self.dragging: int | None = None
        (self._pts_artist,) = ax.plot([], [], "o", color="yellow", ms=8, mec="k", zorder=5)
        (self._line_artist,) = ax.plot([], [], "-", color="cyan", lw=2.0, zorder=4)
        xr = abs(np.diff(ax.get_xlim())[0]); zr = abs(np.diff(ax.get_ylim())[0])
        self._pick_r = 0.04 * max(xr, zr)             # grab/delete radius, mm
        self._done = False
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self.on_press),
            self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion),
            self.fig.canvas.mpl_connect("button_release_event", self.on_release),
            self.fig.canvas.mpl_connect("key_press_event", self.on_key),
            self.fig.canvas.mpl_connect("close_event", lambda _e: self._finish()),
        ]

    def _toolbar_active(self) -> bool:
        """True while a pan/zoom tool is engaged (so clicks don't add stray points)."""
        mgr = getattr(self.fig.canvas, "manager", None)
        tb = getattr(mgr, "toolbar", None) or getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    def _nearest(self, x, z):
        if not self.points:
            return None, np.inf
        d = [np.hypot(px - x, pz - z) for px, pz in self.points]
        i = int(np.argmin(d))
        return i, d[i]

    def on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None or self._toolbar_active():
            return
        if e.button == 1:                                  # add, or grab an existing point
            i, dist = self._nearest(e.xdata, e.ydata)
            if i is not None and dist <= self._pick_r:
                self.dragging = i
            else:
                self.points.append((e.xdata, e.ydata))
                self.dragging = len(self.points) - 1
            self._redraw()
        elif e.button == 3:                                # delete nearest
            i, dist = self._nearest(e.xdata, e.ydata)
            if i is not None and dist <= self._pick_r:
                self.points.pop(i)
                self._redraw()

    def on_motion(self, e):
        if self.dragging is None or e.inaxes != self.ax or e.xdata is None:
            return
        self.points[self.dragging] = (e.xdata, e.ydata)
        self._redraw()

    def on_release(self, _e):
        self.dragging = None

    def on_key(self, e):
        if e.key in ("enter", "return"):
            self._finish()
        elif e.key in ("backspace", "delete") and self.points:
            self.points.pop()
            self._redraw()

    def ordered_points(self):
        """Current points ordered along the line, anchored at the first clicked point."""
        return _order_points(np.asarray(self.points, dtype=float),
                             anchor=self.points[0] if self.points else None)

    def _redraw(self):
        px = [p[0] for p in self.points]; pz = [p[1] for p in self.points]
        self._pts_artist.set_data(px, pz)
        if len(self.points) >= 2:
            try:
                ml = fit_spline(self.ordered_points(), n_samples=self.n_samples)  # mm preview
                self._line_artist.set_data(ml.x, ml.z)
            except Exception:
                pass                                        # keep last good line on a bad fit
        else:
            self._line_artist.set_data([], [])
        self.ax.set_title(f"{len(self.points)} pt - click to add / drag to move / "
                          f"right-click to delete / Enter to finish (>= {self.min_points})")
        self.fig.canvas.draw_idle()

    def _finish(self):
        if self._done:
            return
        self._done = True
        for cid in self._cids:
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:
            pass

    def run(self):
        """Block until the user finishes; return the clicked points (mm, insertion order)."""
        import matplotlib.pyplot as plt

        self._redraw()
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(timeout=0)         # until Enter / window close
        pts = list(self.points)
        plt.close(self.fig)
        return pts


def select_mline(bmode_u8, coords, min_points: int = 2, n_samples: int = 250,
                 title: str | None = None) -> MLine:
    """Interactively pick + fine-tune an M-line on a B-mode frame; return the :class:`MLine`.

    Click points in **any order**; the line is drawn live once >= 2 points exist and can be
    fine-tuned by dragging points (see :class:`MLineSelector`). Needs a GUI matplotlib
    backend - one is selected automatically if the current one cannot show a window; if
    none exists it raises, and you can pass points via :class:`MLineConfig` / :func:`fit_spline`.
    """
    _ensure_gui_backend()
    import matplotlib.pyplot as plt

    xs, zs = _grid_axes(coords)
    extent = [xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3]   # mm; depth downward
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(bmode_u8, cmap="gray", extent=extent, aspect="auto")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    if title:
        fig.suptitle(title)

    print("\n>>> M-LINE: left-click points along the anatomy in ANY order (e.g. the two ends "
          "then the middle);\n>>> the cyan line updates live once you have >= 2 points. Drag a "
          "point to move it, right-click to\n>>> delete one. Press ENTER (figure focused) to "
          "finish.\n")
    pts_mm = MLineSelector(ax, min_points=min_points, n_samples=n_samples).run()
    if len(pts_mm) < min_points:
        raise ValueError(
            f"Need at least {min_points} points, got {len(pts_mm)}. (Click on the image "
            "window, then press ENTER; the window must have focus for clicks to register.)"
        )
    ordered_m = _order_points(np.asarray(pts_mm, dtype=float), anchor=pts_mm[0]) * 1e-3  # mm->m
    return fit_spline(ordered_m, n_samples=n_samples)


def draw_mline_on_bmode(bmode_u8, coords, mline: MLine, out_path, title=None):
    """Save a PNG of the chosen M-line drawn on the B-mode (a record of what was sampled)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    xs, zs = _grid_axes(coords)
    extent = [xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(bmode_u8, cmap="gray", extent=extent, aspect="auto")
    ax.plot(mline.x * 1e3, mline.z * 1e3, "-", color="cyan", lw=1.8)
    ax.plot(mline.points[:, 0] * 1e3, mline.points[:, 1] * 1e3, "o", color="yellow", ms=5)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title(title or f"M-line ({mline.length * 1e3:.1f} mm)")
    fig.tight_layout(); fig.savefig(str(out_path), dpi=110); plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def spacetime_plot(D_st, s_m, t_s, out_path, title="", value_unit="um",
                   percentile: float = 99.0, time_window=None):
    """Render a space-time plot ``D(s, t)`` (arc length vs time), optionally windowed.

    Args:
        D_st: ``(n_s, n_t)`` displacement (metres) along the line over time.
        s_m: ``(n_s,)`` arc length (metres). t_s: ``(n_t,)`` time (seconds).
        value_unit: ``"um"`` (default) or ``"mm"`` for the colour scale.
        percentile: symmetric colour range = this percentile of ``|D|``.
        time_window: optional ``(t0, t1)`` seconds to crop the time axis.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    t_s = np.asarray(t_s); s_mm = np.asarray(s_m) * 1e3
    scale = 1e6 if value_unit == "um" else 1e3
    D = np.asarray(D_st) * scale

    sel = slice(None)
    if time_window is not None:
        i0, i1 = np.searchsorted(t_s, time_window[0]), np.searchsorted(t_s, time_window[1])
        sel = slice(max(0, i0), max(i0 + 1, i1))
    D = D[:, sel]; tt = t_s[sel]

    c = float(np.nanpercentile(np.abs(D), percentile)) or 1.0
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(D, aspect="auto", cmap="seismic", vmin=-c, vmax=c,
                   extent=[tt[0], tt[-1], s_mm[-1], s_mm[0]])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("arc length s along M-line (mm)  [valve -> distal]")
    ax.set_title(title or "Displacement space-time  D(s, t)")
    cb = fig.colorbar(im, ax=ax); cb.set_label(f"displacement ({'µm' if value_unit=='um' else 'mm'})")
    fig.tight_layout(); fig.savefig(str(out_path), dpi=120); plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def _detect_peaks(energy, t_s, min_separation_s: float = 0.15, max_events: int = 6,
                  prominence_frac: float = 0.25) -> np.ndarray:
    """Pick the strongest, well-separated peaks of an energy signal -> sorted indices.

    Returns no peaks for a degenerate signal (fewer than 3 frames, or a non-increasing time
    axis), rather than crashing - a very short record (e.g. a 2-frame phantom buffer, whose
    displacement is 1 frame) has no burst to detect.
    """
    energy = np.asarray(energy, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    if energy.size < 3 or t_s.size < 2:
        return np.array([], dtype=int)
    dt = float(np.median(np.diff(t_s)))
    fps = 1.0 / dt if (np.isfinite(dt) and dt > 0) else 0.0
    distance = max(1, int(min_separation_s * fps)) if fps > 0 else 1
    prominence = prominence_frac * (np.nanmax(energy) - np.nanmedian(energy))
    peaks, _ = find_peaks(energy, distance=distance, prominence=max(prominence, 1e-12))
    if peaks.size == 0:
        return np.array([], dtype=int)
    return np.sort(peaks[np.argsort(energy[peaks])[::-1][:max_events]])


def energy_along_line(D_st, edge_frames: int = 30):
    """Per-frame energy (RMS over arc length) of an M-line space-time strip ``D(s, t)``.

    This is the burst-detection signal for the new flow: it measures activity **on the
    anatomy the user drew**, not a generic depth band. Returns ``(energy_raw, energy_masked)``
    with the first/last ``edge_frames`` zeroed (integration/filter transients).
    """
    e = np.sqrt(np.nanmean(np.asarray(D_st) ** 2, axis=0)).astype(float)   # over s -> (n_t,)
    masked = e.copy()
    # Never zero the whole signal: clamp the edge guard to leave a usable middle on short records.
    ef = min(int(edge_frames), max(0, (e.size - 1) // 2))
    if ef:
        masked[:ef] = 0.0
        masked[-ef:] = 0.0
    return e, masked


def detect_line_bursts(D_st, t_s, window_ms: float = 100.0, edge_frames: int = 30,
                       min_separation_s: float = 0.15, max_events: int = 4):
    """Detect bursts in the M-line displacement -> **fixed-width** windows around each peak.

    Centres a window of ``window_ms`` on each strong peak of the *along-line* energy - the
    "100 ms intervals around strong bursts" the windowed re-processing then runs on. Returns
    ``(windows, energy_raw)``.
    """
    t_s = np.asarray(t_s)
    e_raw, e = energy_along_line(D_st, edge_frames)
    peaks = _detect_peaks(e, t_s, min_separation_s, max_events)
    half = 0.5 * window_ms * 1e-3
    windows = [BurstWindow(t_peak=float(t_s[pk]),
                           t0=float(max(t_s[0], t_s[pk] - half)),
                           t1=float(min(t_s[-1], t_s[pk] + half)),
                           score=float(e[pk])) for pk in peaks]
    return windows, e_raw


def reprocess_window(swe_full, mline, window, preset: str = "passive", estimator: str = "kasai",
                     estimator_kwargs=None, analysis_pad_s: float = 0.0,
                     n_mlines: int = 11, mline_spacing_px: float = 1.0):
    """Re-filter + re-estimate the IQ **within one short window**, sampled along the M-line.

    Slices the full IQ to the window's frames (plus optional ``analysis_pad_s`` each side
    for filter roll-off), runs the Stage-B reconstruction on *just those frames* (so the
    filters see the local, transient statistics - the point of windowing), and samples the
    resulting displacement as the median over ``n_mlines`` shifted M-lines. Returns
    ``(D_st, t_win, (f0, f1))`` where ``D_st`` is ``(n_s, n_win)`` and ``t_win`` its times.
    """
    from swi_stage_b import reconstruct_displacement
    from swi_dsp import PRESETS

    fconf = PRESETS[preset]
    t = np.asarray(swe_full.timestamps)
    t0, t1 = window
    f0 = max(0, int(np.searchsorted(t, t0 - analysis_pad_s)))
    f1 = min(len(t), int(np.searchsorted(t, t1 + analysis_pad_s)) + 1)
    swe_win = dataclasses.replace(swe_full, iq=swe_full.iq[f0:f1], timestamps=t[f0:f1])

    _, disp = reconstruct_displacement(swe_win, fconf, estimator=estimator,
                                       estimator_kwargs=estimator_kwargs)
    D_st = sample_along_line_median(disp, swe_full.coords, mline, n_mlines, mline_spacing_px)
    t_win = t[f0:f1][: disp.shape[0]]
    return D_st, t_win, (f0, f1)


def plot_bursts(energy_raw, t_s, windows, out_path, title=""):
    """Save an energy-vs-time plot with the candidate windows shaded (the time axis)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    t_s = np.asarray(t_s)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_s, np.asarray(energy_raw) * 1e6, color="k", lw=1.0)
    for i, w in enumerate(windows):
        ax.axvspan(w.t0, w.t1, color="tab:orange", alpha=0.25)
        ax.axvline(w.t_peak, color="tab:red", lw=1.0)
        ax.text(w.t_peak, ax.get_ylim()[1] * 0.95, f"#{i}", ha="center", va="top",
                color="tab:red", fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("displacement RMS (µm)")
    ax.set_title(title or "Displacement energy vs time — candidate valve-closure windows")
    fig.tight_layout(); fig.savefig(str(out_path), dpi=120); plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
@dataclass
class MLineConfig:
    """Options for :func:`process_mline` (M-line -> detect bursts -> windowed reprocessing).

    Selection: ``points`` (pre-supplied ``(n, 2)`` metres) skips the picker; ``display_time``
    / ``display_frame`` choose the B-mode frame to draw on (default mid-recording).
    Windows: ``window_ms`` is the interval width around each detected burst; ``max_events``
    caps how many bursts; ``window_index`` (None = all) restricts to one; ``time_window``
    manually overrides detection with a single window. ``analysis_pad_ms`` adds IQ each side
    for filter roll-off (cropped from the plots). Reconstruction of each window uses
    ``estimator`` + ``preset`` (+ ``estimator_kwargs``), same as Stage B.
    """

    points: np.ndarray | None = None
    display_frame: int | None = None
    display_time: float | None = None
    # windowed reprocessing
    window_ms: float = 100.0
    max_events: int = 4
    window_index: int | None = None                  # process only burst #k (None = all detected)
    analysis_pad_ms: float = 0.0                      # extra IQ each side for filtering (cropped)
    time_window: tuple[float, float] | None = None    # manual override: one explicit window (s)
    # reconstruction of each window (matches Stage B)
    estimator: str = "kasai"
    preset: str = "passive"
    estimator_kwargs: dict = field(default_factory=dict)
    # line + display
    n_samples: int = 250
    n_mlines: int = 11                   # median over this many vertically-shifted M-lines
    mline_spacing_px: float = 1.0        # depth (z) spacing between shifted M-lines, pixels
    value_unit: str = "um"
    bmode_source: str | None = None      # active: buffer-5 IQ path; None -> use iq_path (passive)
    # per-window reconstruction grid
    fine_grid: bool = True               # re-beamform a FINE ROI around the M-line per window (default)
    fine_dz_um: float = 45.0             # fine axial pitch (resolves the carrier / axial estimate)
    fine_dx_um: float = 300.0            # fine lateral pitch


def _swe_path_for(iq_path: Path) -> Path:
    """The Stage-B result file for a beamformed IQ file (``*_iq.hdf5`` -> ``*_swe.hdf5``)."""
    return iq_path.with_name(iq_path.stem.replace("_iq", "") + "_swe.hdf5")


def load_swe_displacement(swe_path):
    """Load ``(displacement (T,z,x), coords (z,x,3), t_s (T,))`` from a ``*_swe.hdf5``."""
    with File(str(swe_path)) as f:
        td = f.data.tissue_doppler
        coords = np.asarray(td.coordinates[:], dtype=np.float32)
        t_s = np.asarray(td.timestamps[:], dtype=np.float32)
        disp = np.asarray({c.name: c for c in f.custom}["displacement"].data)
    return disp, coords, t_s


def mline_store_path(iq_path) -> Path:
    """Where the M-line control points are persisted (``<stem>_mline.npz``)."""
    iq_path = Path(iq_path)
    return iq_path.with_name(iq_path.stem.replace("_iq", "") + "_mline.npz")


def save_mline(path, mline: MLine):
    """Persist an M-line's control points + sampling density so it can be reloaded."""
    np.savez(str(path), points=np.asarray(mline.points, dtype=float),
             n_samples=int(mline.s.size))


def load_mline(path, n_samples=None) -> MLine:
    """Rebuild an :class:`MLine` from a saved ``.npz`` (see :func:`save_mline`)."""
    d = np.load(str(path))
    ns = int(d["n_samples"]) if "n_samples" in d else (n_samples or 250)
    return fit_spline(d["points"], n_samples=ns)


def get_mline(iq_path, config: MLineConfig | None = None, force_reselect: bool = False) -> MLine:
    """Get the M-line: reuse the saved one if present, else select it interactively.

    A previously selected line is stored next to the IQ as ``<stem>_mline.npz`` (plus a
    ``<stem>_mline.png`` record). If that exists and ``force_reselect`` is False (and no
    explicit ``config.points`` is given), it is reloaded - no GUI. Otherwise the B-mode
    frame is shown for picking (or ``config.points`` is used), and the line is saved.
    """
    config = config or MLineConfig()
    iq_path = Path(iq_path)
    stem = iq_path.stem.replace("_iq", "")
    store = mline_store_path(iq_path)

    if not force_reselect and config.points is None and store.is_file():
        mline = load_mline(store, config.n_samples)
        print(f"  [M-line] reuse {store.name} ({mline.length * 1e3:.1f} mm, "
              f"{mline.points.shape[0]} pts) - force_reselect to redraw")
        return mline

    bmode_path = Path(config.bmode_source) if config.bmode_source else iq_path
    if config.display_frame is not None:
        frame = config.display_frame
    elif config.display_time is not None:
        with File(str(iq_path)) as f:
            ts = np.asarray(f.data.beamformed_data.timestamps[:])
        frame = int(np.searchsorted(ts, config.display_time))
    else:
        with File(str(bmode_path)) as f:
            frame = f.data.beamformed_data.values.shape[0] // 2
    bmode_u8, bmode_coords, n = load_bmode_frame(bmode_path, frame)
    frame = int(np.clip(frame, 0, n - 1))

    if config.points is not None:
        mline = fit_spline(np.asarray(config.points), n_samples=config.n_samples)
    else:
        mline = select_mline(bmode_u8, bmode_coords, n_samples=config.n_samples,
                             title=f"M-line on {bmode_path.name} frame {frame}")
    save_mline(store, mline)
    draw_mline_on_bmode(bmode_u8, bmode_coords, mline, iq_path.with_name(stem + "_mline.png"),
                        title=f"M-line ({mline.length * 1e3:.1f} mm) @ frame {frame}")
    print(f"  [M-line] {mline.length * 1e3:.1f} mm, {mline.points.shape[0]} pts "
          f"-> {stem}_mline.png (+ .npz)")
    return mline


def compute_spacetime(iq_path, mline: MLine, config: MLineConfig | None = None):
    """Along-line displacement -> auto burst windows -> per-window IQ reprocessing -> plots.

    Given an already-chosen ``mline``: samples the full-measurement displacement along it
    (overview), auto-detects strong bursts and puts a ``window_ms`` window around each,
    re-filters the IQ inside each window (short-window clutter + band-pass, which differs
    from whole-record filtering), and writes a space-time plot per window. Returns
    ``(mline, results)`` with ``results`` a list of ``(BurstWindow, D_st, png)``.
    """
    from swi_dsp import load_passive_stream, PRESETS

    config = config or MLineConfig()
    iq_path = Path(iq_path)
    stem = iq_path.stem.replace("_iq", "")
    pad_s = config.analysis_pad_ms * 1e-3

    disp_full, coords, t_s = load_swe_displacement(_swe_path_for(iq_path))
    swe = load_passive_stream(iq_path)
    print(f"  [space-time] {stem}: {swe.n_frames} IQ frames @ {swe.fps:.0f} Hz, "
          f"full displacement over {t_s[0]:.3f}-{t_s[-1]:.3f}s")

    # Overview: full-duration displacement along the line (whole-record filtering),
    # median over the vertically-shifted M-line band.
    D_full = sample_along_line_median(disp_full, coords, mline,
                                      config.n_mlines, config.mline_spacing_px)
    overview_png = iq_path.with_name(stem + "_mline_spacetime_full.png")
    spacetime_plot(D_full, mline.s, t_s, overview_png, value_unit=config.value_unit,
                   title=f"{stem}  D(s,t) full measurement (whole-record filtering)")

    # Auto 100 ms windows around the strong along-line bursts (or a manual override).
    if config.time_window is not None:
        windows = [BurstWindow(t_peak=float(np.mean(config.time_window)),
                               t0=config.time_window[0], t1=config.time_window[1], score=float("nan"))]
        energy_raw, _ = energy_along_line(D_full)
    else:
        windows, energy_raw = detect_line_bursts(
            D_full, t_s, window_ms=config.window_ms, max_events=config.max_events)
    plot_bursts(energy_raw, t_s, windows, iq_path.with_name(stem + "_bursts.png"),
                title=f"{stem} — along-M-line energy, {int(config.window_ms)} ms burst windows")
    print(f"    {len(windows)} burst window(s):")
    for i, w in enumerate(windows):
        print(f"      #{i}: peak {w.t_peak*1e3:.0f} ms, window [{w.t0*1e3:.0f}, {w.t1*1e3:.0f}] ms")
    if config.window_index is not None:
        if not 0 <= config.window_index < len(windows):
            raise IndexError(f"window_index {config.window_index} out of range (0..{len(windows)-1}).")
        windows = [windows[config.window_index]]

    # Re-process the IQ within each window and make its space-time plot. By default this
    # re-beamforms a FINE axial ROI around the M-line (swi_finegrid.mline_spacetime_fine) -
    # the same "fine grid reform around the selected M-line" the speed stage uses; set
    # config.fine_grid=False to re-filter the coarse Stage-A IQ instead.
    results = []
    for i, w in enumerate(windows):
        png = iq_path.with_name(stem + f"_mline_win{i}_{w.t0*1e3:.0f}-{w.t1*1e3:.0f}ms.png")
        if config.fine_grid:
            from swi_finegrid import mline_spacetime_fine
            grid_band = PRESETS[config.preset].bulk_band
            fine_pad = pad_s if pad_s > 0 else 0.030    # need roll-off room for the short-window filter
            D_win, t_win, _, _ = mline_spacetime_fine(
                str(iq_path), window=w.window, estimator=config.estimator, pad_s=fine_pad,
                dz_um=config.fine_dz_um, dx_um=config.fine_dx_um, bulk_band=grid_band,
                n_mlines=config.n_mlines, line_spacing_m=config.mline_spacing_px * 1e-4,
                save=False)
            crop = (w.t0, w.t1)
            spacetime_plot(D_win, mline.s, t_win, png, value_unit=config.value_unit,
                           time_window=crop,
                           title=f"{stem}  D(s,t) window #{i} [{w.t0*1e3:.0f}-{w.t1*1e3:.0f} ms] "
                                 f"(FINE grid {config.estimator})")
            print(f"      window #{i}: fine-grid {config.estimator} -> {png.name}")
        else:
            D_win, t_win, (f0, f1) = reprocess_window(
                swe, mline, w.window, preset=config.preset, estimator=config.estimator,
                estimator_kwargs=config.estimator_kwargs, analysis_pad_s=pad_s,
                n_mlines=config.n_mlines, mline_spacing_px=config.mline_spacing_px)
            crop = (w.t0, w.t1) if pad_s > 0 else None
            spacetime_plot(D_win, mline.s, t_win, png, value_unit=config.value_unit, time_window=crop,
                           title=f"{stem}  D(s,t) window #{i} [{w.t0*1e3:.0f}-{w.t1*1e3:.0f} ms] "
                                 f"(re-filtered on {f1 - f0} frames)")
            print(f"      window #{i}: reprocessed frames {f0}-{f1} "
                  f"({(f1 - f0) / swe.fps * 1e3:.0f} ms) -> {png.name}")
        results.append((w, D_win, png))
    return mline, results


def process_mline(iq_path, config: MLineConfig | None = None, force_reselect: bool = False):
    """M-line (select or reuse) + the space-time plots, in one call. See :func:`get_mline`
    and :func:`compute_spacetime` for the two halves (which the unified pipeline skips
    independently). ``force_reselect`` re-opens the picker even if a saved line exists.
    Returns ``(mline, results)``."""
    config = config or MLineConfig()
    mline = get_mline(iq_path, config, force_reselect=force_reselect)
    return compute_spacetime(iq_path, mline, config)


if __name__ == "__main__":
    # swi_mline is a library. The runnable driver (with the IQ path + config) is
    # run_mline.py - running THIS file alone would otherwise do nothing.
    print(__doc__)
    print("\nThis module is a library; run the driver instead:\n"
          "    python run_mline.py\n"
          "(edit IQ_FILE / MLineConfig there). Or from a script:\n"
          "    from swi_mline import process_mline, MLineConfig\n"
          "    process_mline(r'...\\CombinedData_buffer4_iq.hdf5', MLineConfig())")
