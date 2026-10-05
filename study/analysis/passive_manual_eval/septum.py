"""Septal wall thickness across a drawn M-line, from one B-mode frame.

Geometry: the M-line (buffer-4 coordinates, mm) gives the wall direction. At positions s along the
middle part of the line, an intensity profile (dB) is sampled along the local NORMAL, offset
-20..+20 mm (+ = deeper, i.e. towards the LV in PLAX). Profiles are averaged over +-2 mm along the
line (speckle), then the wall is found in each averaged profile:

* centre   the maximum of the 2 mm-smoothed profile within +-6 mm of the line (the line is drawn on
           the septum, but not necessarily on its middle);
* floors   the minimum of the smoothed profile within 4-16 mm on either side of the centre
           (RV cavity above, LV cavity below);
* edges    'half'  : where the 0.6 mm-smoothed profile falls below floor + 0.5 (peak - floor),
                     walking outward from the centre on each side;
           'grad'  : the steepest rise (RV side) and steepest fall (LV side) of the 1 mm-smoothed
                     profile between the floor and the centre.
* valid    contrast (peak - floor) >= MIN_CONTRAST_DB on BOTH sides.

The image thickness is the median over valid positions. Without a ground truth, plausibility is
checked by: agreement between buffers 1 and 3 and between beats, and systolic thickening.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

MIN_CONTRAST_DB = 6.0
OUTIN_DB = 6.0
OFF = np.arange(-20, 20.001, 0.2)            # mm, + = deeper
DOFF = 0.2


def line_frame(pts):
    pts = np.asarray(pts, float)
    p0, p1 = pts[0], pts[-1]
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    return p0, p1, u, n


def resample_line(pts, step=1.0):
    """Points every ``step`` mm along the polyline + unit tangents."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    sl = np.hypot(*seg.T)
    cum = np.concatenate([[0], np.cumsum(sl)])
    s = np.arange(0, cum[-1], step)
    x = np.interp(s, cum, pts[:, 0])
    z = np.interp(s, cum, pts[:, 1])
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    tang = seg[k] / sl[k, None]
    return s, np.stack([x, z], 1), tang, cum[-1]


def profiles(db, x, z, pts, frac=(0.15, 0.85), along_avg_mm=2.0):
    """(s_mm (n,), profiles (n, len(OFF))) of dB along local normals (+ = deeper)."""
    s, P, T, L = resample_line(pts, 0.5)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    N[N[:, 1] < 0] *= -1
    keep = (s >= frac[0] * L) & (s <= frac[1] * L)
    s, P, N = s[keep], P[keep], N[keep]
    Q = P[:, None, :] + OFF[None, :, None] * N[:, None, :]
    xi = (Q[..., 0] - x[0]) / (x[1] - x[0])
    zi = (Q[..., 1] - z[0]) / (z[1] - z[0])
    prof = map_coordinates(db, [zi.ravel(), xi.ravel()], order=1, cval=np.nan).reshape(Q.shape[:2])
    k = max(1, int(round(along_avg_mm / 0.5)))
    if k > 1 and len(prof) >= k:
        c = np.cumsum(np.vstack([np.zeros(prof.shape[1]), prof]), axis=0)
        prof = (c[k:] - c[:-k]) / k
        s = s[k // 2: k // 2 + len(prof)]
    return s, prof


def wall(profile):
    """Edges of the wall in one profile -> dict or None."""
    p = np.asarray(profile, float)
    if not np.isfinite(p).all():
        return None
    p06 = gaussian_filter1d(p, 0.6 / DOFF)
    p1 = gaussian_filter1d(p, 1.0 / DOFF)
    p2 = gaussian_filter1d(p, 2.0 / DOFF)
    win = np.abs(OFF) <= 6
    ic = np.where(win)[0][np.argmax(p2[win])]
    c = OFF[ic]
    peak = p06[max(ic - 5, 0): ic + 6].max()
    out = dict(centre=c)
    for side, sgn in (("rv", -1), ("lv", +1)):
        d = sgn * (OFF - c)
        m = (d >= 4) & (d <= 16)
        if not m.any():
            return None
        j_floor = np.where(m)[0][np.argmin(p2[m])]
        floor = p2[j_floor]
        out[f"floor_{side}"] = floor
        out[f"contrast_{side}"] = peak - floor
        level = floor + 0.5 * (peak - floor)
        # walk outward from the centre until below the level
        idx = range(ic, j_floor + sgn, sgn)
        e = None
        prev = ic
        for j in idx:
            if p06[j] < level:
                # linear interpolation between prev and j
                a, b = p06[prev], p06[j]
                f = (a - level) / (a - b) if a != b else 0
                e = OFF[prev] + f * (OFF[j] - OFF[prev])
                break
            prev = j
        out[f"half_{side}"] = e
        # 'outin': walk from the cavity floor TOWARDS the centre; the edge is the first crossing above
        # floor + max(OUTIN_DB, 0.3 (peak - floor)). Unlike 'half' it does not stop at a dark gap
        # inside the wall (two bright endocardial layers with darker myocardium between).
        lev2 = floor + max(OUTIN_DB, 0.3 * (peak - floor))
        e2, prev = None, j_floor
        for j in range(j_floor, ic - sgn, -sgn):
            if p06[j] >= lev2:
                a, b = p06[prev], p06[j]
                f = (lev2 - a) / (b - a) if a != b else 0
                e2 = OFF[prev] + f * (OFF[j] - OFF[prev])
                break
            prev = j
        out[f"outin_{side}"] = e2
        g = np.gradient(p1, DOFF)
        lo, hi = sorted((ic, j_floor))
        seg = slice(lo, hi + 1)
        jj = np.arange(lo, hi + 1)
        k = jj[np.argmax(g[seg])] if side == "rv" else jj[np.argmin(g[seg])]
        out[f"grad_{side}"] = OFF[k]
    out["valid"] = (out["contrast_rv"] >= MIN_CONTRAST_DB and out["contrast_lv"] >= MIN_CONTRAST_DB
                    and out["half_rv"] is not None and out["half_lv"] is not None)
    out["t_half"] = (out["half_lv"] - out["half_rv"]) if out["half_rv"] is not None and out["half_lv"] is not None else np.nan
    out["t_grad"] = out["grad_lv"] - out["grad_rv"]
    out["t_outin"] = (out["outin_lv"] - out["outin_rv"]) if out["outin_rv"] is not None and out["outin_lv"] is not None else np.nan
    return out


def thickness(env, x, z, pts):
    """Image-level thickness from a linear envelope image (z, x) and the line (mm)."""
    db = 20 * np.log10(env / (np.percentile(env, 99.5) + 1e-30) + 1e-6)
    s, prof = profiles(db, x, z, pts)
    rows = [wall(p) for p in prof]
    ok = [r for r in rows if r is not None]
    val = [r for r in ok if r["valid"]]
    res = dict(n_pos=len(rows), frac_valid=len(val) / max(len(rows), 1),
               contrast_rv=float(np.median([r["contrast_rv"] for r in ok])) if ok else np.nan,
               contrast_lv=float(np.median([r["contrast_lv"] for r in ok])) if ok else np.nan,
               centre_offset=float(np.median([r["centre"] for r in ok])) if ok else np.nan)
    if val:
        th = np.array([r["t_half"] for r in val])
        tg = np.array([r["t_grad"] for r in val])
        to = np.array([r["t_outin"] for r in val], float)
        res.update(t_outin=float(np.nanmedian(to)) if np.isfinite(to).any() else np.nan,
                   rv_edge_outin=float(np.nanmedian([r["outin_rv"] if r["outin_rv"] is not None else np.nan for r in val])),
                   lv_edge_outin=float(np.nanmedian([r["outin_lv"] if r["outin_lv"] is not None else np.nan for r in val])))
        mid = np.array([0.5 * (r["half_rv"] + r["half_lv"]) for r in val])
        res.update(t_half=float(np.median(th)), t_half_iqr=float(np.subtract(*np.percentile(th, [75, 25]))),
                   t_grad=float(np.median(tg)), t_grad_iqr=float(np.subtract(*np.percentile(tg, [75, 25]))),
                   wall_mid_offset=float(np.median(mid)),
                   rv_edge=float(np.median([r["half_rv"] for r in val])),
                   lv_edge=float(np.median([r["half_lv"] for r in val])))
    # the mean profile (aligned on the line) for figures
    res["mean_profile"] = np.nanmedian(prof, axis=0) if len(prof) else None
    return res
