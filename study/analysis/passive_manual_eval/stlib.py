"""Space-time helpers: sampling along straight lines, line scores, best-line search.

Conventions (as swp.manual.slope_gui): a space-time ``d`` is (n_t, n_r), ``t`` in s on the buffer-4
clock, ``r`` in m along the M-line (r = 0 = first clicked point = basal end). A line is anchored at
(t_a, r_a) with speed c [m/s] = dr/dt: t(r) = t_a + (r - r_a) / c. Lines are sampled PER r sample
(uniform weight per mm of septum), linearly interpolated in t, so fast and near-vertical lines are
scored as well as slow ones (the older per-t sampling returned nothing for lines crossing in < 5
frames).
"""
from __future__ import annotations

import numpy as np

SPEEDS = np.concatenate([-np.geomspace(0.5, 20, 41)[::-1], np.geomspace(0.5, 20, 41), [np.inf]])


def sample_line(d, t, r, t_a, r_a, c):
    """values of d along the line (n_r,), NaN where the line leaves the time axis."""
    dt = t[1] - t[0]
    tt = np.full(r.shape, t_a) if not np.isfinite(c) else t_a + (r - r_a) / c
    fi = (tt - t[0]) / dt
    ok = (fi >= 0) & (fi <= len(t) - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, len(t) - 2)
    fr = fi - i0
    cols = np.arange(r.size)
    v = d[i0, cols] * (1 - fr) + d[i0 + 1, cols] * fr
    v[~ok] = np.nan
    return v


def panel_rms(d, t, t0, t1):
    m = (t >= t0) & (t <= t1)
    return float(np.sqrt(np.mean(d[m] ** 2))) + 1e-30


def line_score(d, t, r, t_a, r_a, c, rms=None, min_cover=0.5):
    """(track, coherence, cover): mean |d| along the line / panel RMS; |mean d| / mean |d|."""
    v = sample_line(d, t, r, t_a, r_a, c)
    ok = np.isfinite(v)
    cover = ok.mean()
    if cover < min_cover:
        return np.nan, np.nan, cover
    rms = np.sqrt(np.nanmean(d ** 2)) if rms is None else rms
    a = np.abs(v[ok])
    return float(a.mean() / rms), float(abs(v[ok].mean()) / (a.mean() + 1e-30)), float(cover)


def line_scores_all(d, t, r, c, tau_idx, r_ref):
    """Vectorised: for speed c, lines through (t[tau_idx], r_ref) -> (|mean| track (n_tau,), signed
    mean (n_tau,), cover (n_tau,)). Uses frames as the time unit."""
    dt = t[1] - t[0]
    sh = np.zeros(r.size) if not np.isfinite(c) else (r - r_ref) / c / dt
    fi = tau_idx[:, None] + sh[None, :]
    ok = (fi >= 0) & (fi <= len(t) - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, len(t) - 2)
    fr = fi - i0
    cols = np.arange(r.size)[None, :]
    v = d[i0, cols] * (1 - fr) + d[i0 + 1, cols] * fr
    v = np.where(ok, v, np.nan)
    cover = ok.mean(axis=1)
    with np.errstate(invalid="ignore"):
        absmean = np.nanmean(np.abs(v), axis=1)
        smean = np.nanmean(v, axis=1)
    return absmean, smean, cover


def best_line(d, t, r, t0, t1, speeds=SPEEDS, min_cover=0.6, objective="abs", r_ref=None):
    """Best straight line with its reference time inside [t0, t1].

    objective 'abs'    : mean |d| along the line (a ridge of either sign, or a mix);
              'signed' : |mean d| along the line (one polarity band = one wavefront).
    Returns dict(c, t_a, r_a, score, coherence) with score normalised by the panel RMS over [t0, t1].
    """
    rms = panel_rms(d, t, t0, t1)
    r_ref = float(r.mean()) if r_ref is None else r_ref
    tau = np.where((t >= t0) & (t <= t1))[0].astype(float)
    best = dict(score=-1.0)
    for c in speeds:
        am, sm, cov = line_scores_all(d, t, r, c, tau, r_ref)
        s = am if objective == "abs" else np.abs(sm)
        s = np.where(cov >= min_cover, s, -1.0)
        k = int(np.nanargmax(s))
        if s[k] > best["score"]:
            best = dict(c=float(c), t_a=float(t[int(tau[k])]), r_a=r_ref, score=float(s[k]),
                        coherence=float(abs(sm[k]) / (am[k] + 1e-30)))
    best["score"] = best["score"] / rms
    return best


def flat_fraction(d, t, t0, t1):
    """Energy share of the per-time spatial mean inside [t0, t1] (in-phase / bulk motion)."""
    m = (t >= t0) & (t <= t1)
    w = d[m]
    mean_r = w.mean(axis=1, keepdims=True)
    return float(np.sum(np.broadcast_to(mean_r, w.shape) ** 2) / (np.sum(w ** 2) + 1e-30))


def burst_ratio(d, t, t0, t1):
    m = (t >= t0) & (t <= t1)
    if (~m).sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean(d[m] ** 2)) / (np.sqrt(np.mean(d[~m] ** 2)) + 1e-30))


def lag_profile(d, t, r, t0, t1, ref_mm=5.0, max_lag_ms=30.0):
    """Arrival delay along r: lag of max cross-correlation of every column with the mean of the
    first ``ref_mm`` (basal) -> (lag_s (n_r,), corr (n_r,)). Parabolic sub-sample peak."""
    m = (t >= t0) & (t <= t1)
    w = d[m] - d[m].mean(axis=0, keepdims=True)
    dt = t[1] - t[0]
    ref = w[:, r <= r[0] + ref_mm * 1e-3].mean(axis=1)
    L = int(round(max_lag_ms * 1e-3 / dt))
    n = w.shape[0]
    lags = np.arange(-L, L + 1)
    lag = np.full(r.size, np.nan)
    cc = np.full(r.size, np.nan)
    rn = np.sqrt(np.sum(ref ** 2)) + 1e-30
    for j in range(r.size):
        x = w[:, j]
        xn = np.sqrt(np.sum(x ** 2)) + 1e-30
        c = np.array([np.sum(ref[max(0, -k):n - max(0, k)] * x[max(0, k):n - max(0, -k)]) for k in lags]) / (rn * xn)
        k = int(np.argmax(c))
        sub = 0.0
        if 0 < k < len(c) - 1:
            den = c[k - 1] - 2 * c[k] + c[k + 1]
            sub = 0.5 * (c[k - 1] - c[k + 1]) / den if den != 0 else 0.0
        lag[j] = (lags[k] + sub) * dt
        cc[j] = c[k]
    return lag, cc
