"""Fit a passive wavefront by following its ridge, instead of clicking two points on it.

Why this exists
---------------
Two estimators were tried before this one and both fail on these panels:

* the **slant stack** (``swp.viz.metrics.slant_stack_speed``) maximises a *global* semblance,
  which a band of bulk motion satisfies as well as the wave does - so the fitted line often
  does not sit on the wavefront at all (docs/passive_speed_estimation.md);
* a **two-click manual pick** (``manual_slope.py``) does sit on the wave, but the speed is the
  difference of two hand-placed points, so a 1 ms slip over a 5 ms moveout is a 25 % error.

What is measured here is not a slope hypothesis but the wavefront itself: the local extremum is
*followed* along the M-line from the strongest point of the panel, one along-line sample at a
time, and a straight line is then fitted to that track. The step in time per sample is capped at
``dr/cmin`` so the track cannot hop onto a neighbouring band.

Why not "maximise the amplitude sampled along the line", as docs suggested? Because on these
panels that objective is **flat**: over a 20 mm M-line a genuine wave moves by ~5 ms while its
period is ~60 ms, so the mean amplitude along the line barely changes between 2 m/s and 15 m/s
(measured on C000000023: the 95 %-of-maximum band covers the entire 1-20 m/s search range). The
amplitude along the line tells you *whether the line is on the wave*; it cannot tell you *what
slope the wave has*. Only the moveout of the ridge can.

Both polarities (crest and trough) are tracked. The one carrying more amplitude is reported; the
other is an independent check, and the two disagreeing is a sign the panel has no single wave.

**Caveat, measured 2026-09-21: this estimator reads systematically HIGH on a dispersive packet.**
It follows the instantaneous maximum, and where the wave is dispersive that peak drifts earlier
with distance relative to the envelope, which inflates the apparent speed. On the AVC window of
C000000023 it gives 4.64 m/s against 3.56 (hand-drawn), 3.93 (cross-correlation group delay),
3.96 (slant stack) and 3.12 (phase speed at the dominant 14 Hz bin) - it is the outlier, not the
reference. Phase speed on that window runs from 1.4 m/s at 7 Hz to 9.3 m/s at 28 Hz, so "the"
speed is band-dependent; the agreed measurand is group delay over the band.

Use it for triage, for seeing *where* the wavefront is, and for the ridge track itself - which is
genuinely informative, e.g. it shows the MVC velocity panel propagating only over the first 14 mm
and going stationary beyond. Do not report its speed as the measurement.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RidgeFit", "fit_ridge_speed"]


class RidgeFit(dict):
    """Result of one ridge fit; a dict so it serialises straight to JSON."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:                                        # noqa: BLE001
            raise AttributeError(k) from e


def _subsample_peak(col, k):
    """Parabolic refinement of the index of a maximum, so the track is not quantised to frames."""
    if 0 < k < col.size - 1:
        y0, y1, y2 = col[k - 1], col[k], col[k + 1]
        den = y0 - 2 * y1 + y2
        return k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    return float(k)


def _track(sig, t, r, seed_col, seed_row, cmin):
    """Follow the local maximum of ``sig`` outwards from a seed; -> ridge time per position [s]."""
    nt, nr = sig.shape
    dt, dr = np.median(np.diff(t)), np.median(np.diff(r))
    max_step = max(1, int(np.ceil((dr / cmin) / dt)))       # cap: never faster than cmin
    idx = np.full(nr, np.nan)
    idx[seed_col] = _subsample_peak(sig[:, seed_col], seed_row)
    for direction in (+1, -1):
        k, j = seed_row, seed_col
        while True:
            j += direction
            if not 0 <= j < nr:
                break
            lo, hi = max(0, k - max_step), min(nt, k + max_step + 1)
            k = lo + int(np.argmax(sig[lo:hi, j]))
            idx[j] = _subsample_peak(sig[:, j], k)
    return np.interp(idx, np.arange(nt), t)


def _theil_sen(r_mm, t_ms, rng, n_pairs=40000):
    """Median pairwise slope dr/dt [mm/ms = m/s]; robust to the ends of the track wandering."""
    n = r_mm.size
    i = rng.choice(n, size=(min(n_pairs, n * 8), 2))
    i = i[i[:, 0] != i[:, 1]]
    dr = r_mm[i[:, 1]] - r_mm[i[:, 0]]
    dt = t_ms[i[:, 1]] - t_ms[i[:, 0]]
    ok = np.abs(dt) > 1e-9
    return float(np.median(dr[ok] / dt[ok]))


def _fit_line(ridge_t, r, n_boot, seed):
    """Theil-Sen speed + a block bootstrap CI + the straightness of the track."""
    r_mm, t_ms = r * 1e3, ridge_t * 1e3
    rng = np.random.default_rng(seed)
    c = _theil_sen(r_mm, t_ms, rng)
    # The M-line is resampled to 250 points but carries only ~4 independent positions, so a
    # per-point bootstrap would be meaningless; resample contiguous blocks instead.
    block = max(5, r_mm.size // 20)
    cs = [_theil_sen(r_mm[i], t_ms[i], rng) for i in
          (np.concatenate([np.arange(s, s + block)
                           for s in rng.integers(0, r_mm.size - block, size=r_mm.size // block)])
           for _ in range(n_boot))]
    cs = np.asarray(cs)
    A = np.vstack([r_mm, np.ones_like(r_mm)]).T
    coef, *_ = np.linalg.lstsq(A, t_ms, rcond=None)
    resid = t_ms - A @ coef
    return (c, (float(np.percentile(cs, 5)), float(np.percentile(cs, 95))),
            float(1 - resid.var() / t_ms.var()), float(resid.std()))


def fit_ridge_speed(st, t_event_s=None, search_ms=40.0, cmin=0.4, n_boot=500, seed=0):
    """Fit the wavefront of a space-time panel by tracking its ridge.

    Parameters
    ----------
    st : SpaceTime
        ``res.st`` from ``swp.viz.pipeline.run_pipeline`` (``data`` (n_t, n_r), ``r`` [m], ``t`` [s]).
    t_event_s : float, optional
        Detected event time; the seed is taken within ``+/- search_ms`` of it, so the fit follows
        the wave belonging to *this* cardiac event rather than the panel's global maximum.
    cmin : float
        Slowest speed the track may follow [m/s]; also the cap that stops it hopping bands.

    Returns
    -------
    RidgeFit with ``speed`` [m/s, signed: + travels toward increasing r], ``ci`` (5-95 %),
    ``r2`` and ``resid_ms`` (straightness of the track), ``tracks`` (mean |signal| on the ridge /
    panel RMS), ``polarity``, ``ridge_t`` and the ``other`` polarity's fit for cross-checking.
    """
    d, r, t = np.asarray(st.data), np.asarray(st.r), np.asarray(st.t)
    rms = float(np.sqrt((d ** 2).mean()))
    near = (np.abs(t - t_event_s) <= search_ms * 1e-3 if t_event_s is not None
            else np.ones(t.size, bool))
    if not near.any():
        near = np.ones(t.size, bool)

    out = []
    for polarity, sign in (("crest", +1.0), ("trough", -1.0)):
        sig = sign * d
        row, col = np.unravel_index(
            int(np.argmax(np.where(near[:, None], sig, -np.inf))), sig.shape)
        ridge_t = _track(sig, t, r, col, row, cmin)
        c, ci, r2, resid = _fit_line(ridge_t, r, n_boot, seed)
        amp = float(np.mean([sig[int(np.argmin(np.abs(t - ridge_t[j]))), j]
                             for j in range(r.size)]))
        out.append(RidgeFit(polarity=polarity, speed=c, ci=list(ci), r2=r2, resid_ms=resid,
                            tracks=amp / (rms + 1e-20), ridge_t=ridge_t,
                            seed_t_s=float(t[row])))
    best, other = sorted(out, key=lambda f: -f["tracks"])
    best = RidgeFit(best)
    best["other"] = RidgeFit({k: v for k, v in other.items() if k != "ridge_t"})
    # A single straight wavefront shows the same moveout in both polarities half a period apart.
    best["polarity_spread"] = abs(best["speed"] - other["speed"]) / max(abs(best["speed"]), 1e-9)
    return best


def line_at(fit, r, quantile=0.5):
    """The fitted straight line as t(r) [s], anchored on the median offset of its own ridge."""
    t0 = np.quantile(fit["ridge_t"] - r / fit["speed"], quantile)
    return t0 + r / fit["speed"]
