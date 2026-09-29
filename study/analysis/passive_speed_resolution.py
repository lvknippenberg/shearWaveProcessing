"""How precisely does a hand-drawn slope fix the speed? (manual passive study)

For every usable window (confidence >= 2) the hand line's anchor is kept and the line is tilted
over a slowness grid p = 1/c. The along-line score (mean |signal| on the line / panel RMS, as in
passive_manual_prelim.py) traces a peak around the best tilt; its width at 90 % of the peak is the
range of lines that fit the panel about equally well. Reported both in slowness (s/m) and as the
implied speed interval.

The expectation: the width is set in TIME - the wavefront's temporal thickness (~ a few ms at
15-150 Hz) against the delay it builds up over the line, L / c. It is therefore about constant in
slowness, and the speed interval grows as c^2: dc = c^2 dp.

Output: study/logs/passive_manual_prelim/speed_resolution.csv (+ printed summary).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.manual import store as S   # noqa: E402

D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
VIEW = "velocity gauss"
LEVEL = 0.9


def _score(d, r, t, t_a, r_a, p):
    """p = signed slowness (s/m); line r = r_a + (t - t_a) / p."""
    rr = r_a + (t - t_a) / p
    ok = (rr >= r[0]) & (rr <= r[-1])
    if ok.sum() < 5:
        return np.nan
    ri = np.clip(np.round(np.interp(rr[ok], r, np.arange(r.size))).astype(int), 0, r.size - 1)
    return float(np.mean(np.abs(d[ri, np.arange(t.size)[ok]])) / (np.sqrt(np.mean(d ** 2)) + 1e-30))


def profile(npz, slope):
    z = np.load(npz, allow_pickle=True)
    j = [k for k in range(int(z["n_views"])) if str(z[f"v{k}_name"]) == VIEW][0]
    r, t = np.asarray(z[f"v{j}_r"], float), np.asarray(z[f"v{j}_t"], float)
    d = np.asarray(z[f"v{j}_data"], float).T                  # (n_r, n_t)
    sh = slope["shared"]
    c = float(sh["speed_m_s"])
    t_a, r_a = sh["anchor_t_ms"] * 1e-3, sh["anchor_r_mm"] * 1e-3
    p0 = 1.0 / c
    ps = p0 + np.linspace(-0.6, 0.6, 481)                     # s/m, around the hand slowness
    ps = ps[np.abs(ps) > 1 / 50.0]                             # keep |c| < 50 m/s
    sc = np.array([_score(d, r, t, t_a, r_a, p) for p in ps])
    return ps, sc, p0, r[-1]


def main():
    w = pd.read_csv(os.path.join(D, "windows_scored.csv"))
    rows = []
    for _, x in w[(w.confidence >= 2)].iterrows():
        p = S.Paths(x.folder)
        sl = (S.read_json(p.slopes_json) or {}).get(str(int(x.window)))
        if not sl or not (sl.get("shared") or {}).get("speed_m_s"):
            continue
        ps, sc, p0, L = profile(p.st_npz(int(x.window)), sl)
        ok = np.isfinite(sc)
        ps, sc = ps[ok], sc[ok]
        k = int(np.argmin(np.abs(ps - p0)))
        kmax = k + int(np.argmax(sc[max(k - 40, 0):k + 41])) - min(k, 40)   # local peak near the hand line
        thr = LEVEL * sc[kmax]
        lo = kmax
        while lo > 0 and sc[lo - 1] >= thr:
            lo -= 1
        hi = kmax
        while hi < sc.size - 1 and sc[hi + 1] >= thr:
            hi += 1
        p_lo, p_hi = ps[lo], ps[hi]
        c_hand = 1 / p0
        c_fast = 1 / p_lo if p_lo > 0 else np.inf            # smaller slowness = faster
        c_slow = 1 / p_hi
        rows.append(dict(subject=x.subject, window=int(x.window), label=x.label, confidence=x.confidence,
                         c_hand=c_hand, c_peak=1 / ps[kmax], dp_s_per_m=p_hi - p_lo,
                         c_slow=c_slow, c_fast=c_fast, L_mm=L * 1e3,
                         delay_ms=L / c_hand * 1e3, peak_score=sc[kmax]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(D, "speed_resolution.csv"), index=False)
    out["rel_width"] = (np.minimum(out.c_fast, 50) - out.c_slow) / out.c_hand
    bins = pd.cut(out.c_hand, [0, 2, 3, 4, 6, 50])
    print(f"{len(out)} usable windows, view '{VIEW}', band = lines scoring >= {LEVEL:.0%} of the best")
    print(out.groupby(bins).agg(n=("c_hand", "size"), slowness_width=("dp_s_per_m", "median"),
                                c_slow=("c_slow", "median"), c_hand=("c_hand", "median"),
                                c_fast=("c_fast", "median"), rel_width=("rel_width", "median"),
                                delay_ms=("delay_ms", "median")).round(3).to_string())
    print("\nslowness width vs c: Spearman rho %.2f" % out[["dp_s_per_m", "c_hand"]].corr("spearman").iloc[0, 1])
    print("windows whose fast bound is open (any faster line fits as well): %d of %d"
          % ((out.c_fast >= 50).sum(), len(out)))


if __name__ == "__main__":
    main()
