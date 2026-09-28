"""Calibrate the combined buffer-3 head estimator on the corrected trigger-count reference (2026-09-28).

Anatomy features (A) only. Score = z(buffer-1, per-frame normalised, circular phase)
+ z(expected-motion) + w * z(continuity), w fixed or scaled by the wrap's phase jump. Reports exact
accuracy and the margin threshold keeping >= 99 % exact, and the same for continuity alone (the only
method without ECG). Reads the unwrap_cache; boundary corrected as in unwrap_explore_eval.

    python study/analysis/unwrap_combined_calibration.py -> study/logs/unwrap_combined_calibration_20260928.log
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unwrap_explore_eval import (CACHE, SHIFT, best_margin, cdist, feats, phase, rowz, s_b1, s_cont,  # noqa: E402
                                 s_ss, zs)


def thr_at(g, col, err, target):
    g = g.sort_values(col, ascending=False)
    ok = (g[err] == 0).to_numpy()
    prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    k = np.flatnonzero(prec >= target)
    return (g[col].iloc[k.max()], k.max() + 1) if k.size else (np.inf, 0)


rows = []
for p in sorted(glob.glob(str(CACHE / "*.npz"))):
    d = np.load(p)
    if not (bool(d["n1_ok"]) and bool(d["n3_ok"])):
        continue
    n = len(d["A3"])
    fm3, fm1 = float(d["frame_ms3"]), float(d["frame_ms1"])
    r = np.asarray(d["r"], float)
    rr = float(np.median(np.diff(r))) if r.size > 1 else np.nan
    t3c = d["t3"] + fm3 / 2 + SHIFT * fm3
    truth = int(d["tc_first"])
    if abs(float(d["t1"][0] - d["t3"][-1])) < fm3 / 2:
        t3c = t3c - fm3
        truth = (truth - 1) % n if truth >= 0 else truth
    ph1, ph3 = phase(d["t1"] + fm1 / 2, r), phase(t3c, r)
    X1, X3 = feats(d, "A", False)
    M = X3 @ X1.T
    wrap = (n * fm3) % rr if np.isfinite(rr) else np.nan
    wj = min(wrap, rr - wrap) if np.isfinite(rr) else np.nan
    ct = -s_cont(X3, n)
    row = dict(folder=Path(p).stem, n=n, truth=truth, ecg=bool(d["ecg_ok"]), wrap_jump=wj)
    h, m = best_margin(ct)
    row.update(cont=h, cont_m=m, cont_e=cdist(h, truth, n) if truth >= 0 else np.nan)
    if bool(d["ecg_ok"]):
        b1 = s_b1(rowz(M), ph1, ph3, n, circ_rr=rr)
        ss = s_ss(X1, X3, ph1, ph3, n, fm3)
        base = zs(b1) + (zs(ss) if ss is not None else 0)
        for tag, w in (("w05", 0.5), ("w10", 1.0), ("wj", float(np.clip(wj / 300, 0, 1.5)) if np.isfinite(wj) else 0.5)):
            h, m = best_margin(base + w * zs(ct))
            row.update({tag: h, tag + "_m": m, tag + "_e": cdist(h, truth, n) if truth >= 0 else np.nan})
    rows.append(row)
df = pd.DataFrame(rows)
gt = df[(df.truth >= 0) & df.ecg]
rest = df[(df.truth < 0) & df.ecg]
print(f"reference with ECG {len(gt)}; without trigger count, ECG ok {len(rest)}, no ECG {int(((df.truth < 0) & ~df.ecg).sum())}")
for tag in ("w05", "w10", "wj"):
    t99, k99 = thr_at(gt, tag + "_m", tag + "_e", 0.99)
    print(f"{tag}: exact {(gt[tag + '_e'] == 0).mean():.3f}; 99 % point margin {t99:.3f} accepts {k99}/{len(gt)} "
          f"-> resolves {(rest[tag + '_m'] >= t99).sum()}/{len(rest)} without trigger count")
g_all = df[df.truth >= 0]
for target in (0.98, 0.99):
    t, k = thr_at(g_all, "cont_m", "cont_e", target)
    noecg = df[(df.truth < 0) & ~df.ecg]
    print(f"continuity alone (all reference {len(g_all)}): {target:.0%} point margin {t:.3f} accepts {k} "
          f"-> resolves {(noecg.cont_m >= t).sum()}/{len(noecg)} no-ECG folders")
df.to_csv(Path(__file__).resolve().parents[2] / "study" / "logs" / "unwrap_combined_calibration_20260928.csv", index=False)
