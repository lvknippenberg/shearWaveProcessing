"""Evaluate buffer-3 head estimators against the exact trigger count (2026-09-28). Read-only.

Reads the feature cache of unwrap_explore_cache.py. Ground truth = trigger-count head (folders
whose live-run start survived in the log). Stored slot q holds chronological frame
k = (q - first) mod n; chronological frame k has a known trigger time (the last n triggers of the
live run) and so a known time since the preceding R-peak (phase). Methods (each -> head + margin):

  b1        CURRENT: per candidate head, mean corr of every buffer-3 frame with the buffer-1 frame
            at the same phase (+-10 ms); margin = gap to the best non-adjacent head
  anchorR   USER'S PROPOSAL: buffer-1 frame at the R-peak -> most similar stored buffer-3 slot q*;
            the chronological index k_R of the buffer-3 frame at the R-peak is known from the log
            -> head = q* - k_R. (Two R-peaks inside the buffer-3 span: the higher-corr one.)
  vote      the anchor repeated for EVERY buffer-1 frame; head = most votes
  ss        expected-motion match: buffer 3's own frame-to-frame similarity matrix vs the one
            buffer 1 predicts for the same phases (each buffer compared only with itself)
  cont      continuity: the weakest neighbour link is the wrap (swp.acquisition.unwrap)
Features: A = anatomy (band-pass 1-6 mm), L = low-pass log envelope; suffix 'd' = temporally
de-meaned per buffer (static structures removed, only motion left).

    python study/analysis/unwrap_explore_eval.py
-> study/logs/unwrap_explore_eval_20260928.csv (+ summary on stdout)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
CACHE = _REPO / "study" / "analysis" / "unwrap_cache"
SHIFT = -1
# 2026-09-28: triggerlog._runs lets the buffer-3 live-loop run absorb buffer 1's FIRST trigger when
# it falls 39.4 +- 3 ms after the last loop trigger (~30 % of folders; gap t1[0]-t3[-1] ~ 0 ms).
# That overcounts the run by one: reference head and buffer-3 times one frame off. Fixed here.
FIX_BOUNDARY = "--raw-boundary" not in sys.argv


def phase(t, r):
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        p = r[r <= ti]
        if p.size:
            out[i] = ti - p.max()
    return out


def norm_rows(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def feats(d, key, demean):
    X1, X3 = d[key + "1"].astype(np.float32), d[key + "3"].astype(np.float32)
    if demean:
        X1, X3 = X1 - X1.mean(0), X3 - X3.mean(0)
    return norm_rows(X1), norm_rows(X3)


def best_margin(v, higher=True):
    v = np.asarray(v, float)
    n = len(v)
    s = np.where(np.isfinite(v), -v if higher else v, np.inf)
    i = int(np.argmin(s))
    if not np.isfinite(s[i]):
        return -1, np.nan
    rest = [s[k] for k in range(n) if min((k - i) % n, (i - k) % n) >= 2]
    return i, float(min(rest) - s[i])


def cdist(a, b, n):
    return min((a - b) % n, (b - a) % n)


def m_b1(M, ph1, ph3, n):
    return best_margin(s_b1(M, ph1, ph3, n))


def s_b1(M, ph1, ph3, n, tol=10.0, circ_rr=None):
    """Score curve of the current buffer-1 method. circ_rr: match phases circularly (mod RR)."""
    sc = np.full(n, np.nan)
    ok1 = np.isfinite(ph1)
    for first in range(n):
        vals = []
        for q in range(n):
            ph = ph3[(q - first) % n]
            if not np.isfinite(ph):
                continue
            dd = np.abs(ph1 - ph)
            if circ_rr:
                dd = np.minimum(dd, circ_rr - dd)
            dd = np.where(ok1, dd, np.inf)
            j = int(np.argmin(dd))
            if dd[j] < tol:
                vals.append(M[q, j])
        if len(vals) >= n // 2:
            sc[first] = np.mean(vals)
    return sc


def _anchor_votes(M, js, ph1, ph3, n, half):
    votes = np.zeros(n)
    for j in js:
        ks = np.flatnonzero(np.abs(ph3 - ph1[j]) < half)
        if not ks.size:
            continue
        qs = int(np.argmax(M[:, j]))
        for k in ks:
            votes[(qs - k) % n] += 1.0 / ks.size
    return votes


def m_anchorR(M, ph1, ph3, n, fm3):
    ok = np.flatnonzero(np.isfinite(ph1))
    if not ok.size:
        return -1, np.nan
    jR = ok[np.argmin(ph1[ok])]                       # buffer-1 frame closest after the R-peak
    ks = np.flatnonzero(np.abs(ph3 - ph1[jR]) < fm3 / 2)
    if not ks.size:
        return -1, np.nan
    col = M[:, jR]
    qs = int(np.argmax(col))
    srt = np.sort(col)
    k = ks[0] if ks.size == 1 else ks[0]              # both candidates are equally supported; take the first
    return int((qs - k) % n), float(srt[-1] - srt[-2])


def m_vote(M, ph1, ph3, n, fm3):
    js = np.flatnonzero(np.isfinite(ph1))
    v = _anchor_votes(M, js, ph1, ph3, n, fm3 / 2)
    if v.sum() == 0:
        return -1, np.nan
    i, m = best_margin(v)
    return i, m / v.sum()


def m_ss(X1, X3, ph1, ph3, n, fm3):
    sc = s_ss(X1, X3, ph1, ph3, n, fm3)
    return (-1, np.nan) if sc is None else best_margin(sc)


def s_ss(X1, X3, ph1, ph3, n, fm3):
    ok1 = np.flatnonzero(np.isfinite(ph1))
    jk = np.full(n, -1)
    for k in range(n):
        if np.isfinite(ph3[k]):
            dd = np.abs(ph1[ok1] - ph3[k])
            if dd.min() < fm3 / 2:
                jk[k] = ok1[np.argmin(dd)]
    valid = np.flatnonzero(jk >= 0)
    if valid.size < n // 2:
        return None
    S1 = X1[jk[valid]] @ X1[jk[valid]].T
    S3 = X3 @ X3.T
    iu = np.triu_indices(valid.size, 1)
    p = S1[iu]
    sc = np.full(n, np.nan)
    for first in range(n):
        q = (valid + first) % n
        o = S3[np.ix_(q, q)][iu]
        sc[first] = np.corrcoef(o, p)[0, 1]
    return sc


def m_adj(X1, X3, t1c, t3c, n, n_b1=3):
    """TIME ADJACENCY (no ECG): buffer 1 starts 0-60 ms after the live loop's last trigger, so the
    newest stored buffer-3 frame is the one most like the first buffer-1 frames. Score per head =
    corr of its newest frame with the mean of buffer-1 frames 0..n_b1-1."""
    ref = X1[:n_b1].mean(0)
    ref /= np.linalg.norm(ref) + 1e-12
    s = X3 @ ref                                          # per stored slot
    sc = np.array([s[(first - 1) % n] for first in range(n)])
    return best_margin(sc)


def m_adj2(X1, X3, t1c, t3c, n, n3=4, tol=None):
    """TIME ADJACENCY, trajectory: the newest n3 buffer-3 frames vs buffer-1 frames, weighted by
    how close they are in absolute time (same trigger-log clock; weight exp(-dt/100 ms))."""
    C = X3 @ X1.T
    sc = np.zeros(n)
    for first in range(n):
        tot, wsum = 0.0, 0.0
        for k in range(n - n3, n):
            q = (k + first) % n
            dt = t1c - t3c[k]                               # >0: buffer 1 later
            w = np.exp(-np.abs(dt) / 100.0)
            tot += float((w * C[q]).sum())
            wsum += float(w.sum())
        sc[first] = tot / wsum if wsum > 0 else np.nan
    return best_margin(sc)


def m_cont(X3, n):
    return best_margin(s_cont(X3, n), higher=False)


def s_cont(X3, n):
    M = X3 @ X3.T
    lag = {L: np.array([M[i, (i + L) % n] for i in range(n)]) for L in (1, 2)}
    med = {L: float(np.median(v)) for L, v in lag.items()}
    sc = np.empty(n)
    for first in range(n):
        b = (first - 1) % n
        sc[first] = 0.5 * (lag[1][b] / med[1] + np.mean([lag[2][(b - 1) % n], lag[2][b]]) / med[2])
    return sc


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def rowz(M):
    """Per buffer-3 frame: z-score its correlations across buffer-1 frames (removes the frame's
    own baseline - a sharp frame correlates higher with everything)."""
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-12)


def main():
    rows = []
    files = sorted(glob.glob(str(CACHE / "*.npz")))
    for p in files:
        d = np.load(p)
        if not (bool(d["n1_ok"]) and bool(d["n3_ok"])):
            continue
        name = Path(p).stem
        n = len(d["A3"])
        fm3, fm1 = float(d["frame_ms3"]), float(d["frame_ms1"])
        r = np.asarray(d["r"], float)
        rr = float(np.median(np.diff(r))) if r.size > 1 else np.nan
        t1c = d["t1"] + fm1 / 2
        t3c = d["t3"] + fm3 / 2 + SHIFT * fm3
        ph1 = phase(t1c, r)
        ph3 = phase(t3c, r)
        truth = int(d["tc_first"])
        gap = float(d["t1"][0] - d["t3"][-1])
        fixed = FIX_BOUNDARY and abs(gap) < fm3 / 2
        if fixed:                       # the run absorbed buffer 1's first trigger: drop it
            t3c = t3c - fm3
            ph3 = phase(t3c, r)
            truth = (truth - 1) % n if truth >= 0 else truth
        wrap_ms = (n * fm3) % rr if np.isfinite(rr) else np.nan   # phase jump across the wrap
        row = dict(folder=name, subject=name[:10], n=n, truth=truth, ecg_ok=bool(d["ecg_ok"]),
                   rr=round(rr, 1), wrap_jump_ms=round(min(wrap_ms, rr - wrap_ms), 1) if np.isfinite(rr) else np.nan,
                   gap_b3_b1_ms=round(gap, 1), boundary_fixed=fixed,
                   applied=int(d["unwrap_first"]), applied_method=str(d["unwrap_method"]))
        for key in ("A", "L"):
            for dm in (False, True):
                tag = key + ("d" if dm else "")
                X1, X3 = feats(d, key, dm)
                M = X3 @ X1.T
                res = {"b1": m_b1(M, ph1, ph3, n), "anchorR": m_anchorR(M, ph1, ph3, n, fm3),
                       "vote": m_vote(M, ph1, ph3, n, fm3), "ss": m_ss(X1, X3, ph1, ph3, n, fm3),
                       "cont": m_cont(X3, n),
                       "adj": m_adj(X1, X3, t1c, t3c, n), "adj2": m_adj2(X1, X3, t1c, t3c, n)}
                b1 = s_b1(M, ph1, ph3, n)
                b1z = s_b1(rowz(M), ph1, ph3, n)
                b1c = s_b1(rowz(M), ph1, ph3, n, circ_rr=rr)
                ss = s_ss(X1, X3, ph1, ph3, n, fm3)
                ct = -s_cont(X3, n)
                res["b1z"] = best_margin(b1z)
                res["b1zc"] = best_margin(b1c)
                if ss is not None:
                    res["b1+ss"] = best_margin(zs(b1) + zs(ss))
                    res["b1zc+ss"] = best_margin(zs(b1c) + zs(ss))
                    res["b1zc+ss+cont"] = best_margin(zs(b1c) + zs(ss) + 0.5 * zs(ct))
                res["b1zc+cont"] = best_margin(zs(b1c) + 0.5 * zs(ct))
                for m, (h, mg) in res.items():
                    row[f"{m}_{tag}"] = h
                    row[f"{m}_{tag}_margin"] = mg
                    row[f"{m}_{tag}_err"] = cdist(h, truth, n) if truth >= 0 and h >= 0 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).copy()
    out = _REPO / "study" / "logs" / ("unwrap_explore_eval_20260928.csv" if FIX_BOUNDARY
                                      else "unwrap_explore_eval_rawboundary_20260928.csv")
    df.to_csv(out, index=False)
    gt = df[(df.truth >= 0) & df.ecg_ok]
    print(f"{len(df)} folders; ground truth with valid ECG: {len(gt)} (26 frames {int((gt.n == 26).sum())}, "
          f"32 frames {int((gt.n == 32).sum())})\n")
    meths = [c[:-4] for c in df.columns if c.endswith("_err")]
    tab = []
    for m in meths:
        e = gt[m + "_err"]
        tab.append(dict(method=m, exact=(e == 0).mean(), within1=(e <= 1).mean(),
                        exact26=(e[gt.n == 26] == 0).mean(), exact32=(e[gt.n == 32] == 0).mean()))
    print(pd.DataFrame(tab).sort_values("exact", ascending=False).round(3).to_string(index=False))
    noecg = df[(df.truth >= 0) & ~df.ecg_ok]
    for m in [m for m in meths if m.startswith("adj") or m.startswith("cont")]:
        e = noecg[m + "_err"]
        print(f"  no-ECG ground truth ({len(noecg)}): {m} exact {(e == 0).mean():.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
