"""Preliminary evaluation of the manual reading with the valves detector + window review (2026-10-01).

Read only (Z: folders + study/logs). Inputs:
  study/logs/passive_manual_valves_prelim/slopes.csv            passive_manual.py export, new reading
  study/logs/passive_manual_reference_2026-10-01_energy/...      the same folders, energy detector
  <folder>/output/swp_passive_manual/{windows,review}.json, st_win<i>.npz

1. Confidence per label, new vs old reading; the same events matched (label, t_peak within 60 ms).
2. The window review: the proposals (ROIs as 120 ms windows) against what was accepted (kept /
   moved / added / deleted), and how the fully automatic windows would have done.
3. Hand speeds: MVC vs AVC, beat to beat, and the automatic speeds of each view against the hand.
4. The in-phase basal segment (the "vertical then sloped" pattern): per window, the length of the
   line from r = 0 that moves in phase with its first 5 mm (zero-lag correlation >= 0.9, best lag
   <= 1.5 ms; velocity gauss, t inside the window), and whether it biases the automatic speed.

-> study/logs/passive_manual_valves_prelim/{summary.txt, windows.csv}, figure
   study/montages/passive_manual_valves_prelim/summary.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.manual import store as S                               # noqa: E402

D = os.path.join(REPO, "study", "logs", "passive_manual_valves_prelim")
OLD = os.path.join(REPO, "study", "logs", "passive_manual_reference_2026-10-01_energy", "passive_manual_slopes.csv")
FIG = os.path.join(REPO, "study", "montages", "passive_manual_valves_prelim")
ROOT = "Z:/raw_data"
COL = {"MVC": "#2a78d6", "AVC": "#eb6834", "AK": "#1f9e6e"}
INK2, SURF = "#52514e", "#fcfcfb"


def inphase_length(st, t0, t1, ref_mm=5.0, cc_min=0.9, lag_max_ms=1.5):
    """Length [mm] of the line from r = 0 moving in phase with its first ``ref_mm``."""
    v = st["v2_data"].astype(float)                       # velocity gauss
    t, r = st["v2_t"], st["v2_r"] * 1e3
    m = (t >= t0) & (t <= t1)
    ref = v[m][:, r <= ref_mm].mean(axis=1)
    dt = float(t[1] - t[0])
    nl = int(round(0.025 / dt))
    length = 0.0
    for k in range(r.size):
        x = v[m, k]
        ccs = [np.corrcoef(ref[nl:-nl], x[nl + L: x.size - nl + L])[0, 1] for L in range(-nl, nl + 1)]
        best = int(np.argmax(ccs)) - nl
        if ccs[nl] >= cc_min and abs(best) * dt * 1e3 <= lag_max_ms:
            length = r[k]
        elif r[k] > ref_mm:
            break
    return length


def review_stats(folder):
    p = S.Paths(folder)
    rev, win = S.read_json(p.review_json), S.read_json(p.windows_json)
    if not rev or not win:
        return None
    pro = rev.get("proposals") or []
    acc = rev["windows"]
    auto = [w for w in win["windows"] if not w.get("screened")]
    used = set()
    out = dict(proposed=len(pro), accepted=len(acc), kept=0, moved=0, added=0, shifts_ms=[], auto_cover=[])
    for w in acc:
        d = [(abs((q["t0"] + q["t1"]) / 2 - (w["t0"] + w["t1"]) / 2), j) for j, q in enumerate(pro)
             if j not in used and q["label"] == w["label"]]
        d = min(d) if d else (np.inf, None)
        if d[0] < 0.06:
            used.add(d[1])
            out["kept" if d[0] < 0.005 else "moved"] += 1
            out["shifts_ms"].append(d[0] * 1e3)
        else:
            out["added"] += 1
        # would the fully automatic window have covered this accepted one?
        cov = [max(0.0, min(a["t1"], w["t1"]) - max(a["t0"], w["t0"])) / (w["t1"] - w["t0"])
               for a in auto if a["label"] == w["label"]]
        if w["label"] in ("MVC", "AVC"):
            out["auto_cover"].append(max(cov, default=0.0))
    out["deleted"] = len(pro) - len(used)
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    new = pd.read_csv(os.path.join(D, "slopes.csv"))
    old = pd.read_csv(OLD)
    new = new[~new.skipped]
    new["speed"] = new.speed_m_s.abs()
    old["speed"] = old.speed_m_s.abs()
    L = [f"new reading: {new.folder.nunique()} folders, {len(new)} windows with a score "
         f"({', '.join(f'{k} {v}' for k, v in new.label.value_counts().items())})"]

    # 1. confidence ------------------------------------------------------------------------
    folders = set(new.folder)
    o = old[old.folder.isin(folders) & ~old.skipped]
    for name, df in (("new (valves + review)", new), ("old (energy), same folders", o)):
        L.append(f"  {name}: " + "; ".join(
            f"{lab} n={len(g)} usable(>=2) {100 * (g.confidence >= 2).mean():.0f}% clear {100 * (g.confidence == 3).mean():.0f}% "
            f"zero {int((g.confidence == 0).sum())}" for lab, g in df[df.label.isin(['MVC', 'AVC'])].groupby("label")))
    pairs = []
    for r in new.itertuples():
        c = o[(o.folder == r.folder) & (o.label == r.label)]
        if len(c):
            j = (c.t_peak_ms - r.t_peak_ms).abs().idxmin()
            if abs(c.t_peak_ms[j] - r.t_peak_ms) < 60:
                pairs.append(dict(label=r.label, dconf=r.confidence - c.confidence[j], new_speed=r.speed,
                                  old_speed=c.speed[j], dt=r.t_peak_ms - c.t_peak_ms[j],
                                  both_usable=(r.confidence >= 2) and (c.confidence[j] >= 2)))
    P = pd.DataFrame(pairs)
    L.append(f"  matched to an old window (same label, peak < 60 ms): {len(P)} of {len(new)}; confidence "
             f"up {(P.dconf > 0).sum()}, same {(P.dconf == 0).sum()}, down {(P.dconf < 0).sum()}")
    b = P[P.both_usable]
    if len(b):
        L.append(f"  both usable ({len(b)}): new/old speed median {np.median(b.new_speed / b.old_speed):.2f}, "
                 f"|diff| median {np.median(np.abs(b.new_speed - b.old_speed)):.2f} m/s")

    # 2. the window review -----------------------------------------------------------------
    stats = [s for s in (review_stats(f"{ROOT}/{sub}/{fol}") for sub, fol in
                         new[["subject", "folder"]].drop_duplicates().itertuples(index=False)) if s]
    tot = {k: sum(s[k] for s in stats) for k in ("proposed", "accepted", "kept", "moved", "added", "deleted")}
    sh = np.concatenate([s["shifts_ms"] for s in stats]) if stats else np.array([])
    ac = np.concatenate([s["auto_cover"] for s in stats]) if stats else np.array([])
    L.append(f"window review ({len(stats)} folders): proposed {tot['proposed']}, accepted {tot['accepted']}: "
             f"kept {tot['kept']}, moved {tot['moved']} (median {np.median(sh[sh >= 5]) if (sh >= 5).any() else 0:.0f} ms), "
             f"added {tot['added']}, deleted {tot['deleted']}")
    L.append(f"  fully automatic windows vs the accepted MVC/AVC windows: {(ac >= 0.95).sum()}/{ac.size} "
             f"(>= 95 % overlap), {(ac >= 0.8).sum()}/{ac.size} >= 80 %, none {(ac == 0).sum()}")

    # 3. speeds ----------------------------------------------------------------------------
    u = new[(new.confidence >= 2) & new.label.isin(["MVC", "AVC"])]
    for lab, g in u.groupby("label"):
        L.append(f"hand speed {lab} (conf >= 2, n={len(g)}): median {g.speed.median():.2f} m/s, IQR "
                 f"{g.speed.quantile(.25):.2f}-{g.speed.quantile(.75):.2f}; >= 6 m/s (lower bound) {(g.speed >= 6).sum()}")
    bb = []
    for f, g in u[u.label == "MVC"].groupby("folder"):
        if len(g) >= 2:
            s = g.speed.values
            bb.append(abs(s[0] - s[1]) / np.mean(s[:2]))
    if bb:
        L.append(f"MVC beat to beat (both usable, {len(bb)} folders): |diff| / mean median {100 * np.median(bb):.0f} %")
    for view in ("velocity gauss", "Keijzer velocity", "displacement gauss"):
        a = u[f"auto {view} [m/s]"].abs()
        ok = a < 19.5
        L.append(f"auto {view} / hand (conf >= 2, not railed {ok.sum()}): median {np.median(a[ok] / u.speed[ok]):.2f}")

    # 4. in-phase basal segment ------------------------------------------------------------
    rows = []
    for r in new.itertuples():
        p = S.Paths(f"{ROOT}/{r.subject}/{r.folder}")
        try:
            st = np.load(p.st_npz(int(r.window)))
            w = S.event_windows(p)["windows"][int(r.window)]
        except Exception as exc:                                         # noqa: BLE001
            print("  skip", r.folder, r.window, exc)
            continue
        ln = inphase_length(st, w["t0"], w["t1"])
        rows.append(dict(subject=r.subject, folder=r.folder, window=r.window, label=r.label,
                         confidence=r.confidence, hand=r.speed,
                         line_mm=float(st["v2_r"][-1] * 1e3), inphase_mm=ln))
    W = pd.DataFrame(rows)
    W["auto"] = new.set_index(["folder", "window"]).loc[list(zip(W.folder, W.window)), "auto velocity gauss [m/s]"].abs().values
    W.to_csv(os.path.join(D, "windows.csv"), index=False)
    L.append(f"in-phase basal segment (velocity gauss): >= 10 mm in {(W.inphase_mm >= 10).sum()}/{len(W)} windows "
             f"(MVC {(W[W.label == 'MVC'].inphase_mm >= 10).sum()}/{(W.label == 'MVC').sum()}, AVC "
             f"{(W[W.label == 'AVC'].inphase_mm >= 10).sum()}/{(W.label == 'AVC').sum()}); median {W.inphase_mm.median():.0f} mm")
    q = W[(W.confidence >= 2) & (W.auto < 19.5)]
    for name, g in (("in-phase >= 10 mm", q[q.inphase_mm >= 10]), ("in-phase < 10 mm", q[q.inphase_mm < 10])):
        if len(g):
            L.append(f"  auto/hand {name} (n={len(g)}): median {np.median(g.auto / g.hand):.2f}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(D, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")

    # figure -------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURF)
    labs = ["MVC", "AVC"]
    for k, (name, df) in enumerate((("old (energy)", o), ("new (valves + review)", new))):
        for i, lab in enumerate(labs):
            g = df[df.label == lab]
            bottom = 0
            for c, shade in zip((3, 2, 1, 0), (1.0, 0.7, 0.45, 0.2)):
                h = (g.confidence == c).sum()
                ax[0].bar(i * 3 + k, h, bottom=bottom, color=COL[lab], alpha=shade, edgecolor="white")
                if h:
                    ax[0].text(i * 3 + k, bottom + h / 2, str(c), ha="center", va="center", fontsize=7, color="white")
                bottom += h
            ax[0].text(i * 3 + k, -1.5, "old" if k == 0 else "new", ha="center", fontsize=8)
    ax[0].set_xticks([0.5, 3.5], labs)
    ax[0].set_ylabel("windows (stacked by confidence 3/2/1/0)")
    ax[0].set_title("Confidence, same folders: old detector vs valves + review", fontsize=9, loc="left")
    for lab in labs:
        g = u[u.label == lab]
        ax[1].scatter(g.speed, g[f"auto velocity gauss [m/s]"].abs().clip(upper=20), color=COL[lab], label=lab, s=22)
    ax[1].plot([0, 12], [0, 12], color=INK2, lw=0.8, ls="--")
    ax[1].set_xlabel("hand speed [m/s]")
    ax[1].set_ylabel("automatic (velocity gauss) [m/s]")
    ax[1].set_title("Automatic vs hand speed (conf >= 2)", fontsize=9, loc="left")
    ax[1].legend(frameon=False, fontsize=8)
    for lab in labs:
        g = W[W.label == lab]
        ax[2].scatter(g.inphase_mm + np.random.default_rng(0).uniform(-0.4, 0.4, len(g)),
                      (g.auto / g.hand).clip(upper=6), color=COL[lab], s=22, label=lab)
    ax[2].axhline(1, color=INK2, lw=0.8, ls="--")
    ax[2].set_xlabel("in-phase basal segment [mm]")
    ax[2].set_ylabel("automatic / hand speed")
    ax[2].set_title("Does the in-phase segment bias the automatic speed?", fontsize=9, loc="left")
    for a in ax:
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "summary.png"), dpi=100, facecolor=SURF)
    print("->", D, FIG)


if __name__ == "__main__":
    main()
