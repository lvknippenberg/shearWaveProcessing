"""Study-wide effect of the new passive default (velocity 15-150 Hz) against the v1 (displacement) run.

The study was reprocessed on 2026-09-24 with ``configs/passive.yaml`` (views: Gaussian 0.6 x 1.2 mm
default / unsmoothed / median 1.0 x 2.0 mm, all velocity 15-150 Hz, mean 3, 5 lines x 0.5 mm) on
the SAME windows and M-lines as before; the v1 outputs are in ``swp_passive/v1_displacement/``.
Automatic speeds are the signed slant stack over the whole line (search 1-20 m/s) - known to be
unreliable (docs/passive_speed_estimation.md), so the hand-drawn slopes are the reference where
they exist.

Per folder x window:
  v1_*   the three v1 views (A displacement bp10-150, B displacement bp5-150 median, C velocity
         bp15-90); v2_* the three new views (default, unsmoothed, median)
  railed |c| at a search bound (<= 1.01 or >= 19.9 m/s)
  agree  all three views of a run give the same sign and are within +/-25 % of their median
         (NB: v1 views differ in quantity/band/smoothing, v2 views only in the spatial filter, so
         v2 agreement is expected to be higher and is not a like-for-like quality comparison)
  steep  |c_default| / |c_unsmoothed|: > 1 = the Gaussian makes the front look faster (steeper
         in the M-mode) - the concern raised on report part 2
  hand   median |speed| of the full-line hand slopes (manual_slopes.json); error of each automatic
         estimate against it

    python study/analysis/passive_v2_rerun_summary.py
-> study/logs/passive_v2_rerun_windows.csv, study/montages/passive_v2_rerun_summary.png
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from swp import paths as P                                       # noqa: E402

V1 = {"disp bp10-150 gauss mean3": "v1_A_disp", "disp bp5-150 median NO-temporal": "v1_B_disp_median",
      "velocity bp15-90 gauss1.0 mean5": "v1_C_vel"}
V2 = {"velocity bp15-150 gauss0.6 mean3": "v2_default", "velocity bp15-150 unsmoothed mean3": "v2_unsmoothed",
      "velocity bp15-150 median1.0 mean3": "v2_median"}


def _rows(path):
    return json.load(open(path)) if os.path.exists(path) else []


def railed(c):
    return abs(c) <= 1.01 or abs(c) >= 19.9


def agree(cs):
    cs = np.asarray(cs, float)
    if len(cs) < 3 or not np.all(np.sign(cs) == np.sign(cs[0])):
        return False
    m = np.median(np.abs(cs))
    return bool(np.all(np.abs(np.abs(cs) - m) <= 0.25 * m))


def collect():
    out = []
    for sp in sorted(glob.glob(f"{P.RAW_DATA}/C0*/*/output/swp_passive/v1_displacement/passive_speeds.json")):
        d = os.path.dirname(os.path.dirname(sp))
        folder = os.path.dirname(os.path.dirname(d))
        new = _rows(os.path.join(d, "passive_speeds.json"))
        if not new or not all(r["view"] in V2 for r in new):
            continue                                   # not reprocessed (yet)
        old = _rows(sp)
        hand = {}
        mp = os.path.join(d, "manual_slopes.json")
        for k, v in (json.load(open(mp)).items() if os.path.exists(mp) else []):
            if v.get("part") == "full" and np.isfinite(v.get("speed_m_s", np.nan)):
                hand.setdefault(int(v["window"]), []).append(abs(float(v["speed_m_s"])))
        wins = sorted({r["window"] for r in new})
        for w in wins:
            r = dict(subject=os.path.basename(os.path.dirname(folder)), folder=os.path.basename(folder), window=w)
            for rows, names in ((old, V1), (new, V2)):
                for x in rows:
                    if x["window"] == w and x["view"] in names:
                        r[names[x["view"]]] = float(x["speed_m_s"])
                        r[names[x["view"]] + "_sem"] = float(x["semblance"])
                        r["label"] = x["label"]
            if w in hand:
                r["hand"] = float(np.median(hand[w]))
                r["n_hand"] = len(hand[w])
            out.append(r)
    return out


def main():
    from swp.provenance import stamp_text
    rows = collect()
    if not rows:
        raise SystemExit("no reprocessed folders found")
    keys = ["subject", "folder", "window", "label"] + [k for n in list(V1.values()) + list(V2.values())
                                                        for k in (n, n + "_sem")] + ["hand", "n_hand"]
    out = _REPO / "study/logs/passive_v2_rerun_windows.csv"
    with open(out, "w", newline="") as fh:
        fh.write(stamp_text(config=dict(source="swp_passive/passive_speeds.json vs v1_displacement/")))
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    def col(k, sel=None):
        return np.array([r.get(k, np.nan) for r in rows if sel is None or sel(r)], float)
    n = len(rows)
    print(f"{n} windows in {len({r['folder'] for r in rows})} folders\n")
    print(f"{'view':<16}{'railed':>8}{'median |c|':>12}{'median sem':>12}")
    for k in list(V1.values()) + list(V2.values()):
        c = col(k)
        ok = np.isfinite(c)
        print(f"{k:<16}{np.mean([railed(x) for x in c[ok]]):8.0%}{np.median(np.abs(c[ok])):12.2f}"
              f"{np.nanmedian(col(k + '_sem')):12.2f}")
    for run, names in (("v1", V1), ("v2", V2)):
        a = [agree([r[k] for k in names.values() if k in r]) for r in rows]
        print(f"{run}: all three views agree (sign, +/-25 %) in {sum(a)}/{n} windows")
    # smoothing: does the Gaussian steepen the front?
    ok = [r for r in rows if all(k in r and not railed(r[k]) for k in ("v2_default", "v2_unsmoothed", "v2_median"))
          and np.sign(r["v2_default"]) == np.sign(r["v2_unsmoothed"])]
    st = np.array([abs(r["v2_default"]) / abs(r["v2_unsmoothed"]) for r in ok])
    sm = np.array([abs(r["v2_median"]) / abs(r["v2_unsmoothed"]) for r in ok])
    print(f"\nsmoothing effect on the automatic speed ({len(ok)} windows, none railed, same sign):"
          f"\n  |c_gauss| / |c_unsmoothed|: median {np.median(st):.3f}, IQR {np.percentile(st, 25):.3f}-"
          f"{np.percentile(st, 75):.3f}, > 1.10 in {np.mean(st > 1.10):.0%}, < 0.90 in {np.mean(st < 0.90):.0%}"
          f"\n  |c_median| / |c_unsmoothed|: median {np.median(sm):.3f}, IQR {np.percentile(sm, 25):.3f}-"
          f"{np.percentile(sm, 75):.3f}")
    # against the hand slopes
    hr = [r for r in rows if "hand" in r]
    print(f"\nagainst the hand-drawn slopes ({len(hr)} windows with full-line picks): |auto| / hand")
    for k in ("v1_A_disp", "v1_C_vel", "v2_default", "v2_unsmoothed", "v2_median"):
        e = np.array([abs(r[k]) / r["hand"] for r in hr if k in r])
        if not len(e):
            continue
        print(f"  {k:<15} median {np.median(e):.2f}, within +/-25 % in {np.mean(np.abs(e - 1) <= 0.25):.0%}, "
              f"railed {np.mean([railed(r[k]) for r in hr if k in r]):.0%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    a1, a2 = col("v1_A_disp"), col("v2_default")
    m = np.isfinite(a1) & np.isfinite(a2)
    ax[0].loglog(np.abs(a1[m]), np.abs(a2[m]), "o", ms=4, alpha=0.6)
    ax[0].plot([1, 20], [1, 20], "k--", lw=0.8)
    ax[0].set_xlabel("v1 view A (displacement 10-150) |c| [m/s]"); ax[0].set_ylabel("v2 default (velocity 15-150) |c| [m/s]")
    ax[0].set_title("automatic speed, every window")
    ax[1].hist(st, bins=np.linspace(0.5, 1.5, 41), alpha=0.7, label="Gaussian 0.6x1.2 / unsmoothed")
    ax[1].hist(sm, bins=np.linspace(0.5, 1.5, 41), alpha=0.5, label="median 1.0x2.0 / unsmoothed")
    ax[1].axvline(1, color="k", lw=0.8); ax[1].legend(fontsize=7)
    ax[1].set_xlabel("speed ratio"); ax[1].set_title("does spatial smoothing steepen the front?")
    for k, mk in (("v1_A_disp", "s"), ("v1_C_vel", "^"), ("v2_default", "o")):
        ax[2].plot([r["hand"] for r in hr if k in r], [abs(r[k]) for r in hr if k in r], mk, ms=5, label=k, alpha=0.7)
    ax[2].plot([1, 8], [1, 8], "k--", lw=0.8); ax[2].set_yscale("log"); ax[2].set_xscale("log")
    ax[2].set_xlabel("hand-drawn |c| [m/s]"); ax[2].set_ylabel("automatic |c| [m/s]"); ax[2].legend(fontsize=7)
    ax[2].set_title("automatic vs hand slopes (full line)")
    fig.tight_layout()
    fp = _REPO / "study/montages/passive_v2_rerun_summary.png"
    fig.savefig(fp, dpi=120)
    print("\nwrote", out, "and", fp)


if __name__ == "__main__":
    main()
