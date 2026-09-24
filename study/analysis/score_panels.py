"""Record a confidence score for each hand-drawn panel: was a wavefront actually visible?

Why this matters. Every benchmark in `docs/passive_speed_estimation.md` compares automatic
estimators *against* the hand-drawn slopes, which invites reading the hand values as ground truth.
They are not: the drawing precision is about +/-25 %, and on top of that sits an untracked term
for panels where the propagation was unclear or not visible at all - the operator still had to put
a line somewhere.

Without a per-panel confidence, panel quality is a confound that cannot be separated from
estimator error. With it, a specific and useful question becomes testable: do the automatic bias,
the displacement-vs-velocity disagreements and the failed field-estimator projections concentrate
in the low-confidence panels? If they do, the methods work where a wave is genuinely visible and
the yield problem is about *event quality* rather than algorithms - a materially different
conclusion from the current one.

Shows each panel with its drawn line and the automatic fit, and takes one keypress:

    3  clear    - an unambiguous wavefront; the line follows something real
    2  plausible - a front is suggested but ambiguous; the line is a reasonable reading
    1  guess    - nothing clearly propagating; the line is a best guess
    0  none     - no usable panel at all
    s  skip (leave unscored)    b  go back one    q  save and quit

Scores are written to ``study/logs/panel_confidence.csv`` after every keypress, so the session can
be stopped and resumed at any point.

**Panels are cached first.** Building one costs a full buffer-4 acquisition read (~1-2 GB over the
network) plus the pipeline, about a minute - and the naive loop pays that per *panel*, reloading
the same acquisition for the two views of the same window. The cache pass groups panels by folder,
reads each acquisition once, and stores every panel as a small npz, after which scoring is
instant and re-scoring costs nothing.

    python study/analysis/score_panels.py --prepare   # one unattended pass, builds the cache
    python study/analysis/score_panels.py             # BLIND scoring (default since 2026-09-24)
    python study/analysis/score_panels.py --compare   # blind vs earlier non-blind scores
    python study/analysis/score_panels.py --unblinded # the original, non-blind display
    python study/analysis/score_panels.py --redo      # re-score panels already scored

**Blind by default (2026-09-24).** The first scoring pass showed the hand and automatic speeds in
the title and drew the hand line, so anchoring could not be excluded. Blind mode shows only the
panel: no subject, event label, speeds or hand line, in a seeded random order, and writes to
``study/logs/panel_confidence_blind.csv`` so the non-blind scores stay available for ``--compare``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))

import numpy as np

VIEWS = {"disp bp10-150 gauss mean3": "displacement",
         "velocity bp15-90 gauss1.0 mean5": "velocity"}
LEVELS = {"3": "clear", "2": "plausible", "1": "guess", "0": "none"}
OUT_UNBLINDED = _REPO / "study" / "logs" / "panel_confidence.csv"
OUT_BLIND = _REPO / "study" / "logs" / "panel_confidence_blind.csv"
OUT = OUT_BLIND
CACHE = _REPO / "study" / "analysis" / "panel_cache"


def _cache_path(c, quantity):
    return CACHE / f"{c['subject']}_win{c['window']}_{c['part']}_{quantity}.npz"


def prepare_cache(panels, config, root, force=False):
    """Build every panel once, grouped by folder so each acquisition is read a single time."""
    import dataclasses
    import swp.passive as P
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline
    from passive_mline_split import split_line

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [c for c in panels
            if force or not all(_cache_path(c, q).exists() for q in VIEWS.values())]
    print(f"caching {len(todo)} window(s) of {len(panels)} "
          f"({len(panels) - len(todo)} already cached)\n")
    for i, c in enumerate(todo, 1):
        folder = f"{root}/{c['folder']}"
        print(f"[{i}/{len(todo)}] {c['subject']} win{c['window']} {c['part']}", flush=True)
        try:
            cfg, p = P._paths(folder, config)
            st, ws = P.read_windows(p["windows_json"])
            w = ws[c["window"]]
            acq = P.load_acq(folder, config)          # the expensive read - once per window
            n = cfg["mline"].get("n_samples", 250)
            ml_full = P._load_line(P._window_npz(p["mlines"], c["window"]), n)
            ml = (ml_full if c["part"] == "full"
                  else mline_from_points(split_line(ml_full, n)[c["part"]], n))
            i0 = P._frame_at_time(acq.t, w.t0 - 0.02)
            i1 = P._frame_at_time(acq.t, w.t1 + 0.02) + 1
            acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
            picks = json.load(open(Path(folder) / "output/swp_passive/manual_slopes.json"))
            for vname, vcfg in P._build_views(cfg, acq):
                if vname not in VIEWS:
                    continue                          # only the two views that were hand-drawn
                res = run_pipeline(acq_w, ml, vcfg, focus=None)
                pk = picks.get(f"win{c['window']}|{c['part']}|{vname}")
                np.savez_compressed(
                    _cache_path(c, VIEWS[vname]),
                    data=res.st.data, r=res.st.r, t=res.st.t,
                    quantity=str(res.st.quantity), mline_mm=float(ml.r[-1] * 1e3),
                    label=str(w.label or "?"), auto=float(
                        pk["auto_speed_m_s"] if pk else np.nan),
                    hand=float(pk["speed_m_s"]) if pk else np.nan,
                    points=np.array(pk["points"][:2], float) if pk else np.zeros((2, 2)))
            del acq, acq_w
        except Exception as e:                                        # noqa: BLE001
            print(f"     FAILED: {type(e).__name__} {e}", flush=True)
    print(f"\n-> {CACHE}")


def load_scores(path=None):
    path = path or OUT
    if not path.exists():
        return {}
    with open(path) as fh:
        return {(r["subject"], int(r["window"]), r["part"], r["quantity"]): r
                for r in csv.DictReader(fh)}


def save_scores(scores):
    rows = list(scores.values())
    if not rows:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "window", "part", "label", "quantity",
                                           "score", "meaning", "hand_speed", "note"])
        w.writeheader()
        w.writerows(rows)


def compare_scores():
    """Agreement between the blind and the earlier non-blind scores, and whether the estimator-
    error-by-confidence trend (docs/passive_speed_estimation.md) survives blind scoring."""
    blind, open_ = load_scores(OUT_BLIND), load_scores(OUT_UNBLINDED)
    both = sorted(set(blind) & set(open_))
    if not both:
        print(f"no panels scored in both {OUT_BLIND.name} and {OUT_UNBLINDED.name}")
        return
    b = np.array([int(blind[k]["score"]) for k in both])
    o = np.array([int(open_[k]["score"]) for k in both])
    print(f"{len(both)} panels scored both ways: identical {np.mean(b == o):.0%}, within one level "
          f"{np.mean(np.abs(b - o) <= 1):.0%}, mean blind - non-blind {np.mean(b - o):+.2f}")
    print(f"{'level':<11}{'blind n':>8}{'non-blind n':>13}")
    for lev, name in sorted(LEVELS.items(), reverse=True):
        print(f"{name:<11}{np.sum(b == int(lev)):8d}{np.sum(o == int(lev)):13d}")
    # the headline: automatic-fit bias by confidence, under both scorings
    for tag, sc in (("non-blind", open_), ("blind", blind)):
        print()
        print(f"automatic |c|/hand by {tag} confidence:")
        for lev, name in sorted(LEVELS.items(), reverse=True):
            ratios = []
            for k in both:
                if sc[k]["score"] != lev:
                    continue
                z = np.load(_cache_path({"subject": k[0], "window": k[1], "part": k[2]}, k[3]))
                h, au = float(z["hand"]), float(z["auto"])
                if np.isfinite(h) and np.isfinite(au) and h:
                    ratios.append(abs(au) / abs(h))
            if ratios:
                print(f"  {name:<10} n={len(ratios):2d}  median {np.median(ratios):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default=str(_REPO / "study" / "logs" / "labelled_panels.json"))
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--redo", action="store_true", help="also show panels already scored")
    ap.add_argument("--prepare", action="store_true",
                    help="build the panel cache and exit (unattended, ~1 min per window)")
    ap.add_argument("--force", action="store_true", help="prepare: rebuild cached panels")
    ap.add_argument("--unblinded", action="store_true",
                    help="original display (speeds, hand line, subject) -> panel_confidence.csv")
    ap.add_argument("--seed", type=int, default=20260924, help="blind mode: panel order")
    ap.add_argument("--compare", action="store_true", help="blind vs non-blind agreement, then exit")
    a = ap.parse_args()
    global OUT
    OUT = OUT_UNBLINDED if a.unblinded else OUT_BLIND
    if a.compare:
        return compare_scores()

    # Import order matters. Several swp.viz.viz modules call matplotlib.use("Agg") at import
    # time, so the interactive backend must be selected AFTER they are loaded or the figures are
    # created on a non-interactive canvas and never appear.
    from manual_slope import draw_panel

    panels_all = json.load(open(a.panels))
    if a.prepare:
        return prepare_cache(panels_all, a.config, a.root, force=a.force)

    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    panels = json.load(open(a.panels))
    scores = load_scores()

    todo = []
    for c in panels:
        for view, quantity in VIEWS.items():
            key = (c["subject"], c["window"], c["part"], quantity)
            if key in scores and not a.redo:
                continue
            todo.append((c, view, quantity, key))
    if not a.unblinded:
        order = np.random.default_rng(a.seed).permutation(len(todo))
        todo = [todo[k] for k in order]
    print(f"{len(todo)} panel(s) to score ({len(scores)} already done)\n"
          f"keys: 3 clear | 2 plausible | 1 guess | 0 none | s skip | b back | q quit\n")

    missing = [t for t in todo if not _cache_path(t[0], t[2]).exists()]
    if missing:
        print(f"  {len(missing)} panel(s) are not cached. Run --prepare first:\n"
              f"    python study/analysis/score_panels.py --prepare\n")
        if len(missing) == len(todo):
            return
        todo = [t for t in todo if _cache_path(t[0], t[2]).exists()]

    class _ST:                       # what draw_panel needs, read straight from the cache
        def __init__(self, z):
            self.data, self.r, self.t = z["data"], z["r"], z["t"]
            self.quantity = str(z["quantity"])

    i = 0
    while 0 <= i < len(todo):
        c, view, quantity, key = todo[i]
        z = np.load(_cache_path(c, quantity), allow_pickle=False)
        st_obj = _ST(z)
        hand, auto_c = float(z["hand"]), float(z["auto"])
        pts = z["points"]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        if a.unblinded:
            title = (f"{c['subject']}  {c['label']}  win{c['window']} ({c['part']}, "
                     f"{float(z['mline_mm']):.0f} mm)  -  {quantity}\n"
                     f"hand {abs(hand):.2f} m/s, automatic {abs(auto_c):.2f} m/s   "
                     f"[{i + 1}/{len(todo)}]")
        else:   # blind: nothing that could anchor the judgement
            title = f"panel {i + 1}/{len(todo)}  -  {quantity}\nis a propagating wavefront visible?"
        draw_panel(ax, st_obj, title)
        if a.unblinded and np.isfinite(hand) and np.any(pts):
            (t1, r1), (t2, r2) = pts
            rl, rh = ax.get_ylim()
            rs = np.array([min(rl, rh), max(rl, rh)])
            ts = (t1 + (rs - r1) * (t2 - t1) / (r2 - r1) if abs(r2 - r1) > 1e-9
                  else np.array([t1, t2]))
            ax.plot(ts, rs, "-", color="lime", lw=2.2, label="hand-drawn")
            ax.set_ylim(rl, rh)
            ax.legend(fontsize=8, loc="lower right")
        ax.set_xlabel(ax.get_xlabel() + "        "
                      "3 clear | 2 plausible | 1 guess | 0 none | s skip | b back | q quit")

        got = {}

        def on_key(ev, got=got):
            if ev.key in tuple(LEVELS) + ("s", "b", "q"):
                got["k"] = ev.key
                fig.canvas.stop_event_loop()

        cid = fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", lambda _e: fig.canvas.stop_event_loop())
        fig.tight_layout()
        plt.show(block=False)
        fig.canvas.start_event_loop(timeout=-1)
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

        k = got.get("k")
        if k == "q" or k is None:
            break
        if k == "b":
            i = max(0, i - 1)
            continue
        if k != "s":
            scores[key] = dict(subject=c["subject"], window=c["window"], part=c["part"],
                               label=c["label"], quantity=quantity, score=int(k),
                               meaning=LEVELS[k],
                               hand_speed=round(abs(hand), 3) if np.isfinite(hand) else "",
                               note="")
            save_scores(scores)
            print(f"  [{i + 1}/{len(todo)}] {c['subject']:<12}{c['label']:<4}{quantity:<13}"
                  f"-> {k} ({LEVELS[k]})")
        i += 1

    save_scores(scores)
    print(f"\n-> {OUT}  ({len(scores)} panels scored)")
    if scores:
        import collections
        print("  " + str(collections.Counter(r["meaning"] for r in scores.values())))


if __name__ == "__main__":
    sys.exit(main())
