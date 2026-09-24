"""The passive methods atlas on all 15 labelled windows, on the old and the new M-lines.

Extends ``passive_methods_atlas.py`` (which built the two published atlases on 7 windows) to the
full hand-labelled set, grouped by the **velocity-panel confidence score**
(``study/logs/panel_confidence.csv``, treated as blind):

    clear      C019 w1, C020 w1, C002 w0, C018 w0, C027 w0, C039 w0
    plausible  C003 w1, C004 w1, C022 w1, C029 w1, C037 w1, C040 w2, C042 w0
    guess/none C045 w0 (guess), C008 w2 (none)

Three M-line sources per window:

    b1  the segment that was hand-labelled: the per-event buffer-1 line, or its left / right half
    b3  the line redrawn on the phase-matched focused-beam frame (buffer 3,
        ``draw_labelled_mlines_b3.py``; ``passive_win<i>_mline_b3.npz``), used in full
    b3to4  the b3 line moved onto buffer 4 by the local rigid motion between the buffer-3 frame
        and buffer 4 at the event (``map_mlines_b3_to_b4.py``; ``passive_win<i>_mline_b3to4.npz``)

Both recipe sets of the atlas are run on both line sources, loading each acquisition once:
``v1`` (view-A displacement base: quantity, bands, directional, smoothing, M-lines, motion,
literature recipes, production views) and ``v2`` (velocity 15-150 Hz base: spatial, temporal,
M-lines, combined, SVD, CFWI).

Score per panel (no hand line needed, so it works on both line sources): the tracking of the
strongest straight line in the panel (normalised Radon, 1-20 m/s) - mean |signal| on it / panel
RMS. A recipe that separates waves from noise scores high on the clear group and low on the
guess/none group; one that invents structure raises the guess/none group too.

    python study/analysis/passive_atlas_all15.py            # build (network, ~70 min) + figures + table
    python study/analysis/passive_atlas_all15.py --figures  # from the cache
    python study/analysis/passive_atlas_all15.py --lines b3to4 --sets v2   # build a subset
-> study/analysis/atlas15_cache/ (not tracked), report/passive_methods/figures15/,
   study/logs/passive_atlas15_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402
import passive_methods_atlas as A                              # noqa: E402  (recipes + helpers)

CACHE = _REPO / "study" / "analysis" / "atlas15_cache"
FIGDIR = _REPO / "report" / "passive_methods" / "figures15"
CONFIG = str(_REPO / "configs" / "passive.yaml")
GROUPS = ("clear", "plausible", "guess/none")
LINES = ("b1", "b3", "b3to4")


def windows():
    """[(subject, window, part, event, group)] from the labelled set + velocity-panel scores."""
    conf = {}
    for r in csv.DictReader(open(_REPO / "study/logs/panel_confidence.csv")):
        if r["quantity"] == "velocity":
            conf[(r["subject"], int(r["window"]), r["part"])] = r["meaning"]
    out = []
    for c in json.load(open(_REPO / "study/logs/labelled_panels.json")):
        k = (c["subject"], int(c["window"]), c["part"])
        s = conf.get(k, "none")
        g = "clear" if s == "clear" else "plausible" if s == "plausible" else "guess/none"
        out.append((*k, c["label"], g, c["folder"]))
    order = {g: i for i, g in enumerate(GROUPS)}
    return sorted(out, key=lambda w: (order[w[4]], w[3], w[0]))


def _npz(line, set_name, subj, win):
    return CACHE / line / set_name / f"{subj}_win{win}.npz"


def build(force=False, lines_sel=LINES, sets_sel=None):
    import swp.passive as SP
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline
    from passive_mline_split import split_line
    for subj, win, part, event, group, folder_rel in windows():
        todo = [(ln, s) for ln in lines_sel for s in (sets_sel or A.SETS)
                if force or not _npz(ln, s, subj, win).exists()]
        if not todo:
            continue
        folder = f"{P.RAW_DATA}/{folder_rel}"
        print(f"{subj} win{win} ({group}, {event})", flush=True)
        cfg, p = SP._paths(folder, CONFIG)
        st, ws = SP.read_windows(p["windows_json"])
        w = ws[win]
        n = cfg["mline"].get("n_samples", 250)
        old = SP._load_line(SP._window_npz(p["mlines"], win), n)
        lines = {"b1": old if part == "full" else mline_from_points(split_line(old, n)[part], n)}
        for ln in ("b3", "b3to4"):
            f = os.path.join(p["mlines"], f"passive_win{win}_mline_{ln}.npz")
            if os.path.exists(f):
                lines[ln] = SP._load_line(f, n)
        acq = SP.load_acq(folder, CONFIG)
        i0 = SP._frame_at_time(acq.t, w.t0 - A.PAD_S)
        i1 = SP._frame_at_time(acq.t, w.t1 + A.PAD_S) + 1
        acq_t = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        del acq
        for ln, set_name in todo:
            if ln not in lines:
                print(f"   no {ln} line - skipped", flush=True)
                continue
            ml = lines[ln]
            acq_w = A._crop(acq_t, ml)
            base = SP._build_views(cfg, acq_w)[0][1]
            panels = {}
            for fam, rows in A.SETS[set_name]().items():
                for label, ov in rows:
                    try:
                        res = run_pipeline(acq_w, ml, dataclasses.replace(base, **ov), focus=None)
                        panels[f"{fam}|{label}"] = (res.st.data.astype(np.float32), res.st.t)
                    except Exception as exc:                    # noqa: BLE001
                        print(f"   {ln}/{set_name} {fam}|{label}: FAILED {type(exc).__name__}: {exc}", flush=True)
            out = _npz(ln, set_name, subj, win)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, keys=np.array(list(panels)), r=ml.r, t_peak=w.t_peak,
                                event=event, group=group, part=part, line_x=ml.x * 1e3, line_z=ml.z * 1e3,
                                **{f"d{i}": v[0] for i, v in enumerate(panels.values())},
                                **{f"t{i}": v[1] for i, v in enumerate(panels.values())})
            print(f"   {ln}/{set_name}: {len(panels)} panels", flush=True)


def _load(line, set_name):
    out = []
    for subj, win, part, event, group, _ in windows():
        f = _npz(line, set_name, subj, win)
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        out.append(dict(name=f"{subj[-3:]} {event}", group=group, z=z,
                        panels={k: (z[f"d{i}"], z[f"t{i}"]) for i, k in enumerate(z["keys"])}))
    return out


def _panel_grid(wins, fam, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = len(wins)
    fig, axes = plt.subplots(len(labels), ncol, figsize=(2.3 * max(ncol, 4), 1.55 * len(labels) + 0.7),
                             squeeze=False)
    for j, W in enumerate(wins):
        tp, r = float(W["z"]["t_peak"]), W["z"]["r"] * 1e3
        for i, lab in enumerate(labels):
            ax = axes[i, j]
            d, t = W["panels"].get(f"{fam}|{lab}", (None, None))
            if d is None:
                ax.axis("off"); continue
            d, t = A._edge_crop(lab, d, t)
            clim = np.percentile(np.abs(d), 99) or 1.0
            ax.imshow(d.T, aspect="auto", cmap="RdBu_r", vmin=-clim, vmax=clim, origin="lower",
                      extent=((t[0] - tp) * 1e3, (t[-1] - tp) * 1e3, r[0], r[-1]))
            ax.axvline(0, color="k", lw=0.4, ls=":")
            ax.tick_params(labelsize=5, length=2)
            if i == 0:
                ax.set_title(f"{W['name']} [{W['group']}]", fontsize=6.5)
            if j == 0:
                ax.set_ylabel(lab if len(lab) < 30 else lab[:28] + "...", fontsize=5.5)
            if i == len(labels) - 1:
                ax.set_xlabel("t - event [ms]", fontsize=5.5)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(h_pad=0.3, w_pad=0.3, rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 85})
    plt.close(fig)


def figures(line="b3"):
    """Per set and family, two pages: the clear group, then plausible + guess/none."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for set_name in A.SETS:
        wins = _load(line, set_name)
        if not wins:
            continue
        for fam, rows in A.SETS[set_name]().items():
            labels = [lab for lab, _ in rows]
            for part, sel in (("clear", ("clear",)), ("rest", ("plausible", "guess/none"))):
                ws = [W for W in wins if W["group"] in sel]
                if ws:
                    name = f"{line}_{set_name}_{fam}_{part}.jpg"
                    _panel_grid(ws, fam, labels, FIGDIR / name,
                                f"{set_name} / {fam} - {' + '.join(sel)} windows - {line} M-lines")
        print(f"figures {line}/{set_name} -> {FIGDIR}", flush=True)


def line_comparison():
    """Anatomy + the default velocity recipe on the old (b1), new (b3) and mapped (b3to4) line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import swp.passive as SP
    from swp.mline.select import load_bmode_frame
    key = "spatial|Gaussian 0.6 x 1.2 mm (default)"      # v2 default recipe
    oldd = {W["name"]: W for W in _load("b1", "v2")}
    newd = {W["name"]: W for W in _load("b3", "v2")}
    mapd = {W["name"]: W for W in _load("b3to4", "v2")}
    ws = windows()
    for part, sel in (("clear", ("clear",)), ("rest", ("plausible", "guess/none"))):
        rows = [(w, oldd[f"{w[0][-3:]} {w[3]}"]) for w in ws
                if w[4] in sel and f"{w[0][-3:]} {w[3]}" in oldd]
        fig, axes = plt.subplots(4, len(rows), figsize=(2.3 * max(len(rows), 4), 9.2), squeeze=False)
        for j, ((subj, win, part_, event, group, folder_rel), W) in enumerate(rows):
            folder = f"{P.RAW_DATA}/{folder_rel}"
            cfg, p = SP._paths(folder, CONFIG)
            recs_path = os.path.join(p["mlines"], "passive_mlines_b3.json")
            recs = json.load(open(recs_path)) if os.path.exists(recs_path) else {}
            rec = recs.get(str(win), {})
            Wn = newd.get(W["name"])
            Wm = mapd.get(W["name"])
            # row 0: buffer-3 frame with both lines; row 1: note
            ax = axes[0, j]
            if "frame" in rec:
                img, coords, _ = load_bmode_frame(os.path.join(p["output"], SP.bmode_file(3)), rec["frame"])
                xs, zs = coords[0, :, 0] * 1e3, coords[:, 0, -1] * 1e3
                ax.imshow(img, cmap="gray", aspect="auto", extent=(xs[0], xs[-1], zs[-1], zs[0]))
            ax.plot(W["z"]["line_x"], W["z"]["line_z"], "--", color="cyan", lw=1.0, label="old (b1)")
            if Wn is not None:
                ax.plot(Wn["z"]["line_x"], Wn["z"]["line_z"], "-", color="yellow", lw=1.4, label="new (b3)")
            if Wm is not None:
                ax.plot(Wm["z"]["line_x"], Wm["z"]["line_z"], "-", color="red", lw=1.0, label="mapped (b3to4)")
            cx, cz = np.mean(W["z"]["line_x"]), np.mean(W["z"]["line_z"])
            ax.set_xlim(cx - 35, cx + 35); ax.set_ylim(cz + 30, cz - 30)
            ax.set_title(f"{W['name']} [{group}]\nb3 frame R+{rec.get('frame_phase_ms', float('nan')):.0f} "
                         f"(event R+{rec.get('event_phase_ms', float('nan')):.0f})"
                         + (" kept old" if rec.get("kept_old_segment") else ""), fontsize=6)
            ax.tick_params(labelsize=5)
            if j == 0:
                ax.legend(fontsize=5, loc="lower left")
            for i, (lab, Wx) in enumerate((("old line (b1)", W), ("new line (b3)", Wn),
                                           ("mapped line (b3to4)", Wm)), start=1):
                ax = axes[i, j]
                if Wx is None or key not in Wx["panels"]:
                    ax.axis("off"); continue
                d, t = Wx["panels"][key]
                tp, r = float(Wx["z"]["t_peak"]), Wx["z"]["r"] * 1e3
                clim = np.percentile(np.abs(d), 99) or 1.0
                ax.imshow(d.T, aspect="auto", cmap="RdBu_r", vmin=-clim, vmax=clim, origin="lower",
                          extent=((t[0] - tp) * 1e3, (t[-1] - tp) * 1e3, r[0], r[-1]))
                ax.axvline(0, color="k", lw=0.4, ls=":")
                ax.tick_params(labelsize=5, length=2)
                if j == 0:
                    ax.set_ylabel(f"{lab}\nr [mm]", fontsize=6)
                if i == 3:
                    ax.set_xlabel("t - event [ms]", fontsize=5.5)
        fig.suptitle("Old (buffer-1 segment), new (buffer-3) and mapped (buffer 3 -> 4) M-line, default velocity "
                     "15-150 Hz recipe. Top: buffer-3 frame (the mapped line belongs to buffer 4)", fontsize=8)
        fig.tight_layout(h_pad=0.3, w_pad=0.3, rect=[0, 0, 1, 0.97])
        fig.savefig(FIGDIR / f"line_comparison_{part}.png", dpi=110)
        plt.close(fig)
    print("line comparison figures written", flush=True)


def scores():
    """Best-line tracking per panel, summarised per recipe and group, for both line sources."""
    from swp.viz.metrics import normalized_radon_speed
    from swp.viz.speed.spacetime import SpaceTime
    from swp.provenance import stamp_text
    rows = []
    for line in LINES:
        for set_name in A.SETS:
            wins = _load(line, set_name)
            for fam, recipes in A.SETS[set_name]().items():
                for lab, _ in recipes:
                    per = {g: [] for g in GROUPS}
                    for W in wins:
                        d, t = W["panels"].get(f"{fam}|{lab}", (None, None))
                        if d is None:
                            continue
                        d, t = A._edge_crop(lab, d, t)
                        per[W["group"]].append(normalized_radon_speed(SpaceTime(d, W["z"]["r"], t, "x"))["tracking"])
                    med = {g: float(np.median(v)) if v else np.nan for g, v in per.items()}
                    rows.append(dict(line=line, set=set_name, family=fam, recipe=lab,
                                     clear=med["clear"], plausible=med["plausible"], guess_none=med["guess/none"],
                                     separation=med["clear"] - med["guess/none"],
                                     n=sum(len(v) for v in per.values())))
    out = _REPO / "study" / "logs" / "passive_atlas15_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(stamp_text(config={"windows": [w[:5] for w in windows()], "score": "normalized_radon tracking"}))
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    for line in LINES:
        print(f"\n=== {line} lines: best-line score, median per group (separation = clear - guess/none)")
        print(f"{'set':<4}{'family':<18}{'recipe':<52}{'clear':>7}{'plaus':>7}{'g/n':>7}{'sep':>7}")
        for r in rows:
            if r["line"] == line:
                print(f"{r['set']:<4}{r['family']:<18}{r['recipe'][:50]:<52}{r['clear']:7.2f}{r['plausible']:7.2f}"
                      f"{r['guess_none']:7.2f}{r['separation']:7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--figures", action="store_true", help="figures + table from the cache only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lines", nargs="*", default=list(LINES), help="line sources to build")
    ap.add_argument("--sets", nargs="*", default=None, help="recipe sets to build (default all)")
    a = ap.parse_args()
    if not a.figures:
        build(a.force, a.lines, a.sets)
    figures("b3")
    line_comparison()
    scores()


if __name__ == "__main__":
    main()
