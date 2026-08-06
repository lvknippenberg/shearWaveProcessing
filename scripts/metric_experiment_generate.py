"""Generate a round of the metric-validation experiment: N randomized recipes on the 25 V phantom,
each rendered as a 3-column (displacement / velocity / acceleration) space-time plot, with the recipe
and the push_specificity metric S (per quantity) saved to a per-round manifest. You then score the
plots blind (swp_gui/score_app.py) and we compare (scripts/metric_experiment_analyze.py).

Rounds live in their own subfolder ``metric_experiment/<round>/`` so nothing is overwritten; the
active round for the scorer is written to ``metric_experiment/current_round.txt``.

    <zea-python> scripts/metric_experiment_generate.py --round round2 --n 100 --seed 2 [--feasible]

``--feasible`` biases the sampler toward recipes that actually produce a wave (rel-reference mode,
a reasonable band-pass) and **rejection-samples on S_max > --threshold**, WITHOUT excluding any option
(every method/mode is still reachable, just less likely to yield a dud) - so the round spans the
non-zero metric range for better resolution.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))

import core                                              # noqa: E402
import registry as reg                                  # noqa: E402
from swp.viz.metrics import push_specificity, origin_coherence  # noqa: E402
from swp.viz.core.geometry import robust_clim           # noqa: E402

PHANTOM_25V = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom/DefaultPatient_SW_data_04-August-2026_13-28-32"
BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
QUANTITIES = ["displacement", "velocity", "acceleration"]


def round_dir(round_name):
    return os.path.join(BASE, round_name)


def set_current_round(round_name):
    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE, "current_round.txt"), "w", encoding="utf-8") as f:
        f.write(round_name)


def _rand_params(method, rng):
    p = {}
    for par in method.params:
        if par.kind == "bool":
            p[par.arg] = bool(rng.random() < 0.5)
        elif par.kind == "select":
            p[par.arg] = str(rng.choice(par.options))
        elif par.kind == "int":
            p[par.arg] = int(rng.integers(int(par.lo), int(par.hi) + 1))
        else:
            p[par.arg] = float(round(rng.uniform(par.lo, par.hi), 3))
    if "f_lo" in p and "f_hi" in p and p["f_lo"] >= p["f_hi"]:
        p["f_lo"], p["f_hi"] = min(p["f_lo"], p["f_hi"] - 50), max(p["f_lo"] + 50, p["f_hi"])
    return p


def _rand_chain(methods, maxn, rng):
    nonnone = [m for m in methods if m.name != "none"]
    k = int(rng.integers(0, maxn + 1))
    return [(m.name, _rand_params(m, rng)) for m in (rng.choice(nonnone) for _ in range(k))]


def random_recipe(rng, feasible=False, mline_source="horizontal_push"):
    est = rng.choice([m for m in reg.ESTIMATOR_METHODS if m.name != "rf_ncc"])  # coarse only
    if feasible:
        mode = "relative_to_reference" if rng.random() < 0.85 else "frame_to_frame"
    else:
        mode = str(rng.choice(reg.MODES))
    motion = _rand_chain(reg.MOTION_METHODS, 2, rng)
    if feasible and rng.random() < 0.8 and not any(
            m[0] in ("temporal_bandpass", "temporal_highpass") for m in motion):
        flo = float(rng.uniform(10, 150)); fhi = float(rng.uniform(max(flo + 100, 300), 900))
        motion = [("temporal_bandpass", {"f_lo": round(flo), "f_hi": round(fhi), "order": 2})] + motion
    return core.Recipe(
        mline_source=mline_source,
        iq_steps=_rand_chain(reg.IQ_METHODS, 2, rng),
        estimator=est.name, est_params=_rand_params(est, rng), mode=mode,
        motion_steps=motion,
        spatial_steps=_rand_chain(reg.SPATIAL_METHODS, 1, rng),
        temporal_steps=_rand_chain(reg.TEMPORAL_METHODS, 1, rng),
        offsets=int(rng.choice([1, 3, 5, 7, 9])),
        offset_step_mm=float(round(rng.uniform(0.4, 1.2), 2)),
        mline_agg=str(rng.choice(["mean", "median"])),
        # active SWE: symmetric ARF push -> only 'outward' or 'none'. leftward/rightward are for
        # passive SWE (valve->apex, single direction) and are excluded here.
        directional=str(rng.choice(["none", "outward"])),
    )


def focused_recipe(rng, mline_source="horizontal_push"):
    """Round-3 narrowing sampler: the family your round-1/2 scores favoured (Loupas + relative-to-
    reference + temporal band-pass + a smoothing step + outward directional + >=5 offsets), with its
    PARAMETERS varied so we can find the best specific settings. Weaker choices (bad band, no smoothing)
    are still drawn so the parameter that matters shows up."""
    flo = float(rng.choice([20, 40, 60, 80, 120, 160, 200]))
    fhi = float(rng.choice([350, 450, 550, 650, 750]))
    motion = [("temporal_bandpass", {"f_lo": int(flo), "f_hi": int(fhi), "order": 2})]
    if rng.random() < 0.25:
        motion = [("polynomial_drift", {"order": int(rng.choice([2, 3]))})] + motion
    # round-3 finding: within the good family, Gaussian is the best smoother and NLM OVER-smooths,
    # so favour Gaussian (varied sigma to find the optimum) and de-emphasise NLM.
    sc = rng.choice(["gauss", "median", "nlm", "none"], p=[0.45, 0.3, 0.1, 0.15])
    if sc == "nlm":
        spatial = [("nlm_denoise", {"h_um": float(round(rng.uniform(2, 12), 1)),
                                    "patch_size": int(rng.choice([5, 7])),
                                    "patch_distance": int(rng.choice([6, 9]))})]
    elif sc == "gauss":
        spatial = [("spatial_smooth", {"sigma_z_m": float(round(rng.uniform(300, 1200))),
                                       "sigma_x_m": float(round(rng.uniform(600, 2400)))})]
    elif sc == "median":
        spatial = [("spatial_median", {"size_z_m": float(round(rng.uniform(300, 900))),
                                       "size_x_m": float(round(rng.uniform(600, 1800)))})]
    else:
        spatial = []
    tc = rng.choice(["mean", "savgol", "none"], p=[0.5, 0.35, 0.15])
    if tc == "mean":
        temporal = [("temporal_moving_mean", {"window": int(rng.choice([3, 5, 7]))})]
    elif tc == "savgol":
        temporal = [("savgol_temporal", {"window": int(rng.choice([5, 7, 9, 11])),
                                         "polyorder": int(rng.choice([2, 3]))})]
    else:
        temporal = []
    return core.Recipe(
        mline_source=mline_source, estimator="loupas", mode="relative_to_reference",
        motion_steps=motion, spatial_steps=spatial, temporal_steps=temporal,
        offsets=int(rng.choice([5, 7, 9])), offset_step_mm=float(round(rng.uniform(0.5, 1.1), 2)),
        mline_agg="mean",
        # active SWE: symmetric push -> 'outward' or 'none' only (mostly outward, which round 1/2 favoured)
        directional=str(rng.choice(["outward", "none"], p=[0.75, 0.25])))


def render_plot(cols, path):
    """cols: {quantity: (res, S)}. Per-panel auto-clim, blind (no recipe/metric shown)."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, q in zip(axs, QUANTITIES):
        res, _ = cols[q]
        st = res.st
        unit = 1e6 if q == "displacement" else (1e3 if q == "velocity" else 1.0)
        d = st.data * unit
        rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        cl = (robust_clim(st.data, rc, 97) * unit) or float(np.nanpercentile(np.abs(d), 97)) or 1.0
        r = st.r * 1e3; t = st.t * 1e3
        ax.imshow(d, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r", vmin=-cl, vmax=cl,
                  aspect="auto", origin="upper")
        ax.axvline(res.r0 * 1e3, color="0.15", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(q, fontsize=11); ax.set_xlabel("r [mm]")
    axs[0].set_ylabel("t [ms]")
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def evaluate(acq, ml, rec):
    """Run the recipe for all three quantities -> {q: (res, push_specificity dict)}."""
    cols = {}
    for q in QUANTITIES:
        cfg = core.to_config(dataclasses.replace(rec, quantity=q), acq)
        rp = core.run_recipe(acq, ml, cfg)
        rn = core.run_recipe(acq, ml, cfg, nopush=True)
        cols[q] = (rp, push_specificity(rp, rn))
    return cols


def item_record(i, png_rel, rec, cols, **extra):
    d = dict(id=i, png=png_rel, recipe=dataclasses.asdict(rec),
             S={q: float(cols[q][1]["S"]) for q in QUANTITIES},
             C_push={q: float(cols[q][1]["C_push"]) for q in QUANTITIES},
             C_nopush={q: float(cols[q][1]["C_nopush"]) for q in QUANTITIES},
             oc={q: float(origin_coherence(cols[q][0].st, cols[q][0].r0)) for q in QUANTITIES},
             S_max=float(max(cols[q][1]["S"] for q in QUANTITIES)),
             S_best_quantity=max(QUANTITIES, key=lambda q: cols[q][1]["S"]))
    d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="round2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--feasible", action="store_true",
                    help="bias toward wave-producing recipes + reject plots whose feasibility gate "
                         "(origin_coherence, which round 1 showed best matches 'a wave is visible') is "
                         "<= threshold. No option is excluded (every method stays reachable).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="feasibility gate on max origin_coherence over quantities (default 0.5)")
    ap.add_argument("--focused", action="store_true",
                    help="round-3 narrowing: sample within the winning recipe family (focused_recipe)")
    ap.add_argument("--folder", default=PHANTOM_25V, help="dataset measurement folder")
    ap.add_argument("--meas", type=int, default=0, help="measurement index within the folder")
    ap.add_argument("--mline", default="horizontal_push",
                    help="'horizontal_push' (phantom) or 'auto'/'manual' (in-vivo anatomical .npz)")
    args = ap.parse_args()

    rdir = round_dir(args.round)
    plots_dir = os.path.join(rdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    r0cfg = core.Recipe(mline_source=args.mline)
    pts = core.mline_points_for(args.folder, args.meas) if args.mline != "horizontal_push" else None
    acq = core.load_acq(args.folder, args.meas, r0cfg, mline_points=pts)
    ml = core.load_mline_for(args.folder, args.meas, acq, r0cfg)
    print(f"[{args.round}] loaded {os.path.basename(args.folder)} meas{args.meas} mline={args.mline}; "
          f"feasible={args.feasible} threshold={args.threshold} focused={args.focused}")

    manifest, t0, i, attempts = [], time.time(), 0, 0
    max_attempts = args.n * (60 if args.feasible else 5)
    while i < args.n and attempts < max_attempts:
        attempts += 1
        rec = (focused_recipe(rng, mline_source=args.mline) if args.focused
               else random_recipe(rng, feasible=args.feasible, mline_source=args.mline))
        try:
            cols = evaluate(acq, ml, rec)
        except Exception:  # noqa: BLE001 - degenerate params -> resample
            continue
        # feasibility gate = origin_coherence (round 1: best proxy for "a wave is visible"; and
        # non-circular for validating push_specificity, which is what we're testing).
        oc_max = max(origin_coherence(cols[q][0].st, cols[q][0].r0) for q in QUANTITIES)
        if args.feasible and oc_max <= args.threshold:
            continue                                     # reject duds (option still reachable next draw)
        png = os.path.join(plots_dir, f"plot_{i:03d}.png")
        render_plot({q: (rp, cols[q][1]["S"]) for q, (rp, _) in cols.items()}, png)
        manifest.append(item_record(i, os.path.relpath(png, rdir), rec, cols,
                                    handpicked=False, expectation="", label="random"))
        i += 1
        if i % 10 == 0:
            print(f"  {i}/{args.n}  (accept {i}/{attempts} = {i/attempts:.0%}, {time.time()-t0:.0f}s)")

    with open(os.path.join(rdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"folder": args.folder, "meas": args.meas, "mline": args.mline,
                   "round": args.round, "n": len(manifest), "seed": args.seed,
                   "feasible": args.feasible, "focused": args.focused, "items": manifest}, f, indent=2)
    set_current_round(args.round)
    print(f"\n[{args.round}] wrote {len(manifest)} plots (from {attempts} draws) -> {rdir}")
    print(f"active round set to '{args.round}'. Score with  streamlit run swp_gui/score_app.py")


if __name__ == "__main__":
    main()
