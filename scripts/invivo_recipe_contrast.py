"""Does the ARF push add anything above cardiac motion - and does the answer depend on the recipe?

Every push is scored against its own **no-push control**: the pre-push reference split in two, the
second part tracked against the first, so the recipe sees cardiac motion only. A recipe that images
a real ARF wave makes the push clearly stronger than its control; one that images motion does not.

Why this script exists (review of 2026-09-24): every earlier in-vivo evaluation fixed the estimator
to displacement **relative to the averaged reference** (`scripts/archive/motion_removal.py`,
`task4b_config_contrast.py`). The cardiac-SWE literature instead estimates **frame-to-frame particle
velocity** (Caenen et al. 2023; Pernot 2011; Song 2016). At our 3.9 MHz demodulation the
reference-relative phase wraps at +/-98.6 um, and in-vivo wall motion over the reference plus
tracking window is 40-90 um, so the earlier "acquisition-limited" verdict was never tested against
the standard recipe.

Fairness rules, applied identically to every recipe and dataset:

* push and control ensembles have the **same number of frames** and are cut to that length
  *before* any temporal filtering, so band-pass edge transients are identical;
* the reference-relative push uses the last ``REF_SPLIT`` reference frames, the control the first
  ``REF_SPLIT`` - equal reference averaging;
* the first frame is dropped in both (the sliding pulse-inversion compound straddles the push);
* recipes differ only in estimator + filters; M-line, offsets and r0 are shared.

Datasets: the four in-vivo acquisitions with hand-drawn M-lines, plus two **positive controls**
where a real wave is known to exist - the static CIRS phantom and Caenen's pig data.

    python scripts/invivo_recipe_contrast.py                      # everything
    python scripts/invivo_recipe_contrast.py --datasets iv0818_61el caenen --recipes current caenen

Writes ``study/logs/invivo_recipe_contrast.csv`` (one row per push x recipe) and the figures
``study/montages/invivo_recipe_contrast{,_montage}.png``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui", "scripts"):
    if os.path.join(_ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _p))

from swp import paths as P                                     # noqa: E402
from swp.provenance import stamp_text                          # noqa: E402
from swp.viz.estimators import loupas_displacement             # noqa: E402
from swp.viz.filters import FIELD_FILTERS                      # noqa: E402
from swp.viz.filters.context import FilterCtx                  # noqa: E402
from swp.viz.filters.directional import outward_spacetime      # noqa: E402
from swp.viz.speed.spacetime import build_spacetime, SpaceTime  # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing              # noqa: E402

REF_SPLIT = 8          # control reference frames; the rest of the pre-push block is the control
SCORE_KW = dict(d_max=14e-3, cmin=1.5, t0_max=2.0e-3)
AMP_BAND = (2e-3, 14e-3)   # |r - r0| range for the amplitude comparison

BP = "temporal_bandpass"
RECIPES = {
    # the settled in-vivo recipe (configs/active.yaml view A)
    "current": dict(mode="relative_to_reference", quantity="displacement",
                    kz=1.0e-3, kx=0.0, shape="box", directional=True,
                    filters=[(BP, {"f_lo": 120, "f_hi": 700, "order": 2}),
                             ("spatial_smooth", {"sigma_z_m": 0.6e-3, "sigma_x_m": 1.2e-3}),
                             ("temporal_moving_mean", {"window": 3})]),
    # same filters, frame-to-frame particle velocity: isolates the estimator mode
    "f2f_current": dict(mode="frame_to_frame", quantity="velocity",
                        kz=1.0e-3, kx=0.0, shape="box", directional=True,
                        filters=[(BP, {"f_lo": 120, "f_hi": 700, "order": 2}),
                                 ("spatial_smooth", {"sigma_z_m": 0.6e-3, "sigma_x_m": 1.2e-3}),
                                 ("temporal_moving_mean", {"window": 3})]),
    # Caenen et al. 2023: lag-1 autocorrelation velocity, Gaussian 1.9x2.0 mm on the
    # autocorrelation before the angle, 6th-order 75-750 Hz zero-phase band-pass. No directional.
    "caenen": dict(mode="frame_to_frame", quantity="velocity",
                   kz=1.9e-3, kx=2.0e-3, shape="gaussian", directional=False,
                   filters=[(BP, {"f_lo": 75, "f_hi": 750, "order": 3})]),
    # ... plus the outward directional filter (Song et al. 2016 use one for ARF waves)
    "caenen_dir": dict(mode="frame_to_frame", quantity="velocity",
                       kz=1.9e-3, kx=2.0e-3, shape="gaussian", directional=True,
                       filters=[(BP, {"f_lo": 75, "f_hi": 750, "order": 3})]),
    # step 2: filter reference + tracking as one continuous record (longer record for the
    # 75 Hz high-pass; needs push-inclusive reference timestamps)
    "caenen_cont": dict(mode="frame_to_frame", quantity="velocity", continuous=True,
                        kz=1.9e-3, kx=2.0e-3, shape="gaussian", directional=True,
                        filters=[(BP, {"f_lo": 75, "f_hi": 750, "order": 3})]),
    # ... plus the published Giannantonio motion filter (linear on velocity = quadratic on
    # displacement), fitted on pre-push + post-wave samples
    "caenen_cont_gian": dict(mode="frame_to_frame", quantity="velocity", continuous=True,
                             kz=1.9e-3, kx=2.0e-3, shape="gaussian", directional=True,
                             filters=[("giannantonio_motion_filter", {"order": 1}),
                                      (BP, {"f_lo": 75, "f_hi": 750, "order": 3})]),
}
DEFAULT_RECIPES = list(RECIPES)
OFFSETS, STEP_M = 7, 0.8e-3


# ------------------------------------------------------------------ datasets
def _one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


DATASETS = {
    "iv0804_30V": dict(kind="folder", phantom=False,
                       folder=_one(os.path.join(P.VOLTAGE_SWEEP, "Invivo", "Luuk30V_*"))),
    "iv0804_40V": dict(kind="folder", phantom=False,
                       folder=_one(os.path.join(P.VOLTAGE_SWEEP, "Invivo", "Luuk40V_*"))),
    "iv0818_41el": dict(kind="folder", phantom=False, rpeak=True,
                        folder=_one(os.path.join(P.INVIVO_0818, "Luuk_41elements_*"))),
    "iv0818_61el": dict(kind="folder", phantom=False, rpeak=True,
                        folder=_one(os.path.join(P.INVIVO_0818, "Luuk_61elements_*"))),
    "phantom": dict(kind="phantom_sweep", phantom=True,
                    root=os.path.join(P.VOLTAGE_SWEEP, "Phantom")),
    "caenen": dict(kind="caenen", phantom=False),
}
POSITIVE_CONTROLS = ("phantom", "caenen")


def iter_pushes(name):
    """Yield (push label, acq, mline, r0) for a dataset."""
    d = DATASETS[name]
    if d["kind"] == "caenen":
        import caenen                                          # swp_gui/caenen.py
        for p in caenen.pushes():
            acq, ml = caenen.load(p)
            acq.meta = dict(acq.meta or {}, push_gap_s=float(acq.t[0] - acq.t_ref[-1]))  # real taxis
            yield f"p{p}", acq, ml, _r0_lateral_crossing(ml, float(acq.push_x))
        return
    import swe_lib as L
    folders = ([d["folder"]] if d["kind"] == "folder"
               else [f for f in sorted(glob.glob(os.path.join(d["root"], "*")))
                     if os.path.isdir(os.path.join(f, "output"))])
    for folder in folders:
        if not folder:
            continue
        for m in range(L.n_pushes(folder)):
            acq, ml, r0 = L.load_push(folder, m, phantom=d["phantom"], cache=False)
            if acq.t_ref is None or not (acq.meta or {}).get("push_gap_s"):
                acq = _with_sequence_timing(acq, folder)
            tag = f"m{m}" if d["kind"] == "folder" else f"{os.path.basename(folder)[-8:]}_m{m}"
            yield tag, acq, ml, r0


def _with_sequence_timing(acq, folder):
    """Reference timestamps from the folder's sequence parameters (phantom files carry none)."""
    import dataclasses
    from retrofit_push_gap import sw_geometry, corrected_t_reference
    g = sw_geometry(folder)
    return dataclasses.replace(acq, t_ref=corrected_t_reference(acq.ref_iq.shape[0], g),
                               meta=dict(acq.meta or {}, push_gap_s=g.push_gap_s()))


# ------------------------------------------------------------------ processing
def ensembles(acq, nopush):
    """(tracked iq, reference iq) of equal length for the push or its control."""
    ref_all = acq.ref_iq
    k = REF_SPLIT
    n = ref_all.shape[0] - k
    if nopush:
        return ref_all[k:], ref_all[:k]
    return acq.iq[:n], ref_all[-k:]


CONT_CONTROL_GAP = 12   # reference frames skipped at the control's join (see _continuous)


def _continuous(acq, rec, kw, nopush):
    """(field, t, first_tracking) for the continuous-record recipes: REF_SPLIT pre frames joined
    to an equal-length tracked block - the push window after the push, the control's inside the
    reference block.

    **The control's join must be as decorrelated as the push's.** In vivo the speckle correlation
    across the push is ~0.8 (vs ~0.95 within a block over the same time; phantom: no drop,
    study/analysis/push_gap_check.py), so the one displacement step that spans the push is noisy
    and, band-passed, leaks a spatially broad transient into the first ~4 ms. A control joined
    across one frame has no such step and made the push look 1.28x stronger in vivo (p = 0.001)
    with no wave present; skipping ``CONT_CONTROL_GAP`` = 12 frames (3.2 ms, correlation ~0.8)
    at the control's join removes it (1.00x / 0.89x) while the phantom's real push effect stays
    (2.08x). Both records are shortened to the same length accordingly.
    """
    import dataclasses
    from swp.viz.slowtime import continuous_record
    k, g = REF_SPLIT, CONT_CONTROL_GAP
    n = acq.ref_iq.shape[0] - k - g
    t_ref = np.asarray(acq.t_ref, float)
    if nopush:
        sub = dataclasses.replace(acq, iq=acq.ref_iq[k + g:], ref_iq=acq.ref_iq[:k],
                                  t=t_ref[k + g:] - t_ref[k + g], t_ref=t_ref[:k] - t_ref[k + g])
    else:
        sub = dataclasses.replace(acq, iq=acq.iq[:n], ref_iq=acq.ref_iq[-k:], t=np.asarray(acq.t, float)[:n],
                                  t_ref=t_ref[-k:])
    r = continuous_record(sub, loupas_displacement, kw, quantity=rec["quantity"], drop_first=1)
    return r.field, r.t, r.first_tracking


def spacetime(acq, ml, r0, rec, nopush):
    kw = dict(dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod, prf=acq.prf,
              kernel_z_m=rec["kz"], kernel_x_m=rec["kx"], kernel_shape=rec["shape"])
    if rec.get("continuous"):
        fld, t, first = _continuous(acq, rec, kw, nopush)
        ctx = FilterCtx(dz=acq.dz, dx=acq.dx, prf=acq.prf, t=t, x=acq.x, z=acq.z,
                        focus_x=float(acq.push_x or 0.0), f_demod=acq.f_demod, c=acq.c)
        for name, params in rec["filters"]:
            fld = FIELD_FILTERS[name](fld, ctx, **params)
        fld, t = fld[first:], t[first:] - t[first]
        st = build_spacetime(fld, acq.z, acq.x, ml, t, quantity=rec["quantity"],
                             n_offsets=OFFSETS, offset_step_m=STEP_M, agg="mean")
        data = outward_spacetime(st.data, st.r, r0) if rec["directional"] else st.data
        return SpaceTime(data, st.r, st.t, st.quantity)
    iq, ref = ensembles(acq, nopush)
    if rec["mode"] == "relative_to_reference":
        est = loupas_displacement(iq, mode="relative_to_reference", reference=ref, **kw)
    else:
        est = loupas_displacement(iq, mode="frame_to_frame", **kw)
    fld = est.velocity if rec["quantity"] == "velocity" else est.displacement
    fld = fld[1:]                                              # drop the push-straddling frame
    t = np.arange(fld.shape[0]) / acq.prf
    ctx = FilterCtx(dz=acq.dz, dx=acq.dx, prf=acq.prf, t=t, x=acq.x, z=acq.z,
                    f_demod=acq.f_demod, c=acq.c)
    for name, params in rec["filters"]:
        fld = FIELD_FILTERS[name](fld, ctx, **params)
    st = build_spacetime(fld, acq.z, acq.x, ml, t, quantity=rec["quantity"],
                         n_offsets=OFFSETS, offset_step_m=STEP_M, agg="mean")
    data = outward_spacetime(st.data, st.r, r0) if rec["directional"] else st.data
    return SpaceTime(data, st.r, st.t, st.quantity)


def amplitude(st, r0):
    d = np.abs(st.r - r0)
    sel = (d >= AMP_BAND[0]) & (d <= AMP_BAND[1])
    return float(np.sqrt(np.mean(st.data[:, sel] ** 2))) if sel.any() else float("nan")


def score(st, r0):
    import swe_lib as L
    oc, sym = L.scores(st, r0, **SCORE_KW)
    return oc, sym, amplitude(st, r0)


# ------------------------------------------------------------------ main
def run(datasets, recipes, out_csv, keep):
    rows, kept = [], {}
    for name in datasets:
        t0 = time.time()
        n = 0
        for tag, acq, ml, r0 in iter_pushes(name):
            n += 1
            for rn in recipes:
                rec = RECIPES[rn]
                try:
                    stp = spacetime(acq, ml, r0, rec, nopush=False)
                    stn = spacetime(acq, ml, r0, rec, nopush=True)
                except Exception as exc:                        # noqa: BLE001
                    print(f"  {name} {tag} {rn}: {exc}")
                    continue
                ocp, symp, ap = score(stp, r0)
                ocn, symn, an = score(stn, r0)
                rows.append(dict(dataset=name, push=tag, recipe=rn, oc_push=ocp, oc_nopush=ocn,
                                 sym_push=symp, sym_nopush=symn, amp_push=ap, amp_nopush=an,
                                 d_oc=ocp - ocn, log2_amp=float(np.log2(ap / an))))
                kept.setdefault((name, rn), []).append((tag, stp, stn, r0, rows[-1]["log2_amp"]))
        print(f"{name}: {n} pushes in {time.time() - t0:.0f}s", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write(stamp_text(config={"recipes": {k: RECIPES[k] for k in recipes},
                                   "REF_SPLIT": REF_SPLIT, "SCORE_KW": SCORE_KW,
                                   "AMP_BAND": AMP_BAND, "OFFSETS": OFFSETS, "STEP_M": STEP_M},
                           prefix="# "))
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv)
    return rows, kept


def summarise(rows, datasets, recipes):
    from scipy.stats import wilcoxon
    print(f"\n{'dataset':<13}{'recipe':<13}{'n':>4}{'amp ratio':>11}{'frac>1':>8}{'p':>9}"
          f"{'  dOC':>7}{'frac>0':>8}{'p':>9}")
    out = {}
    for ds in datasets:
        for rn in recipes:
            r = [x for x in rows if x["dataset"] == ds and x["recipe"] == rn]
            if not r:
                continue
            la = np.array([x["log2_amp"] for x in r]); do = np.array([x["d_oc"] for x in r])
            la, do_ = la[np.isfinite(la)], do[np.isfinite(do)]
            pa = wilcoxon(la).pvalue if len(la) > 5 else float("nan")
            po = wilcoxon(do_).pvalue if len(do_) > 5 else float("nan")
            out[(ds, rn)] = (la, do_)
            print(f"{ds:<13}{rn:<13}{len(r):4d}{2 ** np.median(la):11.2f}{np.mean(la > 0):8.2f}"
                  f"{pa:9.1e}{np.median(do_):7.3f}{np.mean(do_ > 0):8.2f}{po:9.1e}")
    return out


def figure_summary(stats, datasets, recipes, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    cols = plt.cm.tab10(np.arange(len(recipes)))
    for ax, i, lab in ((axes[0], 0, "log2(push / no-push RMS)"), (axes[1], 1, "origin coherence: push - no-push")):
        pos = 0
        ticks, labels = [], []
        for ds in datasets:
            for j, rn in enumerate(recipes):
                if (ds, rn) not in stats:
                    continue
                v = stats[(ds, rn)][i]
                ax.boxplot([v], positions=[pos], widths=0.7, showfliers=False,
                           medianprops=dict(color=cols[j], lw=2))
                ax.plot(pos + np.random.uniform(-0.2, 0.2, len(v)), v, ".", color=cols[j], ms=3, alpha=0.6)
                pos += 1
            ticks.append(pos - len(recipes) / 2 - 0.5); labels.append(ds + ("\n(+ control)" if ds in POSITIVE_CONTROLS else ""))
            pos += 1
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y")
    handles = [plt.Line2D([], [], color=cols[j], lw=3, label=rn) for j, rn in enumerate(recipes)]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("Push against its own no-push control, equal-length windows. Above 0 = the push adds "
                 "signal beyond cardiac motion.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path, dpi=130); plt.close(fig)
    print("wrote", path)


def figure_montage(kept, datasets, recipes, path):
    """One representative push per dataset (median contrast under the last recipe), push | control
    for each recipe - representative, not the best case."""
    rows = [ds for ds in datasets if all((ds, rn) in kept for rn in recipes)]
    fig, axes = plt.subplots(len(rows), 2 * len(recipes), figsize=(2.3 * 2 * len(recipes), 2.5 * len(rows)),
                             squeeze=False)
    import swe_lib as L
    for i, ds in enumerate(rows):
        ref = kept[(ds, recipes[-1])]
        order = np.argsort([x[4] for x in ref])
        tag = ref[order[len(order) // 2]][0]
        for j, rn in enumerate(recipes):
            _, stp, stn, r0, la = next(x for x in kept[(ds, rn)] if x[0] == tag)
            clim = L.draw(axes[i, 2 * j], stp, r0)
            L.draw(axes[i, 2 * j + 1], stn, r0, clim=clim)
            axes[i, 2 * j].set_title(f"{rn} push\n{ds} {tag}", fontsize=7)
            axes[i, 2 * j + 1].set_title(f"no-push (ratio {2 ** la:.2f})", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--recipes", nargs="+", default=DEFAULT_RECIPES, choices=list(RECIPES))
    ap.add_argument("--out", default=os.path.join(_ROOT, "study", "logs", "invivo_recipe_contrast.csv"))
    ap.add_argument("--figdir", default=os.path.join(_ROOT, "study", "montages"))
    a = ap.parse_args()
    rows, kept = run(a.datasets, a.recipes, a.out, keep=True)
    stats = summarise(rows, a.datasets, a.recipes)
    figure_summary(stats, a.datasets, a.recipes, os.path.join(a.figdir, "invivo_recipe_contrast.png"))
    figure_montage(kept, a.datasets, a.recipes, os.path.join(a.figdir, "invivo_recipe_contrast_montage.png"))


if __name__ == "__main__":
    main()
