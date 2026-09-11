"""Task 4 -- does the recommended push (61 el / 1900 cyc) beat the old one (41 el / 1500 cyc)
in vivo?

Both 2026-08-18 in-vivo acquisitions are **R-peak triggered** (SW.WaitForRpeak = 1, unlike the
free-running 2026-08-04 data), 24 pushes at 20 Hz, so push index k corresponds to ~50*k ms after
the R-peak in both -- the two configurations are directly comparable push-by-push in cardiac phase.

For every push each configuration is run twice with the same recipe:
  * PUSH    -- the tracking ensemble after the ARF push;
  * NO-PUSH -- the split pre-push reference (2nd half tracked against the 1st), i.e. the same
               recipe applied to cardiac motion alone, with no shared frames.
A real ARF wave shows up as PUSH clearly better than its own NO-PUSH control; a "V" that is equally
strong in both is cardiac motion shaped by the outward-directional filter (report 11). Both are
scored over the same (control-length, ~8 ms) window so the comparison is like-for-like; the
montages still display the full push window.

    python scripts/task4_invivo_compare.py --root D:/swp_iv --outdir <dir> [--quantity ...]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402

PUSH_RATE_HZ = 20.0
REF_SPLIT = 8                 # pre-push frames used as the control's reference (rest are tracked)
# Metric window. The no-push control is limited to the pre-push block (~8 ms here) while the push
# ensemble runs ~16 ms, so BOTH are scored on the control's window (the push is truncated) with a
# moveout range that fits it - otherwise the slant-stack runs off the end of the short record and
# the origin-coherence normalisation breaks.
SCORE_KW = dict(d_max=14e-3, cmin=1.5, t0_max=2.0e-3)


def label_of(folder):
    b = os.path.basename(folder)
    for part in ("41elements_1500cycles_30V", "61elements_1900cycles_25V"):
        if part in b:
            el, cyc, V = part.split("_")
            return f"{el.replace('elements',' el')} / {cyc.replace('cycles',' cyc')} / {V}"
    return b[:32]


def run_config(folder, recipe, quantity):
    """Per push: (space-time, r0, oc, sym) for PUSH and for the NO-PUSH control."""
    out = []
    for m in range(L.n_pushes(folder)):
        row = {"m": m}
        sts = {}
        for tag, nopush in (("push", False), ("nopush", True)):
            try:
                sts[tag], r0 = L.st_for(folder, m, recipe, quantity=quantity, phantom=False,
                                        nopush=nopush, ref_split=REF_SPLIT)
            except Exception as exc:                              # noqa: BLE001
                print(f"    push {m} {tag} failed: {exc}")
                sts[tag] = None
        n_score = sts["nopush"].data.shape[0] if sts.get("nopush") is not None else None
        for tag in ("push", "nopush"):
            st = sts.get(tag)
            if st is None:
                row[tag] = None
                continue
            scored = L.truncate(st, n_score) if n_score else st
            oc, sym = L.scores(scored, r0, **SCORE_KW)
            row[tag] = (st.data, st.r, st.t, r0, oc, sym)
        L._ACQ_CACHE.pop((folder, m, False), None)
        if row.get("push"):
            print(f"    m{m:2d}  push oc={row['push'][4]:.2f} sym={row['push'][5]:.2f}"
                  + (f"   nopush oc={row['nopush'][4]:.2f} sym={row['nopush'][5]:.2f}"
                     if row.get("nopush") else "   nopush n/a"), flush=True)
        out.append(row)
    return out


def montage(rows, tag, title, outpath, quantity, ncol=6):
    """One panel per push (PUSH space-time), annotated with its own no-push control score."""
    have = [r for r in rows if r.get(tag)]
    nrow = int(np.ceil(len(have) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 3.1 * nrow), squeeze=False)
    unit = 1e3 if quantity == "velocity" else 1e6
    clim = float(np.median([np.percentile(np.abs(r[tag][0]), 99) for r in have])) * unit
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(have):
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        data, rr, tt, r0, oc, sym = r[tag]
        ax.imshow(data * unit, extent=(rr[0] * 1e3, rr[-1] * 1e3, tt[-1] * 1e3, tt[0] * 1e3),
                  cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto", origin="upper")
        ax.axvline(r0 * 1e3, color="0.2", ls="--", lw=0.8, alpha=0.7)
        ax.tick_params(labelsize=6)
        other = r.get("nopush" if tag == "push" else "push")
        extra = f"\nctrl oc {other[4]:.2f}" if other else ""
        ax.set_title(f"m{r['m']}  t={r['m'] / PUSH_RATE_HZ * 1e3:.0f} ms\n"
                     f"oc {oc:.2f}  sym {sym:.2f}{extra}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(outpath, dpi=105)
    plt.close(fig)
    print("wrote", outpath)


def summary(all_rows, outpath, quantity):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {0: "tab:blue", 1: "tab:red"}
    for i, (lbl, rows) in enumerate(all_rows):
        m = [r["m"] for r in rows if r.get("push")]
        t = np.array(m) / PUSH_RATE_HZ * 1e3
        ocp = [r["push"][4] for r in rows if r.get("push")]
        ocn = [r["nopush"][4] if r.get("nopush") else np.nan for r in rows if r.get("push")]
        symp = [r["push"][5] for r in rows if r.get("push")]
        axes[0].plot(t, ocp, "-o", color=colors[i], ms=4, label=f"{lbl}  PUSH")
        axes[0].plot(t, ocn, ":", color=colors[i], alpha=0.7, label=f"{lbl}  no-push control")
        axes[1].plot(t, np.array(ocp) - np.array(ocn), "-o", color=colors[i], ms=4, label=lbl)
        axes[2].plot(t, symp, "-o", color=colors[i], ms=4, label=lbl)
    axes[0].set_title("origin coherence")
    axes[1].set_title("push - no-push contrast  (>0 = the push adds a wave)")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[2].set_title("mirror symmetry (PUSH)")
    for ax in axes:
        ax.set_xlabel("time after R-peak [ms]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"In-vivo 2026-08-18, R-peak triggered: old vs recommended push settings "
                 f"({quantity})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print("wrote", outpath)


def side_by_side(all_rows, outpath, quantity, n_show=6):
    """The best pushes of each configuration, PUSH over NO-PUSH, on a common colour scale."""
    fig, axes = plt.subplots(4, n_show, figsize=(2.6 * n_show, 12.5), squeeze=False)
    unit = 1e3 if quantity == "velocity" else 1e6
    for i, (lbl, rows) in enumerate(all_rows):
        have = [r for r in rows if r.get("push") and r.get("nopush")]
        have.sort(key=lambda r: r["push"][4] - r["nopush"][4], reverse=True)
        sel = have[:n_show]
        clim = float(np.median([np.percentile(np.abs(r["push"][0]), 99) for r in rows
                                if r.get("push")])) * unit
        for j, r in enumerate(sel):
            for k, tag in enumerate(("push", "nopush")):
                ax = axes[2 * i + k][j]
                data, rr, tt, r0, oc, sym = r[tag]
                ax.imshow(data * unit,
                          extent=(rr[0] * 1e3, rr[-1] * 1e3, tt[-1] * 1e3, tt[0] * 1e3),
                          cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto", origin="upper")
                ax.axvline(r0 * 1e3, color="0.2", ls="--", lw=0.8, alpha=0.7)
                ax.tick_params(labelsize=6)
                ax.set_title(f"{'PUSH' if tag == 'push' else 'no-push'}  m{r['m']} "
                             f"(t={r['m'] / PUSH_RATE_HZ * 1e3:.0f} ms)\noc {oc:.2f} sym {sym:.2f}",
                             fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{lbl}\n{'PUSH' if tag == 'push' else 'control'}\nt [ms]",
                                  fontsize=8)
    fig.suptitle("Best pushes by push-minus-control contrast, each over its own no-push control "
                 "(same colour scale within a configuration)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(outpath, dpi=105)
    plt.close(fig)
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quantity", default="displacement")
    ap.add_argument("--recipe", default="invivo", choices=["invivo", "phantom"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rec = L.REC_INVIVO if a.recipe == "invivo" else L.REC_PHANTOM

    # Any subfolder with beamformed buffer-2 IQ. NOT L.measurement_folders(): the 61-element
    # acquisition has no runtime AcquisitionParametersAndECG.mat (it was rebuilt from the session
    # workspace dump), so requiring that file would silently drop it.
    folders = sorted(os.path.join(a.root, d) for d in os.listdir(a.root)
                     if os.path.isdir(os.path.join(a.root, d, "output"))
                     and L.n_pushes(os.path.join(a.root, d)) > 0)
    folders = sorted(folders, key=lambda f: "61elements" in f)      # 41 el first

    all_rows = []
    for folder in folders:
        lbl = label_of(folder)
        print(f"=== {lbl}", flush=True)
        rows = run_config(folder, rec, a.quantity)
        all_rows.append((lbl, rows))
        stem = os.path.basename(folder).split("_SW_")[0]
        montage(rows, "push", f"IN VIVO  {lbl}  -  ARF PUSH, {a.quantity}, {rec['tag']}\n"
                              "title: oc / sym of the push, ctrl oc = its own no-push control",
                os.path.join(a.outdir, f"invivo_{stem}_push.png"), a.quantity)
        montage(rows, "nopush", f"IN VIVO  {lbl}  -  NO-PUSH CONTROL (split pre-push reference), "
                                f"{a.quantity}, {rec['tag']}",
                os.path.join(a.outdir, f"invivo_{stem}_nopush.png"), a.quantity)

    summary(all_rows, os.path.join(a.outdir, f"invivo_summary_{a.quantity[:4]}.png"), a.quantity)
    if len(all_rows) == 2:
        side_by_side(all_rows, os.path.join(a.outdir, f"invivo_best_{a.quantity[:4]}.png"),
                     a.quantity)

    with open(os.path.join(a.outdir, f"invivo_scores_{a.quantity[:4]}.csv"), "w",
              encoding="utf-8") as f:
        f.write("config,push,t_after_Rpeak_ms,oc_push,sym_push,oc_nopush,sym_nopush,oc_contrast\n")
        for lbl, rows in all_rows:
            for r in rows:
                if not r.get("push"):
                    continue
                op, sp = r["push"][4], r["push"][5]
                on, sn = (r["nopush"][4], r["nopush"][5]) if r.get("nopush") else (np.nan, np.nan)
                f.write(f"{lbl},{r['m']},{r['m'] / PUSH_RATE_HZ * 1e3:.0f},"
                        f"{op:.3f},{sp:.3f},{on:.3f},{sn:.3f},{op - on:.3f}\n")
    print("wrote scores CSV")


if __name__ == "__main__":
    main()
