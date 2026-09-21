"""Stage 1 of the field-estimator plan: does the structure tensor recover a wave it was GIVEN?

Runs the pre-registered synthetic tests of ``docs/field_estimator_plan.md`` and prints a verdict
against the exit criteria. Nothing here touches real data: if the estimator cannot pass this, it
will not work on the septum, and the day spent finding that out is the point of the stage.

    python study/analysis/field_validate.py
    python study/analysis/field_validate.py --quick --figure out.png

Exit criteria (V1):
    * plane-wave sweep at realistic SNR: speed within 10 %, direction within 10 degrees
    * two crossing waves: coherence must drop clearly below the single-wave case
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from swp.field.structure import aggregate, estimate_field, interference_indicator
from swp.field.synthetic import (add_bulk_motion, add_noise, apply_pipeline_smoothing,
                                 coords, curved_wave, default_grid, dispersive_packet,
                                 plane_wave, two_waves)

BAND = (5.0, 150.0)


def run_case(field, grid, min_coherence=0.5):
    est = estimate_field(field, grid)
    return aggregate(est, min_coherence=min_coherence), est


def process(field, grid, snr_db=None, bulk=None, axes=None, smooth=True, rng=0):
    """Apply the contaminations and the pipeline's own smoothing, in acquisition order."""
    out = field
    if bulk is not None:
        out = add_bulk_motion(out, axes, amplitude_ratio=bulk, f_hz=3.0, tilt_per_mm=0.004)
    if snr_db is not None:
        out = add_noise(out, snr_db, rng=rng)
    if smooth:
        out = apply_pipeline_smoothing(out, grid, band_hz=BAND)
    return out


def ang_err(est_deg, true_deg):
    """Smallest angle between two directions, modulo 180 deg (a wave and its reverse agree)."""
    d = (est_deg - true_deg + 90.0) % 180.0 - 90.0
    return abs(d)


def sweep_plane(grid, axes, speeds, thetas, freqs, snr_db, bulk=None, label=""):
    rows = []
    for c in speeds:
        for th in thetas:
            for f in freqs:
                fld, _ = plane_wave(grid, c=c, theta_deg=th, f_hz=f, axes=axes)
                fld = process(fld, grid, snr_db=snr_db, bulk=bulk, axes=axes)
                s, _ = run_case(fld, grid)
                if not np.isfinite(s["speed"]):
                    rows.append((c, th, f, np.nan, np.nan, np.nan, np.nan))
                    continue
                rows.append((c, th, f, s["speed"], 100.0 * (s["speed"] / c - 1.0),
                             ang_err(s["theta_deg"], th), s["coherence"]))
    a = np.array(rows, float)
    ok = np.isfinite(a[:, 4])
    err, aerr = np.abs(a[ok, 4]), a[ok, 5]
    print(f"  {label:<34} n={ok.sum():3d}  |speed err| median {np.median(err):5.1f} %  "
          f"p90 {np.percentile(err, 90):5.1f} %  |  dir err median {np.median(aerr):4.1f} deg  "
          f"p90 {np.percentile(aerr, 90):4.1f} deg  |  coh {np.median(a[ok, 6]):.2f}")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--wall-mm", type=float, default=12.0)
    ap.add_argument("--length-mm", type=float, default=40.0)
    ap.add_argument("--figure", default=str(Path(__file__).with_name("field_validate.png")))
    a = ap.parse_args()

    grid = default_grid()
    axes = coords(grid, wall_mm=a.wall_mm, length_mm=a.length_mm, duration_ms=110.0)
    nt, nz, nx = (len(v) for v in axes)
    print(f"grid {nt} x {nz} x {nx}  (dt {grid.dt_ms:.2f} ms, dz {grid.dz_mm:.3f} mm, "
          f"dx {grid.dx_mm:.3f} mm)   ROI {a.wall_mm:.0f} x {a.length_mm:.0f} mm, 110 ms\n")

    speeds = [2.0, 3.0, 5.0] if a.quick else [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    thetas = [0.0, 30.0, 60.0] if a.quick else [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
    freqs = [16.0] if a.quick else [10.0, 16.0, 25.0, 35.0]

    print("=== V1a: plane wave, no contamination ===")
    clean = sweep_plane(grid, axes, speeds, thetas, freqs, snr_db=None, label="noiseless")

    print("\n=== V1b: plane wave vs SNR (with pipeline smoothing) ===")
    per_snr = {}
    for snr in ([20, 10, 0] if a.quick else [30, 20, 15, 10, 6, 3, 0]):
        per_snr[snr] = sweep_plane(grid, axes, speeds, thetas, freqs, snr_db=snr,
                                   label=f"SNR {snr:+3d} dB")

    print("\n=== V1c: plane wave + bulk motion (uniform 3 Hz, plus a slight spatial tilt) ===")
    for bulk in (1.0, 3.0):
        sweep_plane(grid, axes, speeds, thetas, freqs, snr_db=15, bulk=bulk,
                    label=f"bulk x{bulk:.0f} of signal RMS, SNR 15 dB")

    print("\n=== V1d: curved wavefront (point source outside the ROI) ===")
    for src in ((-25.0, 6.0), (-60.0, 6.0)):
        fld, _ = curved_wave(grid, c=3.0, source_mm=src, f_hz=16.0, axes=axes)
        fld = process(fld, grid, snr_db=15, axes=axes)
        s, est = run_case(fld, grid)
        print(f"  source at {src[0]:+5.1f} mm: c={s['speed']:5.2f} "
              f"({100 * (s['speed'] / 3.0 - 1):+5.1f}%)  direction spread "
              f"{s['direction_spread_deg']:4.1f} deg  coh {s['coherence']:.2f}")
    print("    (a curved front SHOULD show a direction spread; a plane one should not)")

    print("\n=== V1e: dispersive packet (c ~ sqrt(f), Lamb A0-like) ===")
    for power in (0.0, 0.5):
        fld, _, c_ref = dispersive_packet(grid, c_ref=3.0, f_ref=16.0, power=power, axes=axes)
        fld = process(fld, grid, snr_db=15, axes=axes)
        s, _ = run_case(fld, grid)
        print(f"  power={power:.1f} (c(16 Hz)={c_ref:.1f} m/s): c={s['speed']:5.2f} "
              f"({100 * (s['speed'] / c_ref - 1):+5.1f}% vs the phase speed at 16 Hz)  "
              f"coh {s['coherence']:.2f}")

    print("\n=== V1f: NEGATIVE CONTROL - two crossing waves ===")
    single, _ = plane_wave(grid, c=3.0, theta_deg=0.0, f_hz=16.0, axes=axes)
    s_single, _ = run_case(process(single, grid, snr_db=15, axes=axes), grid)
    print(f"  one wave        : coh {s_single['coherence']:.3f}  c={s_single['speed']:5.2f}  "
          f"kept {100 * s_single['frac_kept']:4.1f}%")
    cross_coh, env_single = [], interference_indicator(
        process(single, grid, snr_db=15, axes=axes))
    print(f"    envelope modulation (interference indicator): {env_single:.3f}")
    env_cross = []
    for th2 in (30.0, 50.0, 70.0, 90.0):
        fld, _ = two_waves(grid, c=(3.0, 5.0), theta_deg=(0.0, th2), axes=axes)
        s, _ = run_case(process(fld, grid, snr_db=15, axes=axes), grid)
        cross_coh.append(s["coherence"])
        ev = interference_indicator(process(fld, grid, snr_db=15, axes=axes))
        env_cross.append(ev)
        print(f"  two waves, {th2:4.0f} deg apart: coh {s['coherence']:.3f}  "
              f"c={s['speed']:5.2f}  kept {100 * s['frac_kept']:4.1f}%  "
              f"dir spread {s['direction_spread_deg']:4.1f} deg  env {ev:.3f}")
    pure_noise = np.random.default_rng(0).normal(size=(nt, nz, nx))
    s_noise, _ = run_case(apply_pipeline_smoothing(pure_noise, grid, band_hz=BAND), grid)
    print(f"  pure noise      : coh {s_noise['coherence']:.3f}  kept "
          f"{100 * s_noise['frac_kept']:4.1f}%")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT against the pre-registered Stage 1 criteria")
    print("=" * 78)
    ok_clean = np.isfinite(clean[:, 4])
    c_err, c_ang = np.abs(clean[ok_clean, 4]), clean[ok_clean, 5]
    p1 = np.percentile(c_err, 90) <= 10.0 and np.percentile(c_ang, 90) <= 10.0
    print(f"  noiseless plane wave within 10% / 10 deg (p90): "
          f"{np.percentile(c_err, 90):.1f}% / {np.percentile(c_ang, 90):.1f} deg  "
          f"-> {'PASS' if p1 else 'FAIL'}")
    worst_ok = None
    for snr in sorted(per_snr):
        arr = per_snr[snr]
        m = np.isfinite(arr[:, 4])
        if m.sum() and np.percentile(np.abs(arr[m, 4]), 90) <= 10.0 \
                and np.percentile(arr[m, 5], 90) <= 10.0:
            worst_ok = snr if worst_ok is None else min(worst_ok, snr)
    print(f"  lowest SNR still meeting 10% / 10 deg at p90: "
          + (f"{worst_ok} dB  -> PASS" if worst_ok is not None else "none  -> FAIL"))
    drop = s_single["coherence"] - max(cross_coh)
    p3 = drop > 0.05
    print(f"  coherence falls on crossing waves: {s_single['coherence']:.3f} -> "
          f"{max(cross_coh):.3f} (drop {drop:.3f})  -> {'PASS' if p3 else 'FAIL'}")
    print(f"  coherence on pure noise: {s_noise['coherence']:.3f} "
          f"(should be well below {s_single['coherence']:.2f})")
    ratio = min(env_cross) / env_single if env_single > 0 else float("nan")
    print(f"  envelope modulation separates them instead: {env_single:.3f} -> "
          f"{min(env_cross):.3f}-{max(env_cross):.3f} ({ratio:.1f}x)  "
          f"-> {'candidate' if ratio > 2 else 'no'}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axs[0]
        for th in sorted(set(clean[:, 1])):
            m = (clean[:, 1] == th) & ok_clean
            ax.plot(clean[m, 0], clean[m, 4], "o-", ms=4, label=f"{th:.0f} deg")
        ax.axhspan(-10, 10, color="0.85", zorder=0)
        ax.set_xlabel("true speed [m/s]"); ax.set_ylabel("speed error [%]")
        ax.set_title("noiseless, by direction"); ax.legend(fontsize=7, ncol=2)
        ax = axs[1]
        snrs = sorted(per_snr)
        med = [np.nanmedian(np.abs(per_snr[s][:, 4])) for s in snrs]
        p90 = [np.nanpercentile(np.abs(per_snr[s][:, 4]), 90) for s in snrs]
        ax.plot(snrs, med, "o-", label="median"); ax.plot(snrs, p90, "s--", label="p90")
        ax.axhline(10, color="crimson", ls=":"); ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("|speed error| [%]"); ax.set_yscale("log")
        ax.set_title("robustness to noise"); ax.legend(fontsize=8)
        ax = axs[2]
        ax.bar(["1 wave", "2 waves\n30 deg", "50", "70", "90", "noise"],
               [s_single["coherence"]] + cross_coh + [s_noise["coherence"]],
               color=["tab:green"] + ["tab:orange"] * 4 + ["0.6"])
        ax.set_ylabel("coherence"); ax.set_title("does the gate separate them?")
        fig.tight_layout(); fig.savefig(a.figure, dpi=110)
        print(f"\n-> {a.figure}")
    except Exception as e:                                           # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    sys.exit(main())
