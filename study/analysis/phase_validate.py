"""Synthetic validation of the PHASE-GRADIENT estimator, with speckle in the model.

The structure tensor passed its synthetic sweep and then failed on real data because the
generator had no speckle - the confound that actually dominates a real axial displacement field.
This sweep therefore runs speckle by default: a pass here is a meaningful gate, a pass without it
is not.

    python study/analysis/phase_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from swp.field.phase import aggregate_phase, phase_gradient_speed
from swp.field.structure import aggregate, estimate_field
from swp.field.synthetic import (add_noise, apply_pipeline_smoothing, apply_speckle,
                                 axial_clutter, coords, default_grid, plane_wave,
                                 speckle_amplitude)


def ang_err(est, true):
    return abs((est - true + 90.0) % 180.0 - 90.0)


def main():
    g = default_grid()
    ax = coords(g, wall_mm=12.0, length_mm=40.0, duration_ms=110.0)
    speeds = [1.5, 2.0, 3.0, 4.0, 6.0]
    thetas = [0.0, 15.0, 30.0, 45.0, 60.0]
    freqs = [12.0, 16.0, 22.0]

    print("Phase-gradient estimator, swept over c x theta x f, WITH speckle (contrast 1.0)\n")
    print(f"{'condition':<34}{'n':>4}{'|c err| med':>12}{'p90':>8}"
          f"{'dir med':>9}{'p90':>7}{'coh':>7}")
    print("-" * 82)

    for label, contrast, clutter, snr in (
            ("no speckle, SNR 15 dB", 0.0, 0.0, 15),
            ("speckle 0.5, SNR 15 dB", 0.5, 0.0, 15),
            ("speckle 1.0, SNR 15 dB", 1.0, 0.0, 15),
            ("speckle 1.0, SNR 6 dB", 1.0, 0.0, 6),
            ("speckle 1.0 + clutter x0.5", 1.0, 0.5, 15),
            ("speckle 1.0 + clutter x1.0", 1.0, 1.0, 15)):
        rows = []
        for c in speeds:
            for th in thetas:
                for fw in freqs:
                    fld, _ = plane_wave(g, c=c, theta_deg=th, f_hz=fw, axes=ax)
                    if contrast > 0:
                        fld = apply_speckle(fld, speckle_amplitude(g, ax, contrast=contrast))
                    if clutter > 0:
                        fld = axial_clutter(fld, ax, g, ratio=clutter, f_hz=8.0)
                    p = apply_pipeline_smoothing(add_noise(fld, snr, rng=0), g, band_hz=(5, 150))
                    s = aggregate_phase(phase_gradient_speed(p, g, f_hz=fw),
                                        min_coherence=0.2, min_weight_pct=60)
                    if not np.isfinite(s["speed"]):
                        continue
                    rows.append((abs(100 * (s["speed"] / c - 1)),
                                 ang_err(s["theta_deg"], th), s["coherence"]))
        a = np.array(rows)
        print(f"{label:<34}{len(a):>4}{np.median(a[:, 0]):>12.1f}"
              f"{np.percentile(a[:, 0], 90):>8.1f}{np.median(a[:, 1]):>9.1f}"
              f"{np.percentile(a[:, 1], 90):>7.1f}{np.median(a[:, 2]):>7.3f}")

    print("\nSide-by-side on the case that broke the structure tensor "
          "(c=3, theta=0, f=16, speckle 1.0):")
    fld, _ = plane_wave(g, c=3.0, theta_deg=0.0, f_hz=16.0, axes=ax)
    fld = apply_speckle(fld, speckle_amplitude(g, ax, contrast=1.0))
    p = apply_pipeline_smoothing(add_noise(fld, 15, rng=0), g, band_hz=(5, 150))
    st = aggregate(estimate_field(p, g), min_coherence=0.5)
    ph = aggregate_phase(phase_gradient_speed(p, g, f_hz=16.0))
    print(f"  structure tensor : c={st['speed']:5.2f}  theta={st['theta_deg']:+6.1f}")
    print(f"  phase gradient   : c={ph['speed']:5.2f}  theta={ph['theta_deg']:+6.1f}")
    print("  (truth: c=3.00, theta=+0.0)")


if __name__ == "__main__":
    sys.exit(main())
