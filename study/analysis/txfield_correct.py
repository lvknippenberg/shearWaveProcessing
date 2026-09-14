"""Normalise the focused image by the synthesised harmonic transmit field, at varying strength.

The 2nd-harmonic complex field carries ripple at 1.260 deg against the image's 1.284 deg, so the
periodicity matches and the scalloping really is in the coherent transmit field. But the modelled
ripple is ~6x deeper than the image's, because squaring the linear field is a crude stand-in for
nonlinear harmonic generation - the true harmonic field is smoother.

So the correction is applied as ``image / (S/median(S))**gamma`` and gamma is swept. gamma=0 is
no correction; gamma=1 is full division by the model. The useful outcome is a gamma that flattens
the ripple while leaving the point-target PSF alone - the whole reason to prefer this over
REFoCUS, which cost 34% lateral resolution.

The map is clipped before dividing: its raw span is ~170 dB, and dividing by a near-null would
manufacture enormous values out of noise.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phantom_psf as P

OUT = os.path.dirname(os.path.abspath(__file__))
CLIP_DB = 12.0          # limit the correction to +/- this, so nulls cannot blow up

env, co = P.load_env(P.RECONS[0][0])
targets = P.find_targets(env, *P.axes_mm(co)[::-1][::-1])
S_raw = np.load(os.path.join(OUT, "txfield_harm.npy"))

ins = env > 0
S = S_raw / np.median(S_raw[ins])
lo, hi = 10 ** (-CLIP_DB / 20), 10 ** (CLIP_DB / 20)
S = np.clip(S, lo, hi)
print(f"correction map clipped to +/-{CLIP_DB:.0f} dB; "
      f"in-sector span now {20 * np.log10(S[ins].max() / S[ins].min()):.1f} dB")
print()


def report(e, lab):
    rows = P.measure(e, co, targets)
    if not rows:
        print(f"{lab:26s}  (no measurable targets)")
        return None
    lat = np.median([r["lat"] for r in rows])
    ax = np.median([r["ax"] for r in rows])
    cnr = np.median([r["cnr"] for r in rows])
    frac, rms, pk = P.angular_ripple(e, co)
    amp = np.sqrt(frac / 100.0) * rms
    print(f"{lab:26s} {lat:8.2f}mm {ax:7.2f}mm {cnr:6.1f}dB {amp:9.2f}% {pk:8.3f}deg")
    return lat, amp


print(f"{'correction':26s} {'lat -6dB':>9s} {'ax -6dB':>8s} {'CNR':>7s} {'ripple':>10s} {'peak':>9s}")
print("-" * 76)
base = report(env, "none (standard)")
best = None
for g in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    corrected = env / (S ** g)
    r = report(corrected, f"gamma = {g:.2f}")
    if r and (best is None or r[1] < best[1]):
        best = (g, *r)
        np.save(os.path.join(OUT, "env_txfield_corrected.npy"), corrected)

print()
if best and base:
    g, lat, amp = best
    print(f"best: gamma {g:.2f} -> ripple {base[1] / amp:.2f}x lower, "
          f"lateral PSF {lat / base[0]:.3f}x "
          f"({'unchanged' if abs(lat / base[0] - 1) < 0.05 else 'CHANGED'})")
    print("compare REFoCUS on the same phantom: 3.4x lower ripple, PSF 1.34x (34% worse)")
