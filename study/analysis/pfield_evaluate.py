"""Evaluate pfield normalisation: does it flatten the ripple WITHOUT costing resolution?

The whole appeal over REFoCUS is that dividing by a smooth per-pixel sensitivity map should not
change the PSF, whereas REFoCUS measured 34% wider laterally on this phantom. So both numbers
have to be reported together - a ripple reduction that also widens the PSF would just be REFoCUS
by another route.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phantom_psf as P

OUT = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(OUT, "pfield_sensitivity.npy"))

env_std, co = P.load_env(P.RECONS[0][0])
print(f"envelope {env_std.shape}, sensitivity {S.shape}")
assert S.shape == env_std.shape, "sensitivity map and image grid disagree"

# Guard the division: where the transmit field is genuinely tiny, normalising only amplifies
# noise. Floor the map at a low percentile of its in-sector values.
ins = S[env_std > 0]
floor = np.percentile(ins, 5)
S_safe = np.maximum(S, floor)
print(f"sensitivity floored at the 5th pct ({floor:.3f}); "
      f"{100 * (S < floor).mean():.1f}% of pixels clamped")

env_corr = env_std / S_safe

targets = P.find_targets(env_std, *P.axes_mm(co)[::-1][::-1])
print(f"targets: {len(targets)}")

print()
print(f"{'reconstruction':34s} {'lat -6dB':>9s} {'ax -6dB':>8s} {'CNR':>7s} "
      f"{'ripple@line':>12s} {'peak':>9s}")
print("-" * 88)


def report(env, lab):
    rows = P.measure(env, co, targets)
    lat = np.median([r["lat"] for r in rows])
    ax = np.median([r["ax"] for r in rows])
    cnr = np.median([r["cnr"] for r in rows])
    frac, rms, pk = P.angular_ripple(env, co)
    absamp = np.sqrt(frac / 100.0) * rms
    print(f"{lab:34s} {lat:8.2f}mm {ax:7.2f}mm {cnr:6.1f}dB {absamp:11.2f}% {pk:8.3f}deg")
    return lat, absamp


base_lat, base_rip = report(env_std, "standard (coherent all-73)")
corr_lat, corr_rip = report(env_corr, "standard / pfield  (NEW)")
for name, lab in P.RECONS[1:]:
    try:
        e, c = P.load_env(name)
    except (OSError, KeyError):
        continue
    report(e, lab)

print()
print(f"pfield normalisation vs standard:  ripple {base_rip / max(corr_rip, 1e-9):.2f}x lower, "
      f"lateral PSF {corr_lat / base_lat:.3f}x "
      f"({'unchanged' if abs(corr_lat / base_lat - 1) < 0.05 else 'CHANGED'})")
np.save(os.path.join(OUT, "env_pfield_corrected.npy"), env_corr)
