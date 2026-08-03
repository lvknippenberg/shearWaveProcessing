"""Report the ARF push voltage actually delivered in each measurement of a sweep.

Run this right after acquiring a voltage sweep (before spending time on beamforming): it
reads the delivered push voltage from every measurement folder and flags any that sat at the
TPC profile's voltage ceiling -- i.e. where the requested voltage was silently clamped. If the
delivered voltages do not actually increase across the sweep, raise the push TPC profile's
``highVoltageLimit`` / ``maxHighVoltage`` and re-acquire.

Usage:
  python scripts/check_push_voltage.py --root "<Phantom parent folder>"
  python scripts/check_push_voltage.py --folder "<single measurement folder>"
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from swp.acquisition import discover_measurements, read_push_voltage   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", help="parent folder of measurement subfolders (a whole sweep)")
    g.add_argument("--folder", help="a single measurement folder")
    p.add_argument("--profile", type=int, default=5, help="push TPC profile index (default 5)")
    a = p.parse_args()

    if a.folder:
        rows = [(a.folder, read_push_voltage(a.folder, a.profile))]
    else:
        rows = discover_measurements(a.root, a.profile)
    if not rows:
        raise SystemExit(f"no measurement folders with a runtime .mat found under {a.root!r}")

    print(f"{'measurement':40s} {'push V':>7} {'limit':>7} {'maxHV':>7}  note")
    print("-" * 78)
    hvs, any_cap = [], False
    for folder, pv in rows:
        name = os.path.basename(os.path.normpath(folder))[:40]
        hv = "--" if pv.hv is None else f"{pv.hv:.1f}"
        lim = "--" if pv.high_voltage_limit is None else f"{pv.high_voltage_limit:.0f}"
        mx = "--" if pv.max_high_voltage is None else f"{pv.max_high_voltage:.0f}"
        note = "AT PROFILE MAX (possible clamp)" if pv.at_profile_max else ""
        any_cap = any_cap or pv.at_profile_max
        if pv.hv is not None:
            hvs.append(pv.hv)
        print(f"{name:40s} {hv:>7} {lim:>7} {mx:>7}  {note}")

    print("-" * 78)
    if len(hvs) >= 2:
        distinct = sorted(set(round(v, 1) for v in hvs))
        increasing = all(b >= a_ - 0.5 for a_, b in zip(hvs, hvs[1:]))
        print(f"delivered voltages: {[f'{v:.0f}' for v in hvs]}")
        print(f"distinct levels   : {distinct}")
        if len(distinct) < len(hvs):
            print("WARNING: fewer distinct voltages than measurements -- the sweep has repeats.")
        if any_cap:
            print("WARNING: one or more measurements sit at the profile's voltage ceiling; the "
                  "requested sweep was likely clamped. Raise the push TPC profile's "
                  "highVoltageLimit/maxHighVoltage and re-acquire.")
        if not any_cap and len(distinct) == len(hvs) and increasing:
            print("OK: delivered voltages are distinct and monotonic -- the sweep looks valid.")


if __name__ == "__main__":
    main()
