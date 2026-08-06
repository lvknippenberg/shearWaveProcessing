"""Sanity check: does every GUI/registry option actually have an EFFECT on the space-time output
(and the metric)? Catches no-op / broken methods (a filter that silently returns its input, a param
with no reachable effect). For each stage, take a base output (stage = none / loupas / directional
off) and measure each alternative method's relative change ||out - base|| / ||base|| in the
displacement space-time, plus the change in push_specificity S. Anything ~0 is flagged NO-EFFECT.

    <zea-python> scripts/check_options.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))

import core                                              # noqa: E402
import registry as reg                                  # noqa: E402
from swp.viz.metrics import push_specificity            # noqa: E402

PHANTOM_25V = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom/DefaultPatient_SW_data_04-August-2026_13-28-32"
BASE = dict(mline_source="horizontal_push", estimator="loupas", mode="relative_to_reference",
            motion_steps=[("temporal_bandpass", {})], spatial_steps=[], temporal_steps=[],
            offsets=5, directional="outward")


def run(acq, ml, **over):
    rec = core.Recipe(**{**BASE, **over})
    rp = core.run_recipe(acq, ml, core.to_config(rec, acq))
    rn = core.run_recipe(acq, ml, core.to_config(rec, acq), nopush=True)
    return rp.st.data.copy(), push_specificity(rp, rn)["S"]


def reldiff(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-20))


def default_params(method):
    return reg.step_params(method, {})


def main():
    r0 = core.Recipe(mline_source="horizontal_push")
    acq = core.load_acq(PHANTOM_25V, 0, r0)
    ml = core.load_mline_for(PHANTOM_25V, 0, acq, r0)

    flagged = []
    print(f"{'stage / method':42s}{'rel.change':>12}{'dS':>9}   effect?")
    print("-" * 78)

    def check(stage_name, base_over, alts, label_over):
        base_data, base_S = run(acq, ml, **base_over)
        for name in alts:
            over = label_over(name)
            try:
                d, S = run(acq, ml, **over)
            except Exception as exc:  # noqa: BLE001
                print(f"  {stage_name}:{name:32s}{'ERROR':>12}   {type(exc).__name__}")
                flagged.append((stage_name, name, "error"))
                continue
            rc = reldiff(d, base_data)
            eff = "yes" if rc > 1e-6 else "NO-EFFECT"
            if rc <= 1e-6:
                flagged.append((stage_name, name, "no-effect"))
            print(f"  {stage_name}:{str(name):32s}{rc:>12.2e}{S-base_S:>9.2f}   {eff}")

    # IQ pre-filter: base = no IQ filter
    check("iq", dict(iq_steps=[]),
          [m.name for m in reg.IQ_METHODS if m.name != "none"],
          lambda n: dict(iq_steps=[(n, {})]))
    # estimator: base = loupas
    check("est", dict(estimator="loupas"),
          [m.name for m in reg.ESTIMATOR_METHODS if m.name not in ("loupas", "rf_ncc")],
          lambda n: dict(estimator=n))
    # mode
    check("mode", dict(mode="relative_to_reference"), ["frame_to_frame"], lambda n: dict(mode=n))
    # motion removal: base = none
    check("motion", dict(motion_steps=[]),
          [m.name for m in reg.MOTION_METHODS if m.name != "none"],
          lambda n: dict(motion_steps=[(n, {})]))
    # spatial: base = none
    check("spatial", dict(spatial_steps=[]),
          [m.name for m in reg.SPATIAL_METHODS if m.name != "none"],
          lambda n: dict(spatial_steps=[(n, {})]))
    # temporal: base = none
    check("temporal", dict(temporal_steps=[]),
          [m.name for m in reg.TEMPORAL_METHODS if m.name != "none"],
          lambda n: dict(temporal_steps=[(n, {})]))
    # directional: base = off
    check("directional", dict(directional="none"),
          ["outward", "leftward", "rightward"], lambda n: dict(directional=n))
    # M-line offsets: base = 1 line
    check("offsets", dict(offsets=1), [3, 7], lambda n: dict(offsets=n))
    # mline agg (needs >1 offset to matter): base mean @7
    check("mline_agg@7", dict(offsets=7, mline_agg="mean"), ["median"],
          lambda n: dict(offsets=7, mline_agg=n))

    print("-" * 78)
    if flagged:
        print("FLAGGED (no measurable effect on the space-time, or error):")
        for s, n, why in flagged:
            print(f"  - {s}:{n}  ({why})")
    else:
        print("All options have a measurable effect. ✔")


if __name__ == "__main__":
    main()
