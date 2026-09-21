"""Sweep the buffer-1 per-transmit pixel-inclusion rules and report metrics + montages.

    python b1_compare.py phantom [--norm none sum rms] [--include cone nearest]
    python b1_compare.py invivo
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q
import b1_methods as MT
import b1_render as R


def run(tag, norms, include, frame=0, ncols=4, out_tag=""):
    stack, meta = L.build_stack(tag)
    coords = meta["coords"]
    geom = L.geometry_from_meta(meta)
    print(geom.summary())
    n_fr = stack.shape[1]

    cat = MT.catalogue(geom, include)
    sector = Q.sector_mask(coords)

    # Reference (baseline) quantities, computed once.
    base_env = np.abs(M.compose(stack, M.w_all(geom)))
    targets = Q.find_targets(base_env.mean(axis=0), coords) if tag == "phantom" else []
    if targets:
        print(f"{len(targets)} wire targets found on the baseline")

    rows, panels = [], []
    t0 = time.perf_counter()
    for label, (w, _) in cat.items():
        for norm in norms:
            lab = label if norm == "none" else f"{label} /{norm}"
            env = np.abs(M.compose(stack, w, norm=norm))
            cov = M.coverage(w)
            mean_env = env.mean(axis=0)
            row = dict(
                method=lab,
                cov_med=float(np.median(cov[sector])),
                cov_min=float(np.percentile(cov[sector], 5)),
                fov=Q.sector_coverage(mean_env, coords),
                dr=Q.dynamic_range_db(mean_env, sector),
                spk=float(np.mean([Q.speckle_snr(env[k][sector]) for k in range(n_fr)])),
            )
            if targets:
                row.update(Q.target_summary(Q.measure_targets(mean_env, coords, targets)))
            rows.append(row)
            panels.append((lab, env[frame] if tag != "phantom" else mean_env))
            print(f"  {lab:34s} cov {row['cov_med']:5.2f}  fov {row['fov']:5.1f}%  "
                  f"dr {row['dr']:5.1f}dB  spk {row['spk']:4.2f}"
                  + (f"  lat {row.get('lat', float('nan')):5.2f}mm "
                     f"ax {row.get('ax', float('nan')):4.2f}mm "
                     f"cnr {row.get('cnr', float('nan')):5.1f}dB n={row.get('n', 0)}"
                     if targets else ""))
    print(f"[{tag}] {len(rows)} reconstructions in {time.perf_counter() - t0:.0f}s")

    suffix = out_tag or "sweep"
    R.montage(panels, coords, L.FIGS / f"{tag}_{suffix}_shared.png", ncols=ncols,
              level_mode="shared", baseline_env=base_env.mean(axis=0),
              title=f"{tag} buffer 1 - shared display levels (baseline-derived)")
    R.montage(panels, coords, L.FIGS / f"{tag}_{suffix}_own.png", ncols=ncols,
              level_mode="own",
              title=f"{tag} buffer 1 - each panel with its own adaptive levels")
    with open(L.HERE / f"results_{tag}_{suffix}.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag", choices=list(L.DATASETS))
    p.add_argument("--norm", nargs="+", default=["none"])
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--out", default="")
    a = p.parse_args()
    run(a.tag, a.norm, a.include, a.frame, a.ncols, a.out)
