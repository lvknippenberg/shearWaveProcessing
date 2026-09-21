"""Full evaluation of the buffer-1 pixel-inclusion rules.

    python b1_eval.py phantom
    python b1_eval.py invivo --norm none rms

Reports, per reconstruction:

  cov      median effective transmits contributing per in-sector pixel, (sum w)^2 / sum w^2
  fov      % of the sector above -50 dB of its own 99.9th percentile (field of view)
  edge     mean envelope at |angle| 30-40 deg / mean at |angle| < 15 deg, in dB, relative to
           the baseline: how much sector periphery the rule gives away
  dr       dynamic range, 99.9th / 1st percentile in-sector, dB
  spk      speckle SNR (mean/std) in the uniform speckle ROI; 1.91 = fully developed speckle
  gCNR     generalised CNR between the dark and tissue ROIs (invariant to gain / log / gamma)
  C        plain envelope contrast between the same ROIs, dB
  lat/ax   -6 dB wire-target widths, mm (phantom only)
  wCNR     wire target over local background, dB (phantom only)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_methods as MT
import b1_metrics as Q
import b1_render as R
import b1_rois as RO


def roi_pairs(tag):
    if tag == "invivo":
        return RO.INVIVO, [("dark_mid", "tissue_deep"), ("dark_upper", "tissue_near")], "speckle"
    return RO.PHANTOM, [("lesion_bg", "lesion")], "speckle"


def edge_bands(coords):
    x, z = Q.axes_mm(coords)
    X, Z = np.meshgrid(x, z)
    ang = np.abs(np.degrees(np.arctan2(X, Z)))
    deep = Z > 20
    return (ang > 30) & (ang <= 39) & deep, (ang < 15) & deep


def evaluate(tag, norms=("none",), include=None, frame=None, ncols=4, out_tag="sweep"):
    stack, meta = L.build_stack(tag)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    print(geom.summary())
    rois, pairs, spk_name = roi_pairs(tag)
    rmask = {nm: Q.roi_mask(coords, xl, zl) for nm, xl, zl, _ in rois}
    sector = Q.sector_mask(coords)
    e_edge, e_ctr = edge_bands(coords)
    frame = RO.INVIVO_FRAME if (frame is None and tag == "invivo") else frame

    def pick(env):
        """The single image the ROI metrics are read off."""
        return env[frame] if frame is not None else env.mean(axis=0)

    base_env = np.abs(M.compose(stack, M.w_all(geom)))
    base_img = pick(base_env)
    base_edge = 20 * np.log10(base_img[e_edge].mean() / base_img[e_ctr].mean())
    targets = Q.find_targets(base_img, coords) if tag == "phantom" else []
    if targets:
        print(f"{len(targets)} wire targets on the baseline")

    cat = MT.catalogue(geom, include)
    rows, panels = [], []
    t0 = time.perf_counter()
    for label, (w, _) in cat.items():
        for norm in norms:
            lab = label if norm == "none" else f"{label} /{norm}"
            env = np.abs(M.compose(stack, w, norm=norm))
            img = pick(env)
            cov = M.coverage(w)
            g = [Q.gcnr(img[rmask[a]], img[rmask[b]]) for a, b in pairs]
            c = [abs(Q.contrast_db(img[rmask[b]], img[rmask[a]])) for a, b in pairs]
            row = dict(method=lab,
                       cov=float(np.median(cov[sector])),
                       fov=Q.sector_coverage(img, coords),
                       edge=float(20 * np.log10(img[e_edge].mean() / img[e_ctr].mean())
                                  - base_edge),
                       dr=Q.dynamic_range_db(img, sector),
                       spk=Q.speckle_snr(img[rmask[spk_name]]),
                       gcnr=float(np.mean(g)), contrast=float(np.mean(c)))
            if targets:
                row.update(Q.target_summary(Q.measure_targets(img, coords, targets)))
            rows.append(row)
            panels.append((lab, img))
    print(f"[{tag}] {len(rows)} reconstructions in {time.perf_counter() - t0:.0f}s\n")

    hdr = (f"{'method':30s} {'cov':>5s} {'fov%':>6s} {'edge':>6s} {'dr':>6s} {'spk':>5s} "
           f"{'gCNR':>6s} {'C/dB':>6s}")
    if targets:
        hdr += f" {'lat':>6s} {'ax':>5s} {'wCNR':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['method']:30s} {r['cov']:5.2f} {r['fov']:6.1f} {r['edge']:+6.2f} "
                f"{r['dr']:6.1f} {r['spk']:5.2f} {r['gcnr']:6.3f} {r['contrast']:6.2f}")
        if targets:
            line += f" {r.get('lat', np.nan):6.2f} {r.get('ax', np.nan):5.2f} {r.get('cnr', np.nan):6.1f}"
        print(line)

    R.montage(panels, coords, L.FIGS / f"{tag}_{out_tag}_matched.png", ncols=ncols,
              level_mode="matched", baseline_env=base_img,
              title=f"{tag} buffer 1 - shared levels, tissue medians matched")
    R.montage(panels, coords, L.FIGS / f"{tag}_{out_tag}_own.png", ncols=ncols,
              level_mode="own", title=f"{tag} buffer 1 - each panel auto-gained (as shipped)")
    with open(L.HERE / f"results_{tag}_{out_tag}.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag", choices=list(L.DATASETS))
    p.add_argument("--norm", nargs="+", default=["none"])
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--out", default="sweep")
    a = p.parse_args()
    evaluate(a.tag, a.norm, a.include, a.frame, a.ncols, a.out)
