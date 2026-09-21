"""Round 5: what the masks actually cost, measured paired per wire target, and the runtime.

Comparing MEDIAN -6 dB widths across reconstructions hides the effect in target-to-target
scatter (the 66 wires span 25-110 mm and 1.2-2.6 mm).  Pairing - measuring the SAME wire in
both reconstructions and taking the median of the per-target DIFFERENCE - removes that
scatter and is what settles a few-percent resolution change.
"""
from __future__ import annotations

import time

import numpy as np

import b1_lib as L
import b1_masks as M
import b1_metrics as Q
import b1_pfield as P

TAG = "phantom"


def paired_psf():
    stack, meta = L.build_stack(TAG)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    base = np.abs(M.compose(stack, M.w_all(geom))).mean(axis=0)
    targets = Q.find_targets(base, coords)
    base_rows = {(iz, ix): r for (iz, ix), r in
                 zip(targets, Q.measure_targets(base, coords, targets))} \
        if len(Q.measure_targets(base, coords, targets)) == len(targets) else None

    # measure_targets drops targets whose peak does not close; re-measure keeping the key
    def rows_by_target(env):
        out = {}
        x, z = Q.axes_mm(coords)
        dx, dz = abs(x[1] - x[0]), abs(z[1] - z[0])
        for iz, ix in targets:
            lat = Q.fwhm(env[iz], dx, ix)
            ax = Q.fwhm(env[:, ix], dz, iz)
            if lat is None or ax is None:
                continue
            z0, z1 = max(iz - 40, 0), min(iz + 40, env.shape[0])
            x0, x1 = max(ix - 40, 0), min(ix + 40, env.shape[1])
            out[(iz, ix)] = (lat, ax,
                             20 * np.log10(env[iz, ix] /
                                           (np.median(env[z0:z1, x0:x1]) + 1e-20)),
                             z[iz])
        return out

    ref = rows_by_target(base)
    pfr = P.pfield_map(TAG, norm=False)
    sets = [("cone rect x0.75", M.w_cone(geom, "rect", 0.75)),
            ("cone rect x1", M.w_cone(geom, "rect", 1.0)),
            ("cone rect x1.5", M.w_cone(geom, "rect", 1.5)),
            ("cone rect x2", M.w_cone(geom, "rect", 2.0)),
            ("cone rect x3", M.w_cone(geom, "rect", 3.0)),
            ("cone tukey50 x1.5", M.w_cone(geom, "tukey50", 1.5)),
            ("cone hann x2", M.w_cone(geom, "hann", 2.0)),
            ("cone x1.5 * pfield", M.w_cone(geom, "rect", 1.5) * M.w_pfield(geom, pfr)),
            ("nearest-3", M.w_nearest_k(geom, 3)),
            ("nearest-5", M.w_nearest_k(geom, 5))]

    print(f"baseline: {len(ref)} measurable wires, median lateral "
          f"{np.median([v[0] for v in ref.values()]):.2f} mm, axial "
          f"{np.median([v[1] for v in ref.values()]):.2f} mm\n")
    rng = np.random.default_rng(0)

    def boot_ci(d, n=4000):
        """Bootstrap 95% CI on the median of the paired differences."""
        bs = np.array([np.median(rng.choice(d, d.size, replace=True)) for _ in range(n)])
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"{'method':22s} {'d_lat %':>10s} {'95% CI':>16s} {'d_ax %':>9s} {'d_wCNR dB':>10s} "
          f"{'n':>4s} {'shallow':>9s} {'deep':>7s}")
    print("-" * 92)
    for lab, w in sets:
        env = np.abs(M.compose(stack, w)).mean(axis=0)
        cur = rows_by_target(env)
        keys = [k for k in ref if k in cur]
        dlat = np.array([cur[k][0] / ref[k][0] - 1 for k in keys]) * 100
        dax = np.array([cur[k][1] / ref[k][1] - 1 for k in keys]) * 100
        dcnr = np.array([cur[k][2] - ref[k][2] for k in keys])
        zs = np.array([ref[k][3] for k in keys])
        sh = np.median(dlat[zs < 70]) if (zs < 70).any() else np.nan
        dp = np.median(dlat[zs >= 70]) if (zs >= 70).any() else np.nan
        lo, hi = boot_ci(dlat)
        print(f"{lab:22s} {np.median(dlat):+9.2f}% [{lo:+6.2f},{hi:+6.2f}] "
              f"{np.median(dax):+8.1f}% {np.median(dcnr):+9.2f} {len(keys):4d} "
              f"{sh:+8.1f}% {dp:+6.1f}%")


def timing():
    """Cost of the mask in the real pipeline, on the real in-vivo workload."""
    import zea
    from zea import File, init_device
    from zea.ops import Beamform, Cast, Demodulate
    from swp.acquisition.beamform import _ensure_cpu_t_peak, apply_grid, plan_patches_and_chunk
    from swp.acquisition.sequence import read_swi_meta

    init_device(verbose=False)
    root = L.DATASETS["invivo"]["root"]
    swi = read_swi_meta(root / "CombinedData.mat")
    with File(str(root / "output" / "converted" / "CombinedData_buffer1.hdf5")) as fh:
        raw = np.asarray(fh.data.raw_data[:12])
        params = fh.load_parameters()
    apply_grid(params, swi.grids[0])
    _ensure_cpu_t_peak(params)
    coords = np.asarray(params.grid, np.float32)
    _, meta = L.build_stack("invivo")
    geom = L.geometry_from_meta(meta)
    nz, nx = coords.shape[:2]
    n_tx, n_el = raw.shape[1], raw.shape[3]
    flat = np.moveaxis(M.w_cone(geom, "rect", 1.5), 0, -1).reshape(nz * nx, n_tx).astype(
        np.float32)
    num_patches, _ = plan_patches_and_chunk(params, n_tx, n_el)

    def build(mask):
        pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                             Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                      enable_aligned_apodization=mask)],
                            with_batch_dim=True, jit_options=None)
        bf = pipe.prepare_parameters(params)
        if mask:
            bf["flat_aligned_apodization"] = flat
        return pipe, bf

    print(f"\ntiming: {raw.shape[0]} frames x {n_tx} tx onto {nz}x{nx}")
    from swp.acquisition.beamform import free_gpu_memory
    for lab, mask in (("plain DAS (pipeline)", False), ("cone-masked DAS", True)):
        free_gpu_memory()
        pipe, bf = build(mask)
        pipe(data=raw[:2], **bf)                        # warm-up, untimed
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            pipe(data=raw, **bf)
            ts.append(time.perf_counter() - t0)
        print(f"  {lab:22s} {np.median(ts):6.2f} s  ({np.median(ts) / raw.shape[0] * 1e3:.0f} "
              f"ms/frame)   [{min(ts):.2f}-{max(ts):.2f}]")


if __name__ == "__main__":
    paired_psf()
    timing()
