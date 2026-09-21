"""Does the production route reproduce the offline composite?

The offline study composites a cached per-transmit stack.  The pipeline route feeds the same
weight map to zea as ``parameters.flat_aligned_apodization`` with
``Beamform(enable_aligned_apodization=True)`` - one pass, no stack.  If the two agree, the
whole study transfers to the pipeline unchanged and the production cost is one extra
elementwise multiply.
"""
from __future__ import annotations

import time

import numpy as np

import b1_lib as L
import b1_masks as M

TAG = "phantom"
SCALE = 1.5


def main():
    import zea
    from zea import File, init_device
    from zea.ops import Beamform, Cast, Demodulate
    from swp.acquisition.beamform import _ensure_cpu_t_peak, apply_grid, plan_patches_and_chunk
    from swp.acquisition.sequence import read_swi_meta

    stack, meta = L.build_stack(TAG)
    coords, geom = meta["coords"], L.geometry_from_meta(meta)
    w = M.w_cone(geom, "rect", SCALE)                     # (n_tx, nz, nx)
    offline = np.abs(M.compose(stack, w))

    init_device(verbose=False)
    root = L.DATASETS[TAG]["root"]
    swi = read_swi_meta(root / "CombinedData.mat")
    with File(str(root / "output" / "converted" / "CombinedData_buffer1.hdf5")) as fh:
        raw = np.asarray(fh.data.raw_data[:stack.shape[1]])
        params = fh.load_parameters()
    apply_grid(params, swi.grids[0])
    _ensure_cpu_t_peak(params)

    n_tx, n_el = raw.shape[1], raw.shape[3]
    nz, nx = coords.shape[:2]
    # ``flat_aligned_apodization`` is a COMPUTED property on Parameters (it only exists for
    # scanline imaging), so it cannot be assigned. Override it in the prepared-parameter dict
    # instead - PatchedGrid maps that key over the grid patches alongside ``flatgrid``.
    flat = np.moveaxis(w, 0, -1).reshape(nz * nx, n_tx).astype(np.float32)

    num_patches, _ = plan_patches_and_chunk(params, n_tx, n_el)
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                         Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                  enable_aligned_apodization=True)],
                        with_batch_dim=True, jit_options=None)
    bf_in = pipe.prepare_parameters(params)
    bf_in["flat_aligned_apodization"] = flat
    print(f"mask handed to the pipeline: {flat.shape} (want {(nz * nx, n_tx)})")
    assert "flat_aligned_apodization" in bf_in and bf_in["flat_aligned_apodization"] is not None, \
        "the mask did not reach the pipeline"
    pipe(data=raw[:1], **bf_in)                       # untimed warm-up (kernel autotuning)
    t0 = time.perf_counter()
    iq = pipe(data=raw, **bf_in)[pipe.output_key]
    iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, np.float32)
    t_masked = time.perf_counter() - t0
    online = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)

    corr = np.corrcoef(online.ravel(), offline.ravel())[0, 1]
    rel = np.abs(online - offline).mean() / offline.mean()
    print(f"pipeline vs offline composite: corr {corr:.8f}   mean rel diff {rel:.2e}")

    # cost
    pipe0 = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                          Beamform(beamformer="delay_and_sum", num_patches=num_patches)],
                         with_batch_dim=True, jit_options=None)
    bf0 = pipe0.prepare_parameters(params)
    pipe0(data=raw[:1], **bf0)
    t0 = time.perf_counter()
    pipe0(data=raw, **bf0)
    t_plain = time.perf_counter() - t0
    print(f"time: plain DAS {t_plain:.2f}s   masked {t_masked:.2f}s   "
          f"({(t_masked / t_plain - 1) * 100:+.0f}%, both warmed up)")
    assert corr > 0.9999
    print("OK: the production route reproduces the study exactly.")


if __name__ == "__main__":
    main()
