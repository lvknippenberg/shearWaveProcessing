"""Cache zea's simulated transmit pressure field for buffer 1, as a weight map.

This is the weighting ``BufferSpec.pfield`` switches on (off for buffer 1 today, on for
the focused buffer 3).  Computed here so it can be compared on equal terms with the
geometric cone masks, including under the normalisations the pipeline never applies.
"""
from __future__ import annotations

import numpy as np

import b1_lib as L


def pfield_map(tag, norm=True, alpha=1, percentile=10, overwrite=False):
    """(n_tx, nz, nx) float32 transmit-field weight for the buffer-1 grid."""
    cache = L.CACHE / f"{tag}_b1_pfield_{int(norm)}_{alpha}_{percentile}.npy"
    if cache.is_file() and not overwrite:
        return np.load(cache)

    from zea import File, init_device
    from zea.beamform.pfield import compute_pfield
    from swp.acquisition.beamform import apply_grid
    from swp.acquisition.sequence import read_swi_meta

    init_device(verbose=False)
    root = L.DATASETS[tag]["root"]
    meta = read_swi_meta(root / "CombinedData.mat")
    with File(str(root / "output" / "converted" / "CombinedData_buffer1.hdf5")) as fh:
        params = fh.load_parameters()
    apply_grid(params, meta.grids[0])
    d = dict(params)
    pf = compute_pfield(
        sound_speed=float(np.asarray(d["sound_speed"]).ravel()[0]),
        center_frequency=float(np.asarray(d["center_frequency"]).ravel()[0]),
        probe_bandwidth_percent=float(np.asarray(d["probe_bandwidth_percent"]).ravel()[0]),
        n_el=int(np.asarray(d["n_el"]).ravel()[0]),
        probe_geometry=np.asarray(d["probe_geometry"]),
        tx_apodizations=np.asarray(d["tx_apodizations"]),
        grid=np.asarray(params.grid),
        t0_delays=np.asarray(d["t0_delays"]),
        norm=norm, alpha=alpha, percentile=percentile,
    )
    pf = np.asarray(pf.cpu() if hasattr(pf, "cpu") else pf, np.float32)
    # compute_pfield returns (nz, nx, n_tx) or (n_tx, nz, nx) depending on version
    if pf.shape[0] != np.asarray(d["tx_apodizations"]).shape[0]:
        pf = np.moveaxis(pf, -1, 0)
    np.save(cache, pf)
    return pf


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "invivo"
    for kw in (dict(norm=True), dict(norm=False)):
        p = pfield_map(tag, **kw)
        print(kw, p.shape, p.dtype, float(p.min()), float(p.max()))
