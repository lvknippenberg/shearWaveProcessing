"""Validation: compositing the cached per-transmit stack with W=1 must reproduce the
stored standard reconstruction exactly (up to float32 summation order).

Without this check every later number is unanchored - the same validation the nearest-k
ladder used (reproducing the pipeline at k=6).
"""
import sys

import h5py
import numpy as np

import b1_lib as L
import b1_masks as M

tag = sys.argv[1] if len(sys.argv) > 1 else "phantom"
stack, meta = L.build_stack(tag)
coords = meta["coords"]
frames = meta["frames"]

std_path = L.DATASETS[tag]["root"] / "output" / "CombinedData_buffer1_iq.hdf5"
with h5py.File(std_path, "r") as f:
    g = f["tracks/track_0/data/beamformed_data"]
    iq = np.asarray(g["values"][frames])
    co_std = np.asarray(g["coordinates"])
env_std = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)

geom = L.geometry_from_meta(meta)
print(geom.summary())
print(f"grid mine {coords.shape}  stored {co_std.shape}  "
      f"max coord diff {np.abs(coords - co_std).max():.3e} m")

mine = np.abs(M.compose(stack, M.w_all(geom)))
print(f"envelope: mine {mine.shape} stored {env_std.shape}")
num = np.abs(mine - env_std).max()
den = env_std.max()
corr = np.corrcoef(mine.ravel(), env_std.ravel())[0, 1]
rel = np.abs(mine - env_std).mean() / (env_std.mean() + 1e-20)
print(f"max |diff| / max env = {num / den:.3e}    mean rel diff = {rel:.3e}    "
      f"corr = {corr:.8f}")
assert corr > 0.9999, "per-transmit stack does NOT reproduce the pipeline"
print("OK: the cached stack composites back to the stored reconstruction.")
