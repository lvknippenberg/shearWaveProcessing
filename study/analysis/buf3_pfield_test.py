"""Buffer 3 (focused): beamform one frame with pfield OFF (current) vs ON.

The beams fully tile the sector (2.28x overlap), so the radial streaks are not gaps. With
enable_pfield=False every pixel coherently sums all 73 transmits, including beams steered up
to 80 deg away that never insonified it - their delay curves still pass through the pixel and
contribute clutter aligned with the beam directions. enable_pfield weights each transmit's
contribution by its simulated transmit field, which is what Verasonics' own TXPD does.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = r"D:\Luuk van Knippenberg\Github\shearWaveProcessing"
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate

from swp.acquisition.beamform import apply_grid, plan_patches_and_chunk, _ensure_cpu_t_peak
from swp.acquisition.sequence import read_swi_meta

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
CONV = os.path.join(FOLDER, "output", "converted", "CombinedData_buffer3.hdf5")
MAT = os.path.join(FOLDER, "CombinedData.mat")
FRAME = 0

init_device(verbose=True)

meta = read_swi_meta(MAT)
grid = meta.grids[2]                       # buffer 3 (0-based index 2)

with File(CONV) as fh:
    raw = np.asarray(fh.data.raw_data[FRAME:FRAME + 1])
    params = fh.load_parameters()
print(f"raw {raw.shape}")

apply_grid(params, grid)
_ensure_cpu_t_peak(params)
n_tx, n_el = raw.shape[1], raw.shape[3]
num_patches, _ = plan_patches_and_chunk(params, n_tx, n_el)
print(f"num_patches={num_patches}")


def beamform(enable_pfield):
    ops = [Cast(dtype="float32"), Demodulate(),
           Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                    enable_pfield=enable_pfield)]
    pipe = zea.Pipeline(ops, with_batch_dim=True, jit_options=None)
    bf_in = pipe.prepare_parameters(params)
    iq = pipe(data=raw, **bf_in)[pipe.output_key]
    return np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)


out = {}
for flag in (False, True):
    try:
        iq = beamform(flag)
        env = np.sqrt(iq[0, ..., 0] ** 2 + iq[0, ..., 1] ** 2)
        out[flag] = env
        print(f"pfield={flag}: IQ {iq.shape}  env max {env.max():.4g}")
    except Exception as exc:
        print(f"pfield={flag}: FAILED {type(exc).__name__}: {exc}")

if len(out) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, flag in zip(axes, (False, True)):
        db = 20 * np.log10(out[flag] / out[flag].max() + 1e-12)
        ax.imshow(db, cmap="gray", vmin=-50, vmax=0, aspect="auto")
        ax.set_title(f"buffer 3 frame {FRAME} -- enable_pfield={flag}"
                     f"{'  (current)' if not flag else '  (proposed)'}")
    plt.tight_layout()
    plt.savefig("buf3_pfield_compare.png", dpi=95)
    print("wrote buf3_pfield_compare.png")
    np.save("buf3_env_off.npy", out[False])
    np.save("buf3_env_on.npy", out[True])
