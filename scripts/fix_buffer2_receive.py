"""Repair CombinedData.mat when the buffer-2 (shear-wave tracking) Receive layout does not match
the acquisition.

Background
----------
`make_combined_data.m` merges the runtime dynamic parameters with a *base config* that supplies the
constant structs -- including `Receive`, which tells the reader how to slice the raw RF binary.
For a 10-push phantom acquisition it picks `PhantomSweep/BaseConfig_10frames_<cyc>cycles_<el>elements_<pri>PRI.mat`.
Those base configs were recorded on 2026-08-07 with **SW.endDepth = 200 wl (maxAcqLength 262)**, so
their buffer-2 receives are **2688 samples** long. The 2026-08-17/18 acquisitions were made with
**SW.endDepth = 300 wl (maxAcqLength 392) -> 3968 samples**. Only buffer 2 follows SW.endDepth (the
B-mode buffers are fixed full-depth), so only buffer 2 is affected -- which is why the B-modes look
normal while the tracking IQ is scrambled (frame-to-frame speckle correlation collapses ~0.98 -> 0.4
and the reconstructed frames sit at the wrong depth).

Fix
---
Recompute the buffer-2 receive length from the *runtime* parameters, exactly as Verasonics does:

    nsamp = ceil(2 * SW.maxAcqLength * Receive.samplesPerWave / 128) * 128
    startSample = 1 + (acqNum - 1) * nsamp,  endSample = startSample + nsamp - 1,
    Receive.endDepth = nsamp / (2 * samplesPerWave)

and write those back into `CombinedData.mat` (v7.3 HDF5, in place -- values only, no resize).
Verified against the 2026-08-04 phantom data, whose base config already carries the 3968-sample
layout and which reconstructs correctly.

    python scripts/fix_buffer2_receive.py --root D:/swp_ph17 [--check]
"""
from __future__ import annotations

import argparse
import math
import os

import h5py
import numpy as np
import scipy.io as sio


def _refs(f, group, field):
    return np.atleast_1d(np.array(group[field]).squeeze())


def _vals(f, group, field):
    return np.array([float(np.array(f[r]).squeeze()) for r in _refs(f, group, field)])


def expected_nsamp(folder):
    """Buffer-2 receive length implied by the runtime acquisition parameters."""
    m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                    squeeze_me=True, struct_as_record=False)
    return float(m["SW"].maxAcqLength)


def inspect(path):
    with h5py.File(path, "r") as f:
        R = f["Receive"]
        bn = _vals(f, R, "bufnum")
        m = bn == 2
        ns = np.unique(_vals(f, R, "endSample")[m] - _vals(f, R, "startSample")[m] + 1)
        spw = np.unique(_vals(f, R, "samplesPerWave")[m])
        return ns, spw


def patch(folder, dry_run=False):
    path = os.path.join(folder, "CombinedData.mat")
    mal = expected_nsamp(folder)
    with h5py.File(path, "r+" if not dry_run else "r") as f:
        R = f["Receive"]
        bn = _vals(f, R, "bufnum")
        acq = _vals(f, R, "acqNum")
        spw = _vals(f, R, "samplesPerWave")
        ss = _vals(f, R, "startSample")
        es = _vals(f, R, "endSample")
        idx = np.where(bn == 2)[0]
        s = float(np.unique(spw[idx])[0])
        want = int(math.ceil(2 * mal * s / 128.0) * 128)
        have = int(np.unique(es[idx] - ss[idx] + 1)[0])
        if have == want:
            return "ok", have, want
        if dry_run:
            return "needs-fix", have, want

        rows = int(np.array(f["Resource"]["RcvBuffer"]["rowsPerFrame"]) .squeeze().tolist()
                   .__getitem__(0)) if False else None   # (not needed; checked below)
        rpf = [float(np.array(f[r]).squeeze())
               for r in np.atleast_1d(np.array(f["Resource"]["RcvBuffer"]["rowsPerFrame"]).squeeze())]
        n_per_frame = int(acq[idx].max())
        if n_per_frame * want > rpf[1]:
            raise RuntimeError(f"{folder}: {n_per_frame} x {want} exceeds rowsPerFrame {rpf[1]}")

        r_ss = _refs(f, R, "startSample")
        r_es = _refs(f, R, "endSample")
        r_ed = _refs(f, R, "endDepth")
        end_depth = want / (2.0 * s)
        for i in idx:
            new_start = 1.0 + (acq[i] - 1.0) * want
            f[r_ss[i]][...] = np.array(new_start, dtype=f[r_ss[i]].dtype).reshape(f[r_ss[i]].shape)
            f[r_es[i]][...] = np.array(new_start + want - 1, dtype=f[r_es[i]].dtype).reshape(f[r_es[i]].shape)
            f[r_ed[i]][...] = np.array(end_depth, dtype=f[r_ed[i]].dtype).reshape(f[r_ed[i]].shape)
    return "fixed", have, want


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="parent folder of the measurement folders")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    a = ap.parse_args()

    n = {"ok": 0, "fixed": 0, "needs-fix": 0, "skip": 0}
    for d in sorted(os.listdir(a.root)):
        folder = os.path.join(a.root, d)
        if not os.path.isfile(os.path.join(folder, "CombinedData.mat")):
            continue
        try:
            status, have, want = patch(folder, dry_run=a.check)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {d[-8:]}  ERROR {exc}")
            n["skip"] += 1
            continue
        n[status] += 1
        print(f"  {d[-8:]}  {status:10s} buffer2 nsamp have={have} want={want}")
    print(n)


if __name__ == "__main__":
    main()
