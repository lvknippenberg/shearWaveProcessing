"""Confirm the widebeam buffers were rebuilt under the new default, across the study.

The filename does not carry the reconstruction, so the check is the ``description`` string
that ``process_bmode_buffer`` stores - the same mechanism that distinguishes buffer-3 files
written before and after the REFoCUS switch. Also reports frame count and mtime, so a folder
that silently kept an old file cannot pass.
"""
from __future__ import annotations

import glob
import os
import time

import h5py

WANT = "tx-window rect x1"
PAT = "Z:/raw_data/C*/*/output/CombinedData_buffer{b}_iq.hdf5"
BUFFERS = (1, 5)


def check(buffer):
    files = sorted(glob.glob(PAT.format(b=buffer)))
    print(f"buffer {buffer}: {len(files)} IQ files")
    ok, bad = 0, []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                desc = str(f.attrs.get("description", ""))
                n = f["tracks/track_0/data/beamformed_data/values"].shape[0]
        except Exception as exc:                                  # noqa: BLE001
            bad.append((p, f"unreadable: {exc}"))
            continue
        if WANT not in desc:
            bad.append((p, f"description={desc!r}"))
            continue
        ok += 1
        if ok <= 2:
            stamp = time.strftime("%d-%b %H:%M", time.localtime(os.path.getmtime(p)))
            print(f"  ok  {n:4d} fr  {stamp}  {p.split('raw_data')[-1]}")
    print(f"  -> {ok} carry {WANT!r}, {len(bad)} do not\n")
    for q, why in bad:
        print(f"  BAD  {q}\n       {why}")
    return len(bad)


def main():
    return 1 if sum(check(b) for b in BUFFERS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
