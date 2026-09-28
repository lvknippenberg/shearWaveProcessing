"""Inventory for the 2026-09-28 second-acquisition check.

Per subject in Z:/raw_data: the SW_data folders in acquisition order (timestamp in the folder
name), the SECOND one, whether its buffer-1/3 IQ exists and its buffer-3 unwrap flag; plus the
Strain_data folders and whether every Strain_data file listed in the DataHub download manifest is
on disk with the manifest size.

    python study/analysis/second_acq_inventory.py  -> study/logs/second_acq_inventory.csv
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")

ROOT = Path("Z:/raw_data")
_TS = re.compile(r"_(\d{1,2}-[A-Za-z]+-\d{4}_\d{2}-\d{2}-\d{2})$")


def acq_time(name):
    m = _TS.search(name)
    return datetime.strptime(m.group(1), "%d-%B-%Y_%H-%M-%S") if m else datetime.max


def main():
    from swp.acquisition.unwrap import buffer_files, read_flag

    manifest = defaultdict(list)                     # subject -> [(relpath under subject, size)]
    for line in open(ROOT / "_mdr_manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        parts = r["remote"].strip("/").split("/")    # P000000569/Cxxx/<folder>/<file>
        if len(parts) >= 4 and "strain_data" in parts[2].lower():
            manifest[parts[1]].append(("/".join(parts[2:]), r.get("size"), r.get("status")))

    rows = []
    for subj in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("C")):
        sw = sorted((d for d in subj.iterdir() if d.is_dir() and "_sw_data_" in d.name.lower()),
                    key=lambda d: acq_time(d.name))
        st = [d for d in subj.iterdir() if d.is_dir() and "strain_data" in d.name.lower()]
        missing, wrong = 0, 0
        for rel, size, _ in manifest.get(subj.name, []):
            p = subj / rel
            if not p.exists():
                missing += 1
            elif size is not None and p.stat().st_size != size:
                wrong += 1
        row = dict(subject=subj.name, n_sw=len(sw), n_strain_folders=len(st),
                   strain_files_manifest=len(manifest.get(subj.name, [])),
                   strain_missing=missing, strain_size_mismatch=wrong)
        if len(sw) >= 2:
            f = sw[1]
            out = f / "output"
            conv, iq3 = buffer_files(out)
            rec = read_flag(iq3) if iq3 else None
            row.update(second=f.name,
                       b1_iq=(out / "CombinedData_buffer1_iq.hdf5").exists(),
                       b3_iq=iq3 is not None,
                       b3_variants=";".join(sorted(p.name.split("buffer3_")[1].replace("_iq.hdf5", "")
                                                   for p in out.glob("*_buffer3_*_iq.hdf5"))),
                       b4_gif=(out / "CombinedData_buffer4_iq.gif").exists(),
                       unwrap_status=rec.get("status") if rec else "",
                       unwrap_method=rec.get("method") if rec else "")
        rows.append(row)
        print(row, flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    dst = _REPO / "study" / "logs" / "second_acq_inventory.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
