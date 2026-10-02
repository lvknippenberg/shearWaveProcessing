"""Beamform buffer 3 of every Strain_data folder of one subject on a LOCAL copy.

Strain_data folders have no output/ yet, and building CombinedData.mat + IQ would write into the
NAS folder. To keep Z: read-only, copy AcquisitionParametersAndECG.mat + RF_data_3.bin to
<dest>/<folder name>/ and run Stage A (buffer 3 only, + GIF) there.

Run several copies in parallel to go faster (each claims a folder via <local>/.claim; delete
stale .claim files after a crash).

Usage: python local_strain_buffer3.py Z:\\raw_data\\C000000049 <dest root>
"""
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

COPY = ("AcquisitionParametersAndECG.mat", "RF_data_3.bin")


def main():
    subj, dest = Path(sys.argv[1]), Path(sys.argv[2])
    from swp.acquisition import process_folder

    for src in sorted(p for p in subj.iterdir() if p.is_dir() and "strain_data" in p.name.lower()):
        local = dest / subj.name / src.name
        if (local / "output" / "CombinedData_buffer3_iq.gif").is_file():
            print(f"skip {src.name} (done)")
            continue
        local.mkdir(parents=True, exist_ok=True)
        try:                                    # claim the folder so parallel workers never share one
            os.close(os.open(local / ".claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            print(f"skip {src.name} (claimed by another worker)")
            continue
        t0 = time.time()
        for name in COPY:
            s, d = src / name, local / name
            if not d.is_file() or d.stat().st_size != s.stat().st_size:
                shutil.copyfile(s, d)
        print(f"copied {src.name} in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        try:
            process_folder(local, buffers_matlab=[3], save_converted=True)
        except Exception as exc:                                  # noqa: BLE001 - keep going
            print(f"FAILED {src.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"beamformed {src.name} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
