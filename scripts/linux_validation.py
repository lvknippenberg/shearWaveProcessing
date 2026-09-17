"""Validate a new machine/backend: re-run stages 1-2 on one folder into a side directory, then compare.

``run``      convert + beamform + GIFs for ONE measurement folder into ``<folder>/<out-name>``
             (default ``output_linux``), leaving the reference ``<folder>/output`` untouched.
             Writes ``run_info.json`` (backend, device, versions, git commit, timings) next to it.
``compare``  every HDF5 in the reference ``output/`` against its twin in ``<out-name>/``,
             dataset by dataset: shapes, exact equality where it should hold (coordinates,
             timestamps, integers), and for float data the relative RMS error + correlation, with
             an envelope (log) correlation for IQ. GIFs by frame count and mean pixel difference.
             Exits non-zero if anything differs beyond tolerance. Needs only numpy/h5py/imageio.

The existing ``CombinedData.mat`` is reused and validated (no MATLAB needed), so build it on the
Windows machine first for any folder that does not have one yet.

Usage (server, inside the zea container):
    KERAS_BACKEND=jax python scripts/linux_validation.py run     --folder <measurement folder>
    python scripts/linux_validation.py compare --folder <measurement folder>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

REL_TOL = 1e-3          # relative RMS error below this counts as "close"
CHUNK = 64              # frames per chunk when streaming large datasets


# ---------------------------------------------------------------------------------------- run
def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return None


def cmd_run(a):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    folder = Path(a.folder)
    out = folder / a.out_name
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} is not empty - pass --force to write into it anyway")
    out.mkdir(exist_ok=True)

    import numpy as np
    import h5py
    import zea
    from swp.acquisition import process_folder

    info = dict(folder=str(folder), out=str(out), host=platform.node(),
                platform=platform.platform(), python=sys.version.split()[0],
                keras_backend=os.environ["KERAS_BACKEND"], zea=getattr(zea, "__version__", None),
                numpy=np.__version__, h5py=h5py.__version__, git_commit=_git_commit(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        if info["keras_backend"] == "jax":
            import jax
            info.update(jax=jax.__version__, devices=[str(d) for d in jax.devices()])
        elif info["keras_backend"] == "torch":
            import torch
            info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                        devices=[torch.cuda.get_device_name(0)] if torch.cuda.is_available() else [])
    except Exception as exc:                              # noqa: BLE001
        info["device_error"] = str(exc)
    print(json.dumps(info, indent=1), flush=True)

    t0 = time.perf_counter()
    written = process_folder(folder, output_dir=out, make_gifs=not a.no_gifs,
                             save_converted=not a.no_converted, overwrite=True,
                             buffers_matlab=[int(b) for b in a.buffers.split(",")] if a.buffers else None)
    info.update(seconds=round(time.perf_counter() - t0, 1), n_written=len(written),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out / "run_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"\ndone: {len(written)} IQ file(s) in {info['seconds'] / 60:.1f} min -> {out}")


# ------------------------------------------------------------------------------------ compare
def _datasets(h5):
    import h5py
    out = {}
    h5.visititems(lambda name, obj: out.__setitem__(name, obj) if isinstance(obj, h5py.Dataset) else None)
    return out


def _compare_dataset(da, db):
    """-> dict(verdict, detail). Streams along axis 0 for large arrays."""
    import numpy as np

    if da.shape != db.shape:
        return dict(verdict="SHAPE", detail=f"{da.shape} vs {db.shape}")
    if da.dtype.kind not in "fc":
        a, b = da[()], db[()]
        same = np.array_equal(a, b)
        return dict(verdict="exact" if same else "DIFFERENT", detail="" if same else "non-float mismatch")

    is_iq = da.ndim >= 3 and da.shape[-1] == 2
    n0 = da.shape[0] if da.ndim else 1
    s = dict(err2=0.0, ref2=0.0, ab=0.0, aa=0.0, bb=0.0, sa=0.0, sb=0.0, n=0, maxabs=0.0,
             la=0.0, lb=0.0, lab=0.0, laa=0.0, lbb=0.0, ea=0.0, eb=0.0, ne=0)
    exact = True
    for i in range(0, max(n0, 1), CHUNK):
        a = np.asarray(da[i:i + CHUNK] if da.ndim else da[()], dtype=np.float64).ravel()
        b = np.asarray(db[i:i + CHUNK] if db.ndim else db[()], dtype=np.float64).ravel()
        exact &= bool(np.array_equal(a, b))
        d = a - b
        s["err2"] += d @ d; s["ref2"] += a @ a; s["maxabs"] = max(s["maxabs"], float(np.abs(d).max(initial=0)))
        s["sa"] += a.sum(); s["sb"] += b.sum(); s["ab"] += a @ b; s["aa"] += a @ a; s["bb"] += b @ b
        s["n"] += a.size
        if is_iq:
            A = np.asarray(da[i:i + CHUNK], np.float64); B = np.asarray(db[i:i + CHUNK], np.float64)
            ea = np.hypot(A[..., 0], A[..., 1]).ravel(); eb = np.hypot(B[..., 0], B[..., 1]).ravel()
            la = np.log10(ea + 1e-12 * max(ea.max(initial=0), 1e-30))
            lb = np.log10(eb + 1e-12 * max(eb.max(initial=0), 1e-30))
            s["la"] += la.sum(); s["lb"] += lb.sum(); s["lab"] += la @ lb; s["laa"] += la @ la
            s["lbb"] += lb @ lb; s["ea"] += ea.sum(); s["eb"] += eb.sum(); s["ne"] += ea.size
        if not da.ndim:
            break
    if exact:
        return dict(verdict="exact", detail="")

    def corr(sx, sy, sxy, sxx, syy, n):
        cov = sxy - sx * sy / n
        var = (sxx - sx * sx / n) * (syy - sy * sy / n)
        return float(cov / np.sqrt(var)) if var > 0 else float("nan")

    rel = float(np.sqrt(s["err2"] / s["ref2"])) if s["ref2"] > 0 else float("inf")
    r = corr(s["sa"], s["sb"], s["ab"], s["aa"], s["bb"], s["n"])
    detail = f"relRMS {rel:.2e}  corr {r:.6f}  max|d| {s['maxabs']:.3g}"
    if is_iq and s["ne"]:
        lr = corr(s["la"], s["lb"], s["lab"], s["laa"], s["lbb"], s["ne"])
        detail += f"  logEnv corr {lr:.6f}  env ratio {s['eb'] / s['ea']:.4f}"
    return dict(verdict="close" if rel < REL_TOL else "DIFFERENT", detail=detail, rel=rel)


def _compare_gif(pa, pb):
    import numpy as np
    import imageio.v3 as iio

    fa, fb = iio.imread(pa, index=None), iio.imread(pb, index=None)
    if fa.shape != fb.shape:
        return dict(verdict="SHAPE", detail=f"{fa.shape} vs {fb.shape}")
    mad = float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean())
    return dict(verdict="exact" if mad == 0 else ("close" if mad < 1.0 else "DIFFERENT"),
                detail=f"{fa.shape[0]} frames, mean |dpixel| {mad:.3f}")


def cmd_compare(a):
    import h5py

    folder = Path(a.folder)
    ref, new = folder / a.ref_name, folder / a.out_name
    # Only what the standard pipeline writes; output/ also holds one-off reconstructions
    # (incoh, refocus-*, deconv-*, pfieldnorm) that a plain run does not reproduce.
    std = re.compile(r"^(converted[\\/])?CombinedData_buffer\d+(_meas\d+)?(_iq)?\.(hdf5|gif)$")
    ref_files = sorted(p.relative_to(ref) for p in ref.rglob("*.hdf5")
                       if std.match(str(p.relative_to(ref))))
    new_files = {p.relative_to(new) for p in new.rglob("*.hdf5")}
    if a.pattern:
        ref_files = [p for p in ref_files if a.pattern in str(p)]
    rows, bad = [], 0
    has_converted = (new / "converted").is_dir()
    for rel in ref_files:
        if rel not in new_files:
            if rel.parts[0] == "converted" and not has_converted:
                continue                                 # run with --no-converted
            if new_files:                                # only report files the new run should have
                rows.append((str(rel), "-", "MISSING", "not in new output"))
            continue
        with h5py.File(ref / rel, "r") as fa, h5py.File(new / rel, "r") as fb:
            A, B = _datasets(fa), _datasets(fb)
            for extra in sorted(set(B) - set(A)):
                rows.append((str(rel), extra, "EXTRA", "only in new"))
            desc_a, desc_b = fa.attrs.get("description"), fb.attrs.get("description")
            if desc_a != desc_b:
                # Newer runs append the reconstruction as " [delay-and-sum]" etc.; that alone is
                # a label change, not a data difference.
                strip = lambda s: re.sub(r"\s*\[[^\]]*\]$", "", str(s))  # noqa: E731
                verdict = "note" if strip(desc_a) == strip(desc_b) else "DIFFERENT"
                rows.append((str(rel), "@description", verdict, f"{desc_a!r} vs {desc_b!r}"))
            for name in sorted(A):
                if name not in B:
                    rows.append((str(rel), name, "MISSING", "dataset not in new"))
                    continue
                res = _compare_dataset(A[name], B[name])
                rows.append((str(rel), name, res["verdict"], res["detail"]))
        print(f"  compared {rel}", flush=True)
    for pa in sorted(p for p in ref.glob("*.gif") if std.match(p.name)):
        pb = new / pa.name
        if pb.exists() and (not a.pattern or a.pattern in pa.name):
            res = _compare_gif(pa, pb)
            rows.append((pa.name, "(gif)", res["verdict"], res["detail"]))

    report = [f"reference: {ref}", f"new:       {new}", ""]
    width = max((len(r[0]) for r in rows), default=10)
    for f, d, v, det in rows:
        if a.all or v not in ("exact",):
            report.append(f"{v:9s} {f:{width}s} {d}  {det}")
    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    bad = sum(v for k, v in counts.items() if k not in ("exact", "close", "note"))
    report += ["", "summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
               "PASS" if bad == 0 else f"FAIL ({bad} item(s) differ)"]
    text = "\n".join(report)
    print("\n" + text)
    with open(new / "compare_report.txt", "w") as f:
        f.write(text + "\n")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["run", "compare"])
    p.add_argument("--folder", required=True, help="measurement folder (with CombinedData.mat)")
    p.add_argument("--out-name", default="output_linux", help="side output dir inside the folder")
    p.add_argument("--ref-name", default="output", help="compare: reference output dir")
    p.add_argument("--buffers", default=None, help="run: only these MATLAB buffers, e.g. 3,4")
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--no-converted", action="store_true")
    p.add_argument("--force", action="store_true", help="run: write into a non-empty out dir")
    p.add_argument("--pattern", default=None, help="compare: only files containing this text")
    p.add_argument("--all", action="store_true", help="compare: also list exact matches")
    a = p.parse_args()
    return cmd_run(a) if a.mode == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
