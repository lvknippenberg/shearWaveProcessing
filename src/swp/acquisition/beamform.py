"""Stage A / task 1: beamform the SWI RF buffers to IQ and save as zea HDF5.

The pipeline reads RF straight from the Verasonics workspace and beamforms it to
complex IQ on the fixed PData grid (:mod:`swi_meta`), writing one native
``beamformed_data`` file per buffer (and one per measurement for the active
shear-wave buffer). The IQ this produces feeds both the B-mode GIFs (task 2,
:mod:`make_gifs`) and the shear-wave analysis (task 3).

Performance (Stage A optimization):

* **Read RF once, reuse when possible.** If a converted zea file already exists
  for a buffer it is reused (a cheap read-back) instead of re-reading the ``.mat``;
  pass ``overwrite=True`` (CLI ``--overwrite``) to force reconversion. Otherwise
  the RF and metadata are read a single time and used both to write the converted
  zea file (the group's database copy) and to beamform - no re-reading a written
  HDF5 back in. Set ``save_converted=False`` (CLI ``--no-converted``) to skip the
  database copy.
* **Memory-targeted patching, portable across GPUs.** ``num_patches`` is the
  *smallest* count that keeps each patch within a memory budget **scaled to the
  active GPU's total memory** (a 12 GB card gets more patches, an 80 GB card
  fewer), instead of a large fixed number - in eager mode every patch is a separate
  kernel launch, so over-patching a cheap buffer dominated the runtime. If a call
  still OOMs (bad estimate, fragmentation, a shared GPU), it is retried with more
  patches, so it completes on any GPU. Override the budget with
  ``ZEA_SWI_PATCH_BUDGET`` (gathered elements).
* **Backend-aware JIT.** On the JAX backend the whole pipeline is compiled
  (``jit_options="pipeline"``); torch runs eager. Set ``ZEA_SWI_JIT`` to
  ``pipeline``/``ops``/``none`` to override.

Cross-platform: give :func:`process_folder` (or the CLI) a measurement folder
containing ``CombinedData.mat``; outputs go to ``<folder>/output``. Paths use
``pathlib`` and the backend is taken from ``KERAS_BACKEND`` (default ``torch``;
set ``KERAS_BACKEND=jax`` on Linux for JIT).
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
# Reduce torch CUDA allocator fragmentation across long frame loops (reserved
# memory otherwise creeps up until it OOMs on buffers with hundreds of frames).
# Harmless on other backends; must be set before torch is imported via `import zea`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import numpy as np

import zea
from zea import File, Parameters, init_device
from zea.ops import Beamform, Cast, Demodulate
from zea.data.file import CustomElement
from zea.data.convert.verasonics import VerasonicsFile

from .sequence import (
    SPEC_BY_INDEX,
    BufferSpec,
    assemble_tracking_frames,
    read_swi_meta,
    BufferGrid,
)

# Match the PData density (0.5 wavelength pitch = 2 pixels/wavelength) so the grid
# reproduces the Verasonics field of view with predictable memory.
PIXELS_PER_WAVELENGTH = 2.0

# Beamforming memory budget, in "gathered elements" (pixels x transmits x
# elements): the largest per-patch / per-call gather we let a single kernel do.
# ``num_patches`` and the frame-chunk are the smallest counts that keep the gather
# under this budget (fewest kernel launches without OOM).
#
# The budget is scaled to the actual GPU: ``REFERENCE_BUDGET`` is the value tuned
# on a 24 GB GPU (one patch of ~3e8 elements OOMs there, ~8e7 is safe with margin);
# on a card with N GB total we use ``REFERENCE_BUDGET * N / 24`` so a smaller GPU
# gets more patches and a bigger GPU fewer. An OOM at runtime is still caught and
# retried with more patches (see :func:`beamform_frames`), so correctness never
# depends on the estimate being exactly right. Override with ``ZEA_SWI_PATCH_BUDGET``
# (in gathered elements) to pin it manually.
REFERENCE_BUDGET = 6e7
REFERENCE_GPU_GB = 24.0
MIN_BUDGET = 1e7          # floor so a tiny/uncertain GPU still makes progress

_BUDGET_CACHE = {}

# Compression for the stored IQ (native beamformed_data must stay float32, so
# precision is fixed; only the HDF5 filter is tunable). None = fastest write,
# largest files; "lzf" = fast, moderate size.
DEFAULT_COMPRESSION = "lzf"


def _jit_options():
    """Whole-pipeline JIT on JAX, eager elsewhere; overridable via ``ZEA_SWI_JIT``."""
    override = os.environ.get("ZEA_SWI_JIT")
    if override:
        return None if override.lower() == "none" else override
    backend = os.environ.get("KERAS_BACKEND", "torch").lower()
    return "pipeline" if backend == "jax" else None


def free_gpu_memory():
    """Release cached GPU allocations between buffers (no-op off torch/CUDA)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _gpu_total_gb():
    """Total memory (GB) of the active GPU, or ``None`` if it can't be determined.

    Works for the torch and JAX backends; returns ``None`` on CPU / when unknown,
    in which case the reference budget is used and the OOM-retry does the rest.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return props.total_memory / 1e9
    except Exception:
        pass
    try:
        import jax

        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit")
        if limit:
            return limit / 1e9
    except Exception:
        pass
    return None


def memory_budget():
    """Per-patch / per-call gather budget (elements), scaled to this GPU.

    Cached after first call. ``ZEA_SWI_PATCH_BUDGET`` overrides it explicitly.
    """
    if "budget" in _BUDGET_CACHE:
        return _BUDGET_CACHE["budget"]
    override = os.environ.get("ZEA_SWI_PATCH_BUDGET")
    if override:
        budget = float(override)
    else:
        total_gb = _gpu_total_gb()
        budget = (REFERENCE_BUDGET * total_gb / REFERENCE_GPU_GB
                  if total_gb else REFERENCE_BUDGET)
    budget = max(MIN_BUDGET, budget)
    _BUDGET_CACHE["budget"] = budget
    print(f"    patch budget {budget:.2e} elements "
          f"(GPU {_gpu_total_gb() or '?'} GB)")
    return budget


# ---------------------------------------------------------------------------
def _build_params(scan: dict, geom, n_ax, n_el, n_tx):
    """Build a :class:`~zea.Parameters` from a scan dict + probe geometry."""
    # center_frequency is read per-transmit; collapse to a scalar when uniform so
    # wavelength / grid sizing (which expect a scalar) work, matching the file path.
    cf = np.asarray(scan.get("center_frequency"))
    if cf.size > 1 and np.unique(cf).size == 1:
        scan = {**scan, "center_frequency": float(cf.reshape(-1)[0])}
    pdict = {**scan, "probe_geometry": np.asarray(geom, np.float32),
             "n_ax": int(n_ax), "n_el": int(n_el), "n_tx": int(n_tx)}
    pdict = {k: v for k, v in pdict.items() if v is not None}
    return Parameters(**pdict)


def read_buffer(vf: VerasonicsFile, buffer_index: int, converted_path=None,
                compression=DEFAULT_COMPRESSION, overwrite=False):
    """Read one RF buffer and build its :class:`~zea.Parameters`.

    Fast path: if ``converted_path`` already exists (and ``overwrite`` is False),
    the RF and parameters are loaded from that converted zea file instead of
    re-reading and re-decoding the ``.mat`` (a read-back is much cheaper than a
    reconversion). Otherwise the RF and full metadata are read from the workspace
    **once**, and - when ``converted_path`` is given - written there as a standard
    zea file (the product of ``VerasonicsFile.to_zea``, for the group's zea
    database) from the already-in-memory data, so the RF is never read twice.

    Returns ``(raw, params)`` with ``raw`` shaped ``(n_frames, n_tx, n_ax, n_el, 1)``.
    """
    converted_path = Path(converted_path) if converted_path is not None else None

    if converted_path is not None and converted_path.is_file() and not overwrite:
        with File(str(converted_path)) as f:
            raw = f.data.raw_data[:]
            params = f.load_parameters()
        print(f"    reusing existing converted file {converted_path.name}")
        return np.asarray(raw), params

    data_dict, scan_dict, probe_dict, custom_elements = vf.read_verasonics_file(
        buffer_index=buffer_index, allow_accumulate=True,
    )
    raw = np.asarray(data_dict["raw_data"])

    if converted_path is not None:
        converted_path.parent.mkdir(parents=True, exist_ok=True)
        if converted_path.exists():
            converted_path.unlink()
        File.create(
            str(converted_path),
            data=data_dict, scan=scan_dict, probe=probe_dict, custom=custom_elements,
            description="Verasonics data", compression=compression,
            overwrite=True, ignore_warnings=True,
        )

    params = _build_params(scan_dict, probe_dict["probe_geometry"],
                           raw.shape[2], raw.shape[3], raw.shape[1])
    return raw, params


def apply_grid(params, grid: BufferGrid):
    """Fix the beamforming grid (metres, cartesian) on a Parameters object."""
    params.n_ch = 1
    params.grid_type = "cartesian"
    params.pixels_per_wavelength = PIXELS_PER_WAVELENGTH
    params.xlims = grid.xlims
    params.zlims = grid.zlims
    return params


def plan_patches_and_chunk(params, n_tx, n_el):
    """Smallest ``num_patches`` (and frame chunk) that fit the GPU memory budget.

    GPU memory scales with the number of gathered samples,
    ``pixels x transmits x elements``. We split the grid into just enough patches
    to keep one patch under the (GPU-scaled) budget - over-patching only adds
    kernel launches - and pick a frame chunk under the same budget.
    """
    budget = memory_budget()
    pixels = int(params.grid_size_z) * int(params.grid_size_x)
    gather = pixels * n_tx * n_el
    num_patches = max(1, int(np.ceil(gather / budget)))
    per_patch = gather / num_patches
    chunk = int(np.clip(int(budget // max(per_patch, 1)), 1, 32))
    return num_patches, chunk


def build_beamform_pipeline(num_patches: int, is_baseband: bool = False) -> zea.Pipeline:
    """RF/IQ -> complex-IQ beamforming pipeline (delay-and-sum), ``(n_frames, z, x, 2)``.

    Baseband buffers (Verasonics BS100BW/BS50BW, e.g. the active shear-wave
    buffer) are *already* stored as complex IQ by the converter - two channels
    ``[I, Q]`` - so they must **not** be demodulated again; the ``Demodulate``
    step is included only for real-RF buffers (e.g. NS200BW). Feeding an IQ
    buffer through ``Demodulate`` adds a spurious axis and breaks beamforming.

    Args:
        num_patches (int): Grid patch count for the beamformer.
        is_baseband (bool): True when ``data`` is already complex IQ (channel
            dim 2); skips the demodulation step. Defaults to False (real RF).

    Returns:
        zea.Pipeline: The beamforming pipeline.
    """
    operations = [Cast(dtype="float32")]
    if not is_baseband:
        operations.append(Demodulate())
    operations.append(
        Beamform(beamformer="delay_and_sum", num_patches=num_patches, enable_pfield=False)
    )
    return zea.Pipeline(operations, with_batch_dim=True, jit_options=_jit_options())


def _ensure_cpu_t_peak(params):
    """Pre-compute the transmit-pulse peak time on the CPU and store it on ``params``.

    Works around a zea torch-GPU regression: ``Parameters.t_peak`` estimates the peak from
    ``waveforms_two_way`` via ``compute_time_to_peak_stack`` and then calls ``np.asarray`` on
    the result, which on the torch/CUDA backend is a device tensor -> ``TypeError`` (this
    breaks beamforming for the active buffer). Computing ``t_peak`` here (moving the result to
    the host) and setting it explicitly makes the property short-circuit and return the stored
    value, so beamforming runs. Idempotent; a no-op if ``t_peak`` already resolves.
    """
    try:
        _ = np.asarray(params.t_peak)      # already resolvable (e.g. no waveforms) -> nothing to do
        return
    except Exception:
        pass
    try:
        import keras
        from zea.func.ultrasound import compute_time_to_peak_stack

        wf = params._params.get("waveforms_two_way")
        if wf is None:
            wf = params._params.get("waveforms_one_way")
        if wf is not None:
            wf_t = keras.ops.convert_to_tensor(np.asarray(wf, np.float32))
            cf_t = keras.ops.convert_to_tensor(np.asarray(params.center_frequency, np.float32))
            tp = keras.ops.convert_to_numpy(compute_time_to_peak_stack(wf_t, cf_t))
            params.t_peak = np.asarray(tp, np.float32)
            return
    except Exception as exc:  # noqa: BLE001
        print(f"    (t_peak CPU pre-compute failed: {exc}; falling back to 1/fc)")
    # Fallback: default peak time = 1 / center_frequency (per transmit).
    fc = float(np.asarray(params.center_frequency).ravel()[0])
    n_tx = int(getattr(params, "n_tx_total", None) or np.asarray(params.center_frequency).size or 1)
    params.t_peak = np.full(n_tx, 1.0 / fc, np.float32)


def _is_oom_error(exc) -> bool:
    """True for an out-of-GPU-memory error on the torch or JAX backend."""
    msg = str(exc).lower()
    return any(s in msg for s in ("out of memory", "cuda oom", "resource_exhausted",
                                  "resource exhausted", "cuda error: out of memory"))


def beamform_frames(raw, params, max_oom_retries=5):
    """Beamform ``(n_frames, n_tx, n_ax, n_el, 1)`` -> IQ ``(n_frames, z, x, 2)``.

    ``num_patches``/``chunk`` come from the GPU-scaled memory budget. If a call
    still runs out of memory (a wrong estimate, fragmentation, or another process
    sharing the GPU), the patch count is doubled (and the chunk halved) and the
    block is retried - so this completes on any GPU regardless of the budget guess.
    """
    n_tx, n_el = raw.shape[1], raw.shape[3]
    # Baseband/IQ buffers arrive with a channel dim of 2 ([I, Q]) and must not be
    # demodulated again; real-RF buffers have channel dim 1. Branch the pipeline
    # accordingly (see build_beamform_pipeline).
    is_baseband = raw.shape[-1] == 2
    _ensure_cpu_t_peak(params)   # zea torch-GPU t_peak workaround (see helper docstring)
    num_patches, chunk = plan_patches_and_chunk(params, n_tx, n_el)
    pipe = build_beamform_pipeline(num_patches, is_baseband=is_baseband)
    bf_in = pipe.prepare_parameters(params)

    n_frames = raw.shape[0]
    out = []
    start = 0
    processed = 0
    retries = 0
    while start < n_frames:
        block = raw[start:start + chunk]
        try:
            iq = pipe(data=block, **bf_in)[pipe.output_key]
            iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - inspect message for OOM
            if _is_oom_error(exc) and retries < max_oom_retries:
                retries += 1
                free_gpu_memory()
                num_patches *= 2
                chunk = max(1, chunk // 2)
                pipe = build_beamform_pipeline(num_patches, is_baseband=is_baseband)
                bf_in = pipe.prepare_parameters(params)
                print(f"    OOM - retrying with num_patches={num_patches}, chunk={chunk}")
                continue
            raise
        out.append(iq)
        start += block.shape[0]
        processed += 1
        # Periodically release reserved GPU memory: over hundreds of frames the
        # torch caching allocator otherwise fragments and creeps up to an OOM.
        if processed % 16 == 0:
            free_gpu_memory()
    return np.concatenate(out, axis=0), num_patches


def _grid_coordinates(params):
    return np.asarray(params.grid, dtype=np.float32)


def _save_beamformed(path, iq, coords, fps=None, timestamps=None, extra_custom=None,
                     start_time_offset=0.0, description="", compression=DEFAULT_COMPRESSION):
    """Write an IQ stack as native ``beamformed_data`` via ``File.create``."""
    if timestamps is None and fps:
        timestamps = np.arange(iq.shape[0], dtype=np.float32) / fps
    bdata = {
        "values": iq.astype(np.float32),
        "coordinates": coords,
        "labels": np.array(["I", "Q"], dtype=np.str_),
    }
    if timestamps is not None:
        bdata["timestamps"] = timestamps.astype(np.float32)
        bdata["start_time_offset"] = np.float32(start_time_offset)
    if path.exists():
        path.unlink()
    File.create(
        str(path),
        data={"beamformed_data": bdata},
        custom=extra_custom,
        description=description,
        compression=compression,
        overwrite=True,
        ignore_warnings=True,
    )


# ---------------------------------------------------------------------------
def process_bmode_buffer(vf, spec, out_dir, grid, fps, stem, compression,
                         converted_path=None, overwrite=False):
    """Beamform a B-mode buffer straight from the workspace; save one IQ file."""
    raw, params = read_buffer(vf, spec.index, converted_path=converted_path,
                              compression=compression, overwrite=overwrite)
    if np.count_nonzero(raw) == 0:
        print(f"  buffer {spec.matlab} ({spec.name}): all zeros - skipping")
        return None

    apply_grid(params, grid)
    iq, num_patches = beamform_frames(raw, params)
    coords = _grid_coordinates(params)

    out_path = out_dir / f"{stem}_buffer{spec.matlab}_iq.hdf5"
    _save_beamformed(out_path, iq, coords, fps=fps,
                     description=f"{spec.name}: {spec.role}", compression=compression)
    print(f"  buffer {spec.matlab} ({spec.name}): {raw.shape[0]} frames n_tx={raw.shape[1]} "
          f"patches={num_patches} -> IQ {iq.shape} @ {fps or 0:.1f} FPS -> {out_path.name}")
    return out_path


def process_active_buffer(vf, spec, out_dir, grid, sw, stem, compression,
                          converted_path=None, overwrite=False):
    """Reassemble + beamform the active SW buffer; one file per measurement."""
    raw, params = read_buffer(vf, spec.index, converted_path=converted_path,
                              compression=compression, overwrite=overwrite)
    if np.count_nonzero(raw) == 0:
        print(f"  buffer {spec.matlab} ({spec.name}): all zeros - skipping")
        return None
    if sw is None:
        print(f"  buffer {spec.matlab} ({spec.name}): no SW geometry - skipping")
        return None

    tf = assemble_tracking_frames(
        raw, n_reference=sw.n_reference, n_tracking=sw.n_tracking,
        na=sw.na, harmonic=sw.harmonic, pri=sw.pri, pi_mode=spec.pi_mode,
    )
    print(f"  buffer {spec.matlab} ({spec.name}): {tf.n_meas} measurement(s), "
          f"reference {tf.reference.shape[1]} + tracking {tf.tracking.shape[1]} frames "
          f"@ {tf.prf:.0f} Hz [{tf.pi_mode}]")

    # All detect transmits share one TX geometry (the tracking beam); a recombined
    # frame is one transmit. Narrow the transmit selection rather than rewriting arrays.
    apply_grid(params, grid)
    params.set_transmits([0])
    coords = _grid_coordinates(params)

    def _beamform_block(block):
        n_meas, n_fr = block.shape[0], block.shape[1]
        flat = block.reshape((n_meas * n_fr, 1) + block.shape[2:])
        iq, _ = beamform_frames(flat, params)
        return iq.reshape((n_meas, n_fr) + iq.shape[1:])

    ref_iq = _beamform_block(tf.reference)
    trk_iq = _beamform_block(tf.tracking)

    out_paths = []
    for m in range(tf.n_meas):
        out_path = out_dir / f"{stem}_buffer{spec.matlab}_meas{m}_iq.hdf5"
        customs = [
            CustomElement(name="reference_iq", data=ref_iq[m].astype(np.float32),
                          description="pre-push reference IQ (n_ref, z, x, 2=[I,Q])",
                          unit="unitless"),
            CustomElement(name="t_reference", data=tf.t_reference.astype(np.float32),
                          description="reference frame times, s, rel. to first tracking frame",
                          unit="s"),
            # ARF push focal point (metres, imaging-plane coordinates). Persisted here
            # so downstream shear-wave processing can anchor the wave origin r0 from the
            # data instead of a hard-coded assumption. Source: SW.FocusX/Z_wavelength.
            CustomElement(name="push_focus_x", data=np.float32(sw.focus_x),
                          description="ARF push focal point, lateral x (m)", unit="m"),
            CustomElement(name="push_focus_z", data=np.float32(sw.focus_z),
                          description="ARF push focal point, axial z / depth (m)", unit="m"),
        ]
        _save_beamformed(
            out_path, trk_iq[m], coords, timestamps=tf.t_tracking, start_time_offset=0.0,
            extra_custom=customs, compression=compression,
            description=(f"{spec.name} measurement {m}: post-push tracking IQ "
                         f"(PRF {tf.prf:.0f} Hz, {tf.pi_mode}); reference IQ in custom."),
        )
        out_paths.append(out_path)
    print(f"    saved {len(out_paths)} per-measurement file(s), "
          f"tracking IQ {trk_iq.shape[1:]}, reference IQ {ref_iq.shape[1:]}")
    return out_paths


def run(mat_path, output_dir=None, buffers_matlab=None, pi_mode=None,
        compression=DEFAULT_COMPRESSION, save_converted=True, converted_dir=None,
        overwrite=False, sw_roi=None, append_params=True):
    """Beamform a whole SWI measurement to IQ HDF5 files.

    Args:
        save_converted: also write each RF buffer as a standard converted zea file
            (the product of ``VerasonicsFile.to_zea``) - needed for the group's zea
            database. Written from the same in-memory RF used for beamforming, so
            the RF is read only once.
        converted_dir: where the converted zea files go. Defaults to
            ``<output_dir>/converted``.
        overwrite: re-read/re-convert from the ``.mat`` even when a converted file
            already exists. Default False, so an existing converted file is reused
            (faster) rather than reconverted.
        sw_roi: reconstruction window for the active shear-wave buffer (2).
            ``None`` (default) = the ``SW`` struct ROI around the push focus;
            a number scales it; ``"full"`` uses the whole imaging FOV; or pass
            ``((xmin, xmax), (zmin, zmax))`` in metres. See
            :func:`swi_meta.sw_roi_grid`.
    """
    mat_path = Path(mat_path)
    output_dir = Path(output_dir) if output_dir else mat_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_dir = Path(converted_dir) if converted_dir else output_dir / "converted"
    stem = mat_path.stem

    try:
        meta = read_swi_meta(mat_path, sw_roi=sw_roi)
    except Exception as exc:
        print(f"  (could not read SWI meta: {exc}; using defaults)")
        meta = None

    print(f"=== beamforming {mat_path.name} -> {output_dir} (backend "
          f"{os.environ.get('KERAS_BACKEND')}, jit {_jit_options()}, "
          f"save_converted={save_converted}) ===")
    written = []
    with VerasonicsFile(str(mat_path)) as vf:
        present = vf.available_buffers()
        wanted = present if buffers_matlab is None else [k - 1 for k in buffers_matlab]
        for buffer_index in wanted:
            if buffer_index not in present:
                print(f"  buffer {buffer_index + 1}: not present - skipping")
                continue
            spec = SPEC_BY_INDEX.get(buffer_index)
            if spec is None:
                print(f"  buffer {buffer_index + 1}: no spec - skipping")
                continue
            if pi_mode is not None and spec.kind == "active_track":
                spec = BufferSpec(**{**spec.__dict__, "pi_mode": pi_mode})
            grid = meta.grids.get(buffer_index) if meta else None
            fps = meta.fps.get(buffer_index) if meta else None
            converted_path = (converted_dir / f"{stem}_buffer{spec.matlab}.hdf5"
                              if save_converted else None)
            try:
                if spec.kind == "active_track":
                    res = process_active_buffer(vf, spec, output_dir, grid,
                                                meta.sw if meta else None, stem, compression,
                                                converted_path=converted_path, overwrite=overwrite)
                else:
                    res = process_bmode_buffer(vf, spec, output_dir, grid, fps, stem, compression,
                                               converted_path=converted_path, overwrite=overwrite)
                if res:
                    res_list = res if isinstance(res, list) else [res]
                    written.extend(res_list)
                    # Make the shear-wave IQ files self-contained for the visualization
                    # stage: append the scan parameters (demod/tx/probe freq, sound_speed,
                    # wavelength, prf, dz, dx) as custom elements. Only the active tracking
                    # (buffer 2) and the passive source (buffer 4) feed processing, so only
                    # those are retrofitted (existing reference_iq / push_focus are preserved).
                    if append_params and (spec.kind == "active_track" or spec.passive_source):
                        from .scanparams import append_scan_params_to_iq
                        for iq_path in res_list:
                            try:
                                append_scan_params_to_iq(iq_path, converted_path=converted_path)
                            except Exception as exc:  # noqa: BLE001
                                print(f"    (could not append scan params to {iq_path.name}: {exc})")
            finally:
                free_gpu_memory()
    return written


def convert_folder(folder, output_dir=None, converted_dir=None, buffers_matlab=None,
                   overwrite=False, compression=DEFAULT_COMPRESSION):
    """Stage 1 only: convert the Verasonics ``.mat`` RF buffers to zea ``.hdf5``.

    Writes ``<output_dir>/converted/<stem>_buffer<k>.hdf5`` for each present buffer, with no
    beamforming (no GPU needed). ``beamform``/``process_folder`` reuse these files if present.
    """
    folder = Path(folder)
    mat_path = find_mat(folder)
    output_dir = Path(output_dir) if output_dir else folder / "output"
    converted_dir = Path(converted_dir) if converted_dir else output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    stem = mat_path.stem
    written = []
    print(f"=== converting {mat_path.name} -> {converted_dir} ===")
    with VerasonicsFile(str(mat_path)) as vf:
        present = vf.available_buffers()
        wanted = present if buffers_matlab is None else [k - 1 for k in buffers_matlab]
        for buffer_index in wanted:
            if buffer_index not in present:
                print(f"  buffer {buffer_index + 1}: not present - skipping")
                continue
            spec = SPEC_BY_INDEX.get(buffer_index)
            matlab = spec.matlab if spec else buffer_index + 1
            converted_path = converted_dir / f"{stem}_buffer{matlab}.hdf5"
            if converted_path.is_file() and not overwrite:
                print(f"  buffer {matlab}: converted file exists - skipping (use overwrite)")
                written.append(converted_path)
                continue
            read_buffer(vf, buffer_index, converted_path=converted_path,
                        compression=compression, overwrite=overwrite)
            print(f"  buffer {matlab}: -> {converted_path.name}")
            written.append(converted_path)
    return written


def find_mat(folder: Path) -> Path:
    """Locate the Verasonics workspace ``.mat`` inside a measurement folder.

    Prefers ``CombinedData.mat`` (the merged constant+dynamic workspace the
    beamformer reads). If only the runtime ``AcquisitionParametersAndECG.mat`` is
    present, ``CombinedData.mat`` is built from it first (base config + dynamic
    params merge; see :mod:`swp.acquisition.combined`).
    """
    folder = Path(folder)
    combined = folder / "CombinedData.mat"
    if combined.is_file():
        return combined
    if (folder / "AcquisitionParametersAndECG.mat").is_file():
        from .combined import ensure_combined_data
        return ensure_combined_data(folder)
    mats = sorted(folder.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat workspace found in {folder}")
    return mats[0]


def process_folder(folder, output_dir=None, make_gifs=True, pi_mode=None,
                   compression=DEFAULT_COMPRESSION, save_converted=True, converted_dir=None,
                   overwrite=False, sw_roi=None, append_params=True):
    """End-to-end Stage A for one measurement folder: IQ (+ GIFs) into ``output/``.

    Args:
        folder: measurement folder containing the Verasonics ``.mat`` (and any
            external ``RF_data_*.bin``).
        output_dir: where IQ and GIFs go. Defaults to ``<folder>/output``.
        make_gifs: also render the B-mode GIFs (task 2).
        save_converted: also write the converted RF zea files (for the zea
            database), into ``<output_dir>/converted`` by default.
        overwrite: reconvert from the ``.mat`` even if a converted file exists
            (default False = reuse the existing converted file).
        sw_roi: reconstruction window for the active shear-wave buffer (2);
            ``None`` = the push-focus ROI from the ``SW`` struct (default).
    """
    folder = Path(folder)
    mat_path = find_mat(folder)
    output_dir = Path(output_dir) if output_dir else folder / "output"
    init_device(verbose=True)

    written = run(mat_path, output_dir=output_dir, pi_mode=pi_mode, compression=compression,
                  save_converted=save_converted, converted_dir=converted_dir,
                  overwrite=overwrite, sw_roi=sw_roi, append_params=append_params)

    if make_gifs:
        from .gifs import run as gif_run
        gif_run(output_dir)
    return written


def _parse_sw_roi(value):
    """Parse the ``--sw-roi`` CLI value into what :func:`swi_meta.sw_roi_grid` takes.

    Accepts ``None``/omitted (default ROI), ``"full"``, a scale factor like
    ``"2"``, or an explicit window ``"xmin,xmax,zmin,zmax"`` in **mm**.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() == "full":
        return "full"
    if "," in text:
        parts = [float(v) for v in text.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "--sw-roi window must be 'xmin,xmax,zmin,zmax' in mm")
        xmin, xmax, zmin, zmax = (v * 1e-3 for v in parts)   # mm -> m
        return ((xmin, xmax), (zmin, zmax))
    return float(text)


def _cli():
    p = argparse.ArgumentParser(
        description="Stage A: beamform SWI RF to IQ HDF5 (+ B-mode GIFs) for a folder."
    )
    p.add_argument("folder", help="Measurement folder containing the Verasonics .mat")
    p.add_argument("-o", "--output", default=None, help="Output dir (default <folder>/output)")
    p.add_argument("--no-gifs", action="store_true", help="Skip GIF generation")
    p.add_argument("--no-converted", action="store_true",
                   help="Do not write the converted RF zea files (skip the database copy)")
    p.add_argument("--converted-dir", default=None,
                   help="Where to write converted RF zea files (default <output>/converted)")
    p.add_argument("--overwrite", action="store_true",
                   help="Reconvert from the .mat even if a converted file already exists")
    p.add_argument("--pi-mode", default=None, choices=["sliding", "accumulate"],
                   help="Pulse-inversion recombination for the active buffer")
    p.add_argument("--sw-roi", default=None,
                   help="Buffer-2 reconstruction window: omit for the push-focus ROI "
                        "(default), a scale factor like '2', 'full', or "
                        "'xmin,xmax,zmin,zmax' in mm")
    p.add_argument("--compression", default=DEFAULT_COMPRESSION,
                   help="HDF5 compression for IQ + converted files ('lzf', 'gzip', or 'none')")
    args = p.parse_args()
    compression = None if str(args.compression).lower() == "none" else args.compression
    process_folder(args.folder, output_dir=args.output, make_gifs=not args.no_gifs,
                   pi_mode=args.pi_mode, compression=compression,
                   save_converted=not args.no_converted, converted_dir=args.converted_dir,
                   overwrite=args.overwrite, sw_roi=_parse_sw_roi(args.sw_roi))


if __name__ == "__main__":
    _cli()
