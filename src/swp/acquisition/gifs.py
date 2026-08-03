"""Stage A / task 2: B-mode GIFs from the beamformed IQ files.

Each IQ file written by :mod:`beamform_swi` holds a ``beamformed_data`` stack
``(n_frames, z, x, 2=[I,Q])`` plus per-frame ``timestamps`` (so the true frame
rate travels with the data). We envelope-detect, log-compress and write a GIF.

Frame-rate handling (the buffers span very different rates: ~25-925 FPS B-mode,
~3.7 kHz tracking):

* **B-mode cine buffers** are played at **3x real time** - your rule that 1.2 s of
  acquisition becomes a 3.6 s GIF. Because the GIF container effectively caps at
  50 fps, buffers faster than 150 FPS (e.g. the 925 Hz diverging-wave buffer)
  keep an evenly-spaced subset of frames played at 50 fps, so the *playback
  duration* still equals 3x real time even though not every frame is shown.
* **Active tracking** (buffer 2, ``*_meas*``) covers only ~10-30 ms, far too brief
  for a 3x GIF, so it is played at a fixed **15 fps** and the true duration is
  reported.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import numpy as np

from zea import File
from zea.display import to_8bit
from zea.io_lib import save_video

REAL_TIME_STRETCH = 3.0      # GIF playback = 3x real acquisition time
GIF_FPS_CAP = 50.0           # GIF container practical max
TRACKING_GIF_FPS = 15.0      # fixed rate for the (very brief) tracking movies
DYNAMIC_RANGE = (-50, 0)


def iq_to_bmode(iq: np.ndarray, dynamic_range=DYNAMIC_RANGE) -> list[np.ndarray]:
    """Envelope-detect + log-compress an IQ stack into 8-bit B-mode frames.

    Args:
        iq: ``(n_frames, z, x, 2)`` array of [I, Q].

    Returns:
        List of ``(z, x)`` uint8 frames. Normalisation uses the whole clip's
        maximum so brightness is stable across frames.
    """
    envelope = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)     # (n_frames, z, x)
    peak = envelope.max()
    if peak > 0:
        envelope = envelope / peak
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(envelope + 1e-12)
    return [to_8bit(frame, dynamic_range, pillow=False) for frame in db]


def _fps_from_timestamps(bdata, fallback=25.0) -> float:
    """Recover the acquisition frame rate from per-frame timestamps."""
    ts = getattr(bdata, "timestamps", None)
    if ts is None:
        return fallback
    ts = np.asarray(ts)
    if ts.size < 2:
        return fallback
    dt = np.median(np.diff(ts))
    return float(1.0 / dt) if dt > 0 else fallback


def realtime_gif(images, out_path, acquisition_fps, stretch=REAL_TIME_STRETCH,
                 fps_cap=GIF_FPS_CAP):
    """Write a GIF whose playback duration is ``stretch`` x real acquisition time.

    Returns ``(gif_fps, n_shown, realtime_s)``.
    """
    n = len(images)
    realtime_s = n / acquisition_fps if acquisition_fps > 0 else 0.0
    playback_s = stretch * realtime_s
    gif_fps = n / playback_s if playback_s > 0 else fps_cap

    if gif_fps <= fps_cap:
        save_video(images, str(out_path), fps=max(1, round(gif_fps)))
        return gif_fps, n, realtime_s

    # Too many frames for the cap: keep an evenly-spaced subset, play at the cap
    # so total playback (n_shown / cap) still equals the target playback_s.
    n_shown = max(2, int(round(playback_s * fps_cap)))
    idx = np.linspace(0, n - 1, n_shown).round().astype(int)
    save_video([images[i] for i in idx], str(out_path), fps=round(fps_cap))
    return fps_cap, n_shown, realtime_s


def gif_for_file(iq_path: Path):
    """Make one GIF from one IQ HDF5 file."""
    is_tracking = "_meas" in iq_path.stem
    with File(str(iq_path)) as f:
        bdata = f.data.beamformed_data
        iq = np.asarray(bdata.values[:])
        acquisition_fps = _fps_from_timestamps(bdata)

    images = iq_to_bmode(iq)
    out_path = iq_path.with_suffix(".gif")

    if is_tracking:
        save_video(images, str(out_path), fps=round(TRACKING_GIF_FPS))
        realtime_ms = len(images) / acquisition_fps * 1e3 if acquisition_fps else 0.0
        print(f"  {iq_path.name}: tracking GIF {len(images)} frames @ "
              f"{TRACKING_GIF_FPS:.0f} fps (true {realtime_ms:.0f} ms @ "
              f"{acquisition_fps:.0f} Hz) -> {out_path.name}")
    else:
        gif_fps, n_shown, realtime_s = realtime_gif(images, out_path, acquisition_fps)
        note = (f"{realtime_s * 1e3:.0f} ms real -> {REAL_TIME_STRETCH:.0f}x at "
                f"{acquisition_fps:.0f} Hz")
        if n_shown < len(images):
            note += f" (showing {n_shown}/{len(images)} @ {gif_fps:.0f} fps)"
        print(f"  {iq_path.name}: B-mode GIF, {note} -> {out_path.name}")
    return out_path


def run(iq_dir):
    """Make a GIF for every ``*_iq.hdf5`` file in ``iq_dir``."""
    iq_dir = Path(iq_dir)
    files = sorted(iq_dir.glob("*_iq.hdf5"))
    if not files:
        print(f"No *_iq.hdf5 files found in {iq_dir}")
        return
    print(f"=== GIFs from {len(files)} IQ file(s) in {iq_dir} ===")
    for iq_path in files:
        gif_for_file(iq_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Render B-mode GIFs from *_iq.hdf5 files.")
    p.add_argument("iq_dir", help="Directory containing *_iq.hdf5 files")
    run(p.parse_args().iq_dir)
