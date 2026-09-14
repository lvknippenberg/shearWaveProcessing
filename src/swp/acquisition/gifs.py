"""Stage A / task 2: B-mode GIFs from the beamformed IQ files.

Each IQ file written by :mod:`beamform_swi` holds a ``beamformed_data`` stack
``(n_frames, z, x, 2=[I,Q])`` plus per-frame ``timestamps`` (so the true frame
rate travels with the data). We envelope-detect, log-compress and write a GIF.

Frame-rate handling (the buffers span very different rates: ~18-925 FPS B-mode,
~3.7 kHz tracking):

* **B-mode cine buffers** play in **real time** by default (``stretch=1.0``): a
  1.0 s acquisition becomes a 1.0 s GIF, whatever its frame rate. A GIF stores
  each frame's delay in whole **centiseconds**, so an arbitrary acquisition rate
  is not directly representable; :func:`realtime_gif` keeps every frame when the
  nearest legal delay reproduces the duration closely enough, and otherwise
  resamples the clip onto that delay. The ultrafast buffers hit the 20 ms floor
  and are sub-sampled (925 Hz diverging-wave -> ~50 of 926 frames); the slow ones
  repeat a frame here and there - but the *playback duration always matches the
  acquisition duration* (to ~1%). Pass ``stretch>1`` for slow motion (``3.0`` was
  the old default).
* **Active tracking** (buffer 2, ``*_meas*``) covers only ~10-30 ms - far too
  brief to watch in real time - so it is played at a fixed **15 fps** and the
  resulting slow-motion factor is reported.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import numpy as np

from zea import File
from zea.display import to_8bit
from zea.io_lib import save_video

REAL_TIME_STRETCH = 1.0      # GIF playback duration / real acquisition duration (1 = real time)
MIN_DELAY_CS = 2             # 20 ms: fastest delay GIF viewers honour (1 cs is widely clamped)
MAX_DELAY_CS = 100           # 1 s per frame
TRACKING_GIF_FPS = 15.0      # fixed rate for the (very brief) tracking movies
# Keep every frame only while the quantised delay is within this relative error of
# the true frame interval; beyond it, resample the clip onto the quantised delay.
TIMING_TOLERANCE = 0.02
DYNAMIC_RANGE = (-50, 0)     # legacy fixed window (see ``adaptive=False``)

# --- display (post-processing applied to the GIFs only; the stored IQ is untouched) ---
# The old rendering used the clip **maximum** as the white point and a fixed 50 dB window. On
# this study that left most acquisitions too dark: the maximum is a single specular reflector,
# and the tissue bulk sits far below it. Deriving both ends from the clip's own distribution -
# white point from the 99.9th percentile, range down to the noise floor - fixes that, and a
# tone curve then redistributes the mid-tones. Measured per acquisition, so a weak acoustic
# window is still rendered to full contrast rather than being crushed by a global scale.
DEFAULT_CURVE = "gamma2"     # digitised Verasonics-style S-curve; see swp.viz.tonecurves
DEFAULT_GAIN_DB = 0.0        # brightness; >0 lowers the white point and starts clipping
HI_PCT, LO_PCT = 99.9, 5.0   # white point / noise-floor percentiles, over in-sector pixels
DR_LIMITS = (25.0, 90.0)     # clamp the derived range, so a degenerate clip cannot blow it up


def iq_to_bmode(iq: np.ndarray, dynamic_range=DYNAMIC_RANGE, curve=DEFAULT_CURVE,
                gain_db=DEFAULT_GAIN_DB, adaptive=True) -> list[np.ndarray]:
    """Envelope-detect + log-compress an IQ stack into 8-bit B-mode frames.

    Args:
        iq: ``(n_frames, z, x, 2)`` array of [I, Q].
        dynamic_range: fixed ``(lo_dB, hi_dB)`` window, used only when ``adaptive`` is False.
        curve: display tone curve name (:mod:`swp.viz.tonecurves`); ``"linear"`` disables it.
        gain_db: brightness. Positive values lower the white point, brightening the image and
            allowing the brightest structures to clip.
        adaptive: derive the white point and range from this clip's own distribution
            (default). ``False`` restores the previous behaviour exactly: normalise by the
            clip maximum and apply the fixed ``dynamic_range`` with no tone curve.

    Returns:
        List of ``(z, x)`` uint8 frames. Levels are taken over the whole clip, so brightness
        is stable across frames.
    """
    envelope = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)     # (n_frames, z, x)

    if not adaptive:
        peak = envelope.max()
        if peak > 0:
            envelope = envelope / peak
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(envelope + 1e-12)
        return [to_8bit(frame, dynamic_range, pillow=False) for frame in db]

    from ..viz.tonecurves import apply_curve

    # In-sector pixels only: outside the sector the envelope is exactly 0, which would drag the
    # low percentile to zero and make the derived range meaningless.
    inside = envelope[envelope > 0]
    if inside.size == 0:                                   # empty/all-zero buffer
        return [np.zeros(envelope.shape[1:], np.uint8) for _ in range(envelope.shape[0])]
    ref = float(np.percentile(inside, HI_PCT)) / (10.0 ** (gain_db / 20.0))
    floor = float(np.percentile(inside, LO_PCT))
    ref = max(ref, 1e-12)
    dr = 20.0 * np.log10(ref / floor) if floor > 0 else DR_LIMITS[1]
    dr = float(np.clip(dr, *DR_LIMITS))

    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(envelope / ref + 1e-12)
    norm = np.clip((db + dr) / dr, 0.0, 1.0)
    return [(apply_curve(frame, curve) * 255).astype(np.uint8) for frame in norm]


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


def _save_gif_with_delay(images, out_path, delay_cs: int):
    """Write a GIF with an exact per-frame delay of ``delay_cs`` centiseconds.

    ``zea.io_lib.save_to_gif`` takes an fps and stores ``round(1000 / fps)`` ms,
    which Pillow then truncates to whole centiseconds. Passing the *float* fps
    ``100 / delay_cs`` makes that round-trip exact, so the delay we ask for is the
    delay the file carries (integer fps like 88 would silently become 1 cs, a
    delay most viewers clamp to 100 ms).
    """
    save_video(images, str(out_path), fps=100.0 / delay_cs)


def realtime_gif(images, out_path, acquisition_fps, stretch=REAL_TIME_STRETCH,
                 tolerance=TIMING_TOLERANCE):
    """Write a GIF whose playback duration is ``stretch`` x real acquisition time.

    GIF frame delays are quantised to whole centiseconds, so most acquisition
    rates cannot be reproduced exactly by simply playing every frame. Two cases:

    * the nearest legal delay is within ``tolerance`` of the true frame interval
      -> keep every frame at that delay (no resampling, full temporal detail);
    * otherwise -> resample the clip onto that delay (clamped to
      ``[MIN_DELAY_CS, MAX_DELAY_CS]``) with nearest-neighbour frame selection, so
      the playback *duration* is right to within one frame. Buffers faster than
      50 FPS land on the 20 ms floor and are sub-sampled (925 Hz -> ~50 of 926
      frames); slower ones repeat a few frames instead.

    Returns ``(gif_fps, n_shown, realtime_s, playback_s)``.
    """
    n = len(images)
    if acquisition_fps <= 0:
        _save_gif_with_delay(images, out_path, MIN_DELAY_CS)
        return 100.0 / MIN_DELAY_CS, n, 0.0, n * MIN_DELAY_CS / 100.0

    realtime_s = n / acquisition_fps
    target_s = stretch * realtime_s
    frame_dt = stretch / acquisition_fps               # wanted seconds per shown frame

    delay_cs = int(round(frame_dt * 100))
    keep_all = (MIN_DELAY_CS <= delay_cs <= MAX_DELAY_CS
                and abs(delay_cs / 100.0 - frame_dt) <= tolerance * frame_dt)

    if keep_all:
        shown = list(images)
    else:
        delay_cs = int(np.clip(delay_cs, MIN_DELAY_CS, MAX_DELAY_CS))
        n_shown = max(2, int(round(target_s * 100.0 / delay_cs)))
        idx = np.linspace(0, n - 1, n_shown).round().astype(int)
        shown = [images[i] for i in idx]

    _save_gif_with_delay(shown, out_path, delay_cs)
    return 100.0 / delay_cs, len(shown), realtime_s, len(shown) * delay_cs / 100.0


def gif_for_file(iq_path: Path, stretch=REAL_TIME_STRETCH, curve=DEFAULT_CURVE,
                 gain_db=DEFAULT_GAIN_DB, adaptive=True):
    """Make one GIF from one IQ HDF5 file."""
    is_tracking = "_meas" in iq_path.stem
    with File(str(iq_path)) as f:
        bdata = f.data.beamformed_data
        iq = np.asarray(bdata.values[:])
        acquisition_fps = _fps_from_timestamps(bdata)

    images = iq_to_bmode(iq, curve=curve, gain_db=gain_db, adaptive=adaptive)
    out_path = iq_path.with_suffix(".gif")

    if is_tracking:
        # ~16 ms of tracking: real time is unwatchable, so fix the rate and say
        # how much slower than real time the result is. Quantise to a whole
        # centisecond first, so the rate reported is the rate the file carries.
        delay_cs = int(np.clip(round(100.0 / TRACKING_GIF_FPS), MIN_DELAY_CS, MAX_DELAY_CS))
        gif_fps = 100.0 / delay_cs
        _save_gif_with_delay(images, out_path, delay_cs)
        realtime_ms = len(images) / acquisition_fps * 1e3 if acquisition_fps else 0.0
        slowdown = (acquisition_fps / gif_fps) if acquisition_fps else 0.0
        print(f"  {iq_path.name}: tracking GIF {len(images)} frames @ "
              f"{gif_fps:.4g} fps (true {realtime_ms:.1f} ms @ "
              f"{acquisition_fps:.0f} Hz = 1/{slowdown:.0f} x real time)"
              f" -> {out_path.name}")
    else:
        gif_fps, n_shown, realtime_s, playback_s = realtime_gif(
            images, out_path, acquisition_fps, stretch=stretch)
        speed = "real time" if stretch == 1.0 else f"{1 / stretch:g}x real time"
        note = (f"{acquisition_fps:.1f} Hz, {len(images)} frames = {realtime_s:.2f} s "
                f"-> {playback_s:.2f} s GIF ({speed}); "
                f"{n_shown} frames @ {gif_fps:.4g} fps")
        print(f"  {iq_path.name}: B-mode GIF, {note} -> {out_path.name}")
    return out_path


def run(iq_dir, stretch=REAL_TIME_STRETCH, curve=DEFAULT_CURVE,
        gain_db=DEFAULT_GAIN_DB, adaptive=True):
    """Make a GIF for every ``*_iq.hdf5`` file in ``iq_dir``."""
    iq_dir = Path(iq_dir)
    files = sorted(iq_dir.glob("*_iq.hdf5"))
    if not files:
        print(f"No *_iq.hdf5 files found in {iq_dir}")
        return
    speed = "real time" if stretch == 1.0 else f"{1 / stretch:g}x real time"
    disp = (f"adaptive levels, curve {curve}"
            + (f", {gain_db:+.0f} dB" if gain_db else "")) if adaptive else "legacy fixed -50..0 dB"
    print(f"=== GIFs from {len(files)} IQ file(s) in {iq_dir} ({speed}; {disp}) ===")
    for iq_path in files:
        gif_for_file(iq_path, stretch=stretch, curve=curve, gain_db=gain_db, adaptive=adaptive)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Render B-mode GIFs from *_iq.hdf5 files.")
    p.add_argument("iq_dir", help="Directory containing *_iq.hdf5 files")
    p.add_argument("--stretch", type=float, default=REAL_TIME_STRETCH,
                   help="playback duration / acquisition duration "
                        "(1 = real time, 3 = 3x slow motion)")
    p.add_argument("--curve", default=DEFAULT_CURVE,
                   help=f"display tone curve (default {DEFAULT_CURVE}); 'linear' disables it")
    p.add_argument("--gain-db", type=float, default=DEFAULT_GAIN_DB,
                   help="brightness in dB; >0 brightens and allows clipping")
    p.add_argument("--legacy-display", action="store_true",
                   help="restore the old rendering: clip-max white point, fixed -50..0 dB, "
                        "no tone curve")
    a = p.parse_args()
    run(a.iq_dir, stretch=a.stretch, curve=a.curve, gain_db=a.gain_db,
        adaptive=not a.legacy_display)
