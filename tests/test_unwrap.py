"""Buffer-3 unwrap (swp.acquisition.unwrap): file rewriting, flags and idempotency (synthetic files)."""
from __future__ import annotations

import os
import time

import h5py
import numpy as np
import pytest

pytest.importorskip("zea")
from swp.acquisition import unwrap as U          # noqa: E402

N = 8
FIRST = 3                                        # stored slot 3 was acquired first


def _files(tmp_path, stem="CombinedData"):
    out = tmp_path / "output"
    (out / "converted").mkdir(parents=True)
    conv = out / "converted" / f"{stem}_buffer3.hdf5"
    iq = out / f"{stem}_buffer3_iq.hdf5"
    # frame k of the CHRONOLOGICAL recording carries value k; stored slot q holds (q - FIRST) mod N
    stored = (np.arange(N) - FIRST) % N
    with h5py.File(conv, "w") as f:
        f.attrs["root_attr"] = "kept"
        f.create_dataset(U._RAW, data=np.broadcast_to(stored[:, None, None, None, None], (N, 2, 5, 4, 1)).astype(np.int16),
                         chunks=(1, 2, 5, 4, 1), compression="lzf")
        f["probe/name"] = "custom"
        f["probe/name"].attrs["unit"] = "-"
    with h5py.File(iq, "w") as f:
        f.create_dataset(f"{U._BF}/values", data=np.broadcast_to(stored[:, None, None, None], (N, 3, 4, 2)).astype(np.float32))
        f.create_dataset(f"{U._BF}/timestamps", data=(np.arange(N) * 0.039).astype(np.float32))
        f.create_dataset(f"{U._BF}/coordinates", data=np.zeros((3, 4, 3), np.float32))
    return out, conv, iq


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def estimate(folder, output_dir=None, current_head=0):
        calls.append(folder)
        return U.Estimate("unwrapped", FIRST, -1, "test", {"n_frames": N})

    monkeypatch.setattr(U, "estimate", estimate)
    monkeypatch.setattr(U, "chronological_times",
                        lambda folder, n, shift: (1000.0 + 39.42 * np.arange(n), 20.0 + 39.42 * np.arange(n)))
    return calls


def _frames(conv, iq):
    with h5py.File(conv, "r") as f:
        c = f[U._RAW][:, 0, 0, 0, 0]
    with h5py.File(iq, "r") as f:
        i = f[f"{U._BF}/values"][:, 0, 0, 0]
    return c, i


def test_reorders_both_files_chronologically(tmp_path, fake):
    out, conv, iq = _files(tmp_path)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    c, i = _frames(conv, iq)
    np.testing.assert_array_equal(c, np.arange(N))
    np.testing.assert_array_equal(i, np.arange(N))
    for p in (conv, iq):
        rec = U.read_flag(p)
        assert rec["buffer_unwrapped"] and rec["status"] == "unwrapped" and rec["first_frame"] == FIRST
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(f["custom/frame_time_ms"][:2], [1000.0, 1039.42])
    with h5py.File(conv, "r") as f:
        assert f.attrs["root_attr"] == "kept" and f["probe/name"].attrs["unit"] == "-"
        assert f[U._RAW].compression == "lzf" and f[U._RAW].chunks == (1, 2, 5, 4, 1)
    with h5py.File(iq, "r") as f:
        np.testing.assert_allclose(f[f"{U._BF}/timestamps"][:3], [0, 0.03942, 0.07884], rtol=1e-5)
    assert not list(out.rglob("*.unwrap_tmp"))


def _variant(out, name="CombinedData_buffer3_refocus-adjoint_iq.hdf5"):
    stored = (np.arange(N) - FIRST) % N
    path = out / name
    with h5py.File(path, "w") as f:
        f.create_dataset(f"{U._BF}/values", data=np.broadcast_to(stored[:, None, None, None], (N, 3, 4, 2)).astype(np.float32))
        f.create_dataset(f"{U._BF}/timestamps", data=(np.arange(N) * 0.039).astype(np.float32))
    return path


def test_variants_get_the_same_head(tmp_path, fake):
    out, conv, iq = _files(tmp_path)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    var = _variant(out)                              # e.g. a REFoCUS reconstruction made earlier
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    with h5py.File(var, "r") as f:
        np.testing.assert_array_equal(f[f"{U._BF}/values"][:, 0, 0, 0], np.arange(N))
    assert U.read_flag(var)["first_frame"] == FIRST
    assert len(fake) == 1                            # head reused, not re-estimated
    assert U.unwrap_variants(str(tmp_path), out, gif=False, log=lambda *_: None) == []   # idempotent


def test_second_run_moves_nothing(tmp_path, fake):
    out, conv, iq = _files(tmp_path)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    c, i = _frames(conv, iq)
    np.testing.assert_array_equal(c, np.arange(N))
    np.testing.assert_array_equal(i, np.arange(N))
    assert len(fake) == 1                          # estimated once


def test_reconverted_rf_is_realigned_with_the_stored_head(tmp_path, fake):
    out, conv, iq = _files(tmp_path)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    os.remove(conv)                                # reconverted from the .mat: stored order again
    _, conv2, _ = _files(tmp_path / "again")
    os.replace(conv2, conv)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    c, i = _frames(conv, iq)
    np.testing.assert_array_equal(c, np.arange(N))
    np.testing.assert_array_equal(i, np.arange(N))
    assert len(fake) == 1                          # the stored head was reused, not re-estimated


def test_iq_rebeamformed_from_unwrapped_rf_is_not_moved_again(tmp_path, fake):
    out, conv, iq = _files(tmp_path)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    time.sleep(1.1)
    with h5py.File(iq, "w") as f:                  # re-beamformed from the chronological RF
        f.create_dataset(f"{U._BF}/values", data=np.broadcast_to(np.arange(N)[:, None, None, None], (N, 3, 4, 2)).astype(np.float32))
        f.create_dataset(f"{U._BF}/timestamps", data=(np.arange(N) * 0.039).astype(np.float32))
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    _, i = _frames(conv, iq)
    np.testing.assert_array_equal(i, np.arange(N))
    assert U.read_flag(iq)["first_frame"] == FIRST


def test_ambiguous_leaves_order_and_flags_false(tmp_path, monkeypatch):
    out, conv, iq = _files(tmp_path)
    monkeypatch.setattr(U, "estimate", lambda folder, output_dir=None, current_head=0: U.Estimate("ambiguous", details={}))
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    c, i = _frames(conv, iq)
    stored = (np.arange(N) - FIRST) % N
    np.testing.assert_array_equal(c, stored)
    np.testing.assert_array_equal(i, stored)
    rec = U.read_flag(iq)
    assert rec["status"] == "ambiguous" and not rec["buffer_unwrapped"] and rec["first_frame"] == -1


def _fake_log(monkeypatch, n_before, run_length, n=26, fms=39.42, gap_b1=45.0):
    """Full sequence: [earlier triggers, pause] -> live loop -> buffer 1 -> wait R -> buffer 4 -> SW."""
    from swp.acquisition import triggerlog
    f1, f4, n1, n4 = 1000 / 88.18, 1000 / 925.9, 90, 926
    early = np.arange(n_before) * 11.34                       # another buffer, then a long pause
    run = early[-1] + 4400.0 + np.arange(run_length) * fms if n_before else np.arange(run_length) * fms
    b1 = run[-1] + gap_b1 + np.arange(n1) * f1                # buffer 1 right after the loop
    b4 = b1[-1] + 400.0 + np.arange(n4) * f4
    sw = b4[-1] + 500.0 + np.cumsum(np.tile([1.0, 17.0, 17.0, 13.0], 20))
    trig = np.concatenate([early, run, b1, b4, sw])
    params = {1: dict(n=n1, fps=1000 / f1), 3: dict(n=n, fps=1000.0 / fms), 4: dict(n=n4, fps=1000 / f4)}
    monkeypatch.setattr(triggerlog, "read_log", lambda folder: (np.array([0.0, 900.0]), trig, params))
    return trig


def test_trigger_count_head(monkeypatch):
    _fake_log(monkeypatch, n_before=50, run_length=319)
    tc = U.trigger_count_head("x")
    assert tc["run_length"] == 319 and tc["c"] == (319 - 1) % 26 == 6


def test_trigger_count_needs_the_start_of_the_run(monkeypatch):
    _fake_log(monkeypatch, n_before=0, run_length=900)         # log starts inside the run
    assert U.trigger_count_head("x") is None


def test_buffer1_trigger_is_not_counted_as_a_loop_frame(monkeypatch):
    """Regression (2026-09-28): buffer 1 starting one loop period after the last loop trigger was
    absorbed into the loop run (period-only segmentation) -> run one too long, times one frame late."""
    from swp.acquisition import triggerlog
    trig = _fake_log(monkeypatch, n_before=50, run_length=319, gap_b1=39.42)
    tc = U.trigger_count_head("x")
    assert tc["run_length"] == 319 and tc["c"] == 6
    b3, b1 = triggerlog.buffer_timing("x", 3), triggerlog.buffer_timing("x", 1)
    loop_last = trig[50 + 319 - 1]
    assert b3.frame_times()[-1] == pytest.approx(loop_last)    # last of the loop, not buffer 1's first
    assert b1.t0_ms == pytest.approx(loop_last + 39.42)
    assert triggerlog.buffer_timing("x", 4).n_frames == 926


def test_best_and_margin_ignores_adjacent_candidates():
    v = np.array([0.9, 0.2, 0.25, 0.9, 0.6, 0.9])
    i, m = U.best_and_margin(v, lower_is_better=True)
    assert i == 1 and m == pytest.approx(0.4)      # slot 2 is adjacent; next is slot 4 (0.6)


def test_continuity_finds_a_synthetic_break():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((20, 50))
    walk = np.cumsum(rng.standard_normal((20, 50)) * 0.15, axis=0) + base[0]     # slowly changing frames
    stored = np.roll(walk, -7, axis=0)                                           # head at slot 13
    A = stored - stored.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    first, _ = U.best_and_margin(U.continuity_scores(A), lower_is_better=True)
    assert first == 13


def test_us_counter_overflow_is_unwrapped_before_sorting():
    """Regression (2026-09-28): the µs trigger timestamps are a 32-bit counter; a log spanning its
    overflow was sorted into the wrong order (C000000031 12-09-49)."""
    from swp.acquisition import triggerlog
    W = triggerlog.COUNTER_WRAP_US
    trig = (W - 3000.0) + np.arange(10) * 1000.0                 # crosses the overflow after 3 entries
    trig = np.where(trig >= W, trig - W, trig)                    # as logged
    r = np.array([W - 2500.0, 1500.0])                            # one R-peak on each side of it
    r2, t2 = triggerlog._unwrap_us_counter(r, trig)
    assert np.all(np.diff(t2) == 1000.0)
    assert np.all(np.diff(r2) > 0) and t2[0] < r2[0] < r2[1] < t2[-1]


def _mark_v1(*paths):
    for p in paths:
        with h5py.File(p, "a") as f:
            del f["custom/unwrap_version"]
            f["custom/unwrap_version"] = 1


def test_v1_unwrap_is_redone_from_its_current_head(tmp_path, fake, monkeypatch):
    """VERSION 2 (2026-09-28): a v1 head that was one slot off is re-estimated from the stored order
    (the estimator is told the file's current head) and each file is rotated by the difference."""
    out, conv, iq = _files(tmp_path)
    var = _variant(out)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)      # head FIRST
    _mark_v1(conv, iq, var)
    seen = []

    def estimate2(folder, output_dir=None, current_head=0):
        seen.append(current_head)
        return U.Estimate("unwrapped", (FIRST - 1) % N, -1, "test", {"n_frames": N})
    monkeypatch.setattr(U, "estimate", estimate2)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    assert seen == [FIRST]
    want = (np.arange(N) - 1) % N                 # frame i = stored slot FIRST-1+i = chronological i-1
    c, i = _frames(conv, iq)
    np.testing.assert_array_equal(c, want)
    np.testing.assert_array_equal(i, want)
    with h5py.File(var, "r") as f:
        np.testing.assert_array_equal(f[f"{U._BF}/values"][:, 0, 0, 0], want)
    assert all(U.read_flag(p)["version"] == U.VERSION and U.read_flag(p)["first_frame"] == (FIRST - 1) % N
               for p in (conv, iq, var))
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)      # v2: idempotent
    assert seen == [FIRST]


def test_v1_resolved_but_now_ambiguous_returns_to_stored_order(tmp_path, fake, monkeypatch):
    out, conv, iq = _files(tmp_path)
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    _mark_v1(conv, iq)
    monkeypatch.setattr(U, "estimate", lambda folder, output_dir=None, current_head=0:
                        U.Estimate("ambiguous", details={"n_frames": N}))
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    c, i = _frames(conv, iq)
    stored = (np.arange(N) - FIRST) % N
    np.testing.assert_array_equal(c, stored)
    np.testing.assert_array_equal(i, stored)
    assert not U.read_flag(iq)["buffer_unwrapped"]
