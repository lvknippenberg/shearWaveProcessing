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

    def estimate(folder, output_dir=None):
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
    monkeypatch.setattr(U, "estimate", lambda folder, output_dir=None: U.Estimate("ambiguous", details={}))
    U.unwrap_buffer3(str(tmp_path), out, gif=False, log=lambda *_: None)
    c, i = _frames(conv, iq)
    stored = (np.arange(N) - FIRST) % N
    np.testing.assert_array_equal(c, stored)
    np.testing.assert_array_equal(i, stored)
    rec = U.read_flag(iq)
    assert rec["status"] == "ambiguous" and not rec["buffer_unwrapped"] and rec["first_frame"] == -1


def _fake_log(monkeypatch, n_before, run_length, n=26, fms=39.42):
    from swp.acquisition import triggerlog
    early = np.arange(n_before) * 11.34                       # another buffer, then a long pause
    run = early[-1] + 4400.0 + np.arange(run_length) * fms if n_before else np.arange(run_length) * fms
    after = run[-1] + 500.0 + np.arange(100) * 11.34          # buffer 1 after the loop
    trig = np.concatenate([early, run, after])
    monkeypatch.setattr(triggerlog, "read_log", lambda folder: (np.array([0.0, 900.0]), trig,
                                                                {3: dict(n=n, fps=1000.0 / fms)}))


def test_trigger_count_head(monkeypatch):
    _fake_log(monkeypatch, n_before=50, run_length=319)
    tc = U.trigger_count_head("x")
    assert tc["run_length"] == 319 and tc["c"] == (319 - 1) % 26 == 6


def test_trigger_count_needs_the_start_of_the_run(monkeypatch):
    _fake_log(monkeypatch, n_before=0, run_length=900)         # log starts inside the run
    assert U.trigger_count_head("x") is None


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
