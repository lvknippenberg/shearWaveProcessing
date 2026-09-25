"""The manual passive study (swp.manual): resumable state, staleness, locking, cropped loading."""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from swp.manual import store as S


def _ready_folder(tmp_path):
    out = tmp_path / "C1" / "SW_data_x" / "output"
    out.mkdir(parents=True)
    (out / "CombinedData_buffer4_iq.hdf5").write_bytes(b"")
    (out / "CombinedData_buffer4_iq.gif").write_bytes(b"")
    return str(out.parent)


def _general(p, pts):
    S.save_line(p.general_npz, pts)
    S.write_json(p.general_json, dict(hash=S.points_hash(pts), source_buffer=4,
                                      points_src_mm=(np.asarray(pts) * 1e3).tolist()))


def _windows(p, gen_hash, t_peaks):
    ws = [dict(t_peak=t, t0=t - 0.05, t1=t + 0.05, score=1.0, label="MVC", expect="") for t in t_peaks]
    S.write_json(p.windows_json, dict(key=dict(general_hash=gen_hash), hash=S.windows_hash(ws), windows=ws))
    return S.windows_hash(ws)


def _event(p, whash, i, pts):
    ev = S.read_json(p.events_json) or dict(windows_hash=whash, events={})
    S.save_line(p.event_npz(i), pts)
    ev["events"][str(i)] = dict(hash=S.points_hash(pts), source_buffer=4)
    S.write_json(p.events_json, ev)


def _processed(p, i, line_hash, st_hash="st1"):
    proc = S.read_json(p.processed_json, {})
    proc[str(i)] = dict(line_hash=line_hash, st_hash=st_hash)
    S.write_json(p.processed_json, proc)
    np.savez(p.st_npz(i), x=np.zeros(1))


def test_state_walks_through_every_stage(tmp_path):
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    assert S.state(f)["stage"] == "need-general"
    pts = np.array([[-0.01, 0.05], [0.0, 0.052], [0.012, 0.047]])
    _general(p, pts)
    assert S.state(f)["stage"] == "detecting"
    wh = _windows(p, S.points_hash(pts), [0.05, 0.45])
    s = S.state(f)
    assert s["stage"] == "need-events" and s["need_events"] == [0, 1]
    _event(p, wh, 0, pts)
    _event(p, wh, 1, pts + 1e-3)
    s = S.state(f)
    assert s["stage"] == "processing" and s["need_proc"] == [0, 1]
    _processed(p, 0, S.points_hash(pts))
    _processed(p, 1, S.points_hash(pts + 1e-3))
    s = S.state(f)
    assert s["stage"] == "need-slopes" and s["need_slopes"] == [0, 1]
    S.write_json(p.slopes_json, {"0": dict(st_hash="st1", confidence=3), "1": dict(st_hash="st1", skipped=True)})
    s = S.state(f)
    assert s["stage"] == "done" and s["slopes_done"] == [0] and s["slopes_skipped"] == [1]


def test_redrawn_event_line_invalidates_its_space_time_and_slope(tmp_path):
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    _general(p, pts)
    wh = _windows(p, S.points_hash(pts), [0.05])
    _event(p, wh, 0, pts)
    _processed(p, 0, S.points_hash(pts))
    S.write_json(p.slopes_json, {"0": dict(st_hash="st1", confidence=2)})
    assert S.state(f)["stage"] == "done"
    _event(p, wh, 0, pts + 2e-3)                       # redraw the line
    s = S.state(f)
    assert s["need_proc"] == [0] and s["slopes_done"] == []
    _processed(p, 0, S.points_hash(pts + 2e-3), st_hash="st2")
    assert S.state(f)["need_slopes"] == [0]            # the old slope does not count any more


def test_new_general_line_makes_windows_and_events_stale(tmp_path):
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    _general(p, pts)
    wh = _windows(p, S.points_hash(pts), [0.05])
    _event(p, wh, 0, pts)
    _general(p, pts + 1e-3)
    assert S.state(f)["stage"] == "detecting"
    _windows(p, S.points_hash(pts + 1e-3), [0.06])     # re-detected: different windows
    assert S.state(f)["need_events"] == [0]            # old event line keyed to old windows


def test_failed_detection_is_reported_not_waited_on(tmp_path):
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    _general(p, pts)
    S.write_json(os.path.join(p.dir, "worker_errors.json"),
                 dict(detect=dict(key=S.points_hash(pts), error="boom")))
    s = S.state(f)
    assert s["stage"] == "error" and s["error"] == "boom"


def test_archive_moves_downstream_files_and_keeps_the_general_line(tmp_path):
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    _general(p, pts)
    wh = _windows(p, S.points_hash(pts), [0.05])
    _event(p, wh, 0, pts)
    dest = S.archive(p, S.downstream_files(p), "test")
    assert os.path.exists(p.general_json) and not os.path.exists(p.windows_json)
    assert sorted(os.listdir(dest)) == ["event0_mline.npz", "events.json", "windows.json"]


def test_folder_lock_is_exclusive_and_released(tmp_path):
    p = S.Paths(_ready_folder(tmp_path))
    with S.folder_lock(p) as a:
        with S.folder_lock(p) as b:
            assert a and not b
    with S.folder_lock(p) as c:
        assert c


def test_legacy_folders_come_first(tmp_path):
    for sub in ("C1/A", "C2/B"):
        d = tmp_path / sub
        (d / "output" / "mlines").mkdir(parents=True)
        (d / S.RUNTIME_MAT).write_bytes(b"")
    np.savez(tmp_path / "C2/B/output/mlines/passive_general_mline.npz", points=np.zeros((2, 2)))
    fs = S.find_folders(str(tmp_path))
    assert [os.path.basename(f) for f in fs] == ["B", "A"]


def test_roi_load_equals_crop_of_full_load(tmp_path):
    from swp.viz.io import load_acquisition
    nz, nx, nt = 30, 40, 5
    x = np.linspace(-0.02, 0.02, nx)
    z = np.linspace(0.01, 0.07, nz)
    coords = np.stack(np.broadcast_arrays(x[None, :], 0 * x[None, :], z[:, None]), -1)
    vals = np.random.default_rng(0).standard_normal((nt, nz, nx, 2)).astype(np.float32)
    path = tmp_path / "b4.hdf5"
    with h5py.File(path, "w") as h:
        g = "tracks/track_0/data/beamformed_data"
        h[f"{g}/values"] = vals
        h[f"{g}/coordinates"] = coords
        h[f"{g}/timestamps"] = np.arange(nt) / 925.0
        h["custom/prf"] = 925.0
        h["custom/demodulation_frequency"] = 3.9e6
        h["custom/transmit_frequency"] = 1.95e6
        h["custom/dz"] = z[1] - z[0]
        h["custom/dx"] = x[1] - x[0]
    full = load_acquisition(str(path))
    roi = (-0.005, 0.01, 0.03, 0.05)
    part = load_acquisition(str(path), roi=roi)
    ix = (full.x >= roi[0]) & (full.x <= roi[1])
    iz = (full.z >= roi[2]) & (full.z <= roi[3])
    np.testing.assert_array_equal(part.iq, full.iq[:, iz][:, :, ix])
    np.testing.assert_array_equal(part.x, full.x[ix])
    np.testing.assert_array_equal(part.z, full.z[iz])
    assert part.coords.shape[:2] == (iz.sum(), ix.sum())


def test_display_constants_match_the_gifs():
    pytest.importorskip("zea")
    from swp.acquisition import gifs
    from swp.manual import _light
    assert (_light.DEFAULT_CURVE, _light.HI_PCT, _light.LO_PCT, _light.DR_LIMITS) == \
        (gifs.DEFAULT_CURVE, gifs.HI_PCT, gifs.LO_PCT, gifs.DR_LIMITS)


def test_manual_config_builds_five_views_with_the_literature_filters():
    from swp.viz import runconfig as rc
    cfg = rc.load_config(S.CONFIG)
    views = rc.build_views(cfg)
    names = [n for n, _ in views]
    assert names == ["displacement gauss", "velocity median", "velocity gauss", "Keijzer velocity",
                     "acceleration"]
    kz = dict(views)["Keijzer velocity"]
    assert [s.name for s in kz.iq_filters] == ["iq_slowtime_lowpass"]
    assert kz.estimator_params["kernel_shape"] == "gaussian"
    assert abs(kz.estimator_params["kernel_z_m"] - 4e-3) < 1e-12
    assert all(not v.iq_filters for n, v in views if n != "Keijzer velocity")
    assert dict(views)["acceleration"].quantity == "acceleration"
