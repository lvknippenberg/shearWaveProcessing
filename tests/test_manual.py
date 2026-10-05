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


def test_excluded_measurement_is_never_offered_again(tmp_path):
    from swp.manual.session import Session
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.0, 0.052], [0.012, 0.047]])
    _general(p, pts)
    _windows(p, S.points_hash(pts), [0.05])
    ses = Session([f], retry_skipped=True)
    ses._save_general(p, None, dict(action="exclude"), {})        # v on the general-line prompt
    assert S.state(f)["stage"] == "excluded"
    assert S.read_json(p.general_json)["excluded"] == "not PLAX"
    assert not os.path.exists(p.windows_json)                     # downstream archived, not deleted
    assert any(n.startswith("archive_") and n.endswith("general_excluded") for n in os.listdir(p.dir))
    ses.refresh()
    assert ses.next_task() is None                                # even with --retry-skipped
    ses._save_general(p, None, dict(action="skip"), {})           # x: no septum -> retried
    ses.refresh()
    assert S.state(f)["stage"] == "skipped" and ses.next_task() == ("general", f, None)


def test_manual_view_label_filters_non_plax_folders(tmp_path, monkeypatch):
    from swp.manual.session import Session
    f = _ready_folder(tmp_path)                                   # C1/SW_data_x
    csv = tmp_path / "views.csv"
    monkeypatch.setattr(S, "VIEWS_MANUAL_CSV", csv)
    for label, stage in (("PSAX", "not-plax"), ("Apical", "not-plax"), ("PLAX", "need-general"),
                         ("Unclear", "need-general")):
        csv.write_text(f"subject,folder,label\nC1,SW_data_x,{label}\n")
        S._VIEWS.clear()
        assert S.state(f)["stage"] == stage, label
        assert S.view_hint(f) == f"{label} (manual review)"
    csv.write_text("subject,folder,label\nC1,SW_data_x,PSAX\n")
    S._VIEWS.clear()
    assert S.state(f, view_filter=False)["stage"] == "need-general"
    assert Session([f]).next_task() is None
    assert Session([f], view_filter=False).next_task() == ("general", f, None)
    assert not os.path.exists(S.Paths(f).dir)                     # nothing written for a filtered folder
    S._VIEWS.clear()


def test_mvc_near_r_peak_reuses_the_general_line_without_a_prompt(tmp_path):
    from swp.manual.session import Session
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.0, 0.052], [0.012, 0.047]])
    _general(p, pts)
    wh = _windows(p, S.points_hash(pts), [0.05, 0.45])
    w = S.read_json(p.windows_json)["windows"][0]
    info = dict(label="MVC", phase_ms=31.0, shift_mm=0.4, frame_event=12, reliable=True, why="ok")
    ses = Session([f], reuse=False)
    act = ses.prompt(("event", f, 0), dict(auto_reuse=info, points4_mm=pts * 1e3, window=w, phase={},
                                           preload=dict(source=4, what="general line, reused")))
    assert act == "accept"                                        # no editor was opened
    e = S.read_json(p.events_json)
    assert e["windows_hash"] == wh and e["events"]["0"]["hash"] == S.points_hash(pts)
    assert e["events"]["0"]["auto_reuse"]["shift_mm"] == 0.4
    s = S.state(f)
    assert s["need_proc"] == [0] and s["need_events"] == [1]      # the worker takes it from here


def test_reuse_gate_on_label_and_phase(tmp_path):
    from swp.manual.session import reuse_general
    cfg = dict(labels=["MVC"], max_phase_ms=50, max_perp_mm=1.0, max_shift_mm=3.0)
    p = S.Paths(_ready_folder(tmp_path))                          # no images: must not get that far
    assert not reuse_general(p.folder, p, dict(label="AVC", t_peak=0.3), dict(phase_ms=20), cfg)[0]
    assert not reuse_general(p.folder, p, dict(label="MVC", t_peak=0.1), dict(phase_ms=62), cfg)[0]
    assert not reuse_general(p.folder, p, dict(label="MVC", t_peak=0.1), dict(phase_ms=None), cfg)[0]


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


def test_folder_lock_of_a_killed_session_is_released_at_once(tmp_path):
    import socket
    import subprocess
    import sys
    p = S.Paths(_ready_folder(tmp_path))
    os.makedirs(p.dir, exist_ok=True)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    with open(p.lock, "w") as fh:                        # this host, process no longer running
        fh.write(f"{socket.gethostname()} {gone.pid} now")
    with S.folder_lock(p) as got:
        assert got
    with open(p.lock, "w") as fh:                        # another host: only age releases it
        fh.write(f"elsewhere-{socket.gethostname()} {gone.pid} now")
    with S.folder_lock(p) as got:
        assert not got


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


# ------------------------------------------------------------------ ROI marking (passive_roi.py)
def _roi_editor(preload=None, click_ms=None):
    import matplotlib
    matplotlib.use("Agg")
    from swp.manual.roi_gui import RoiEditor
    t = np.arange(0, 1.0, 1 / 925.9)
    r = np.linspace(0, 0.04, 50)
    v = np.random.default_rng(0).standard_normal((t.size, r.size)) * 0.01
    data = dict(v=v, t_s=t, r_m=r, r_peaks_s=np.array([-0.7, 0.0, 0.7, 1.4]), rr_s=0.7)
    return RoiEditor(data, "test", preload=preload, maximize=False, click_ms=click_ms)


def _ev(ed, x, y=20.0, button=1, ax=None, key=None):
    from types import SimpleNamespace
    return SimpleNamespace(inaxes=ax or ed.ax, xdata=x, ydata=y, button=button, key=key)


def _drag(ed, x0, x1, **kw):
    ed.on_press(_ev(ed, x0, **kw))
    ed.on_motion(_ev(ed, (x0 + x1) / 2, **kw))
    ed.on_motion(_ev(ed, x1, **kw))
    ed.on_release(_ev(ed, x1, **kw))


def test_expected_windows_match_phase_targets():
    pytest.importorskip("zea")
    from swp.passive_valves import expected_windows
    from swp.mline.select import phase_targets
    rp = np.array([0.1, 0.8, 1.5])
    assert expected_windows(rp, 0.7) == phase_targets(rp, 0.7)


def test_roi_editor_draws_moves_labels_and_deletes():
    ed = _roi_editor()
    _drag(ed, 20.0, 120.0)                       # inside MVC R+0-150 ms
    _drag(ed, 400.0, 300.0, y=30.0)              # right-to-left; the AVC window of HR 86 (QS2 ~366 ms)
    _drag(ed, 600.0, 602.0)                      # shorter than MIN_ROI_MS: ignored
    assert [(round(q["t0"]), round(q["t1"]), q["label"]) for q in ed.rois] == \
        [(20, 120, "MVC"), (300, 400, "AVC")]
    _drag(ed, 120.0, 160.0)                      # grab the right end of ROI 1
    assert round(ed.rois[0]["t1"]) == 160 and len(ed.rois) == 2
    ed.on_key(_ev(ed, 350.0, key="l"))           # AVC -> AK
    assert ed.rois[1]["label"] == "AK"
    _drag(ed, 350.0, 370.0)                      # move it: a hand-set label is kept
    assert (round(ed.rois[1]["t0"]), ed.rois[1]["label"]) == (320, "AK")
    ed.on_press(_ev(ed, 50.0, button=3))         # right-click deletes ROI 1
    assert len(ed.rois) == 1
    rec = ed.records()[0]
    assert rec["label"] == "AK" and rec["suggested_label"] == "AVC" and rec["expected"] == "AVC"
    assert abs(rec["t0"] - 0.32) < 1e-6 and abs(rec["phase_ms"] - 370.0) < 0.5


def test_roi_editor_click_places_a_fixed_window():
    ed = _roi_editor()
    _drag(ed, 300.0, 300.0)                      # without click_ms a click draws nothing
    assert ed.rois == []
    ed = _roi_editor(click_ms=120.0)
    _drag(ed, 360.0, 361.0)                      # a click: 120 ms centred on it
    _drag(ed, 10.0, 10.0)                        # near the record start: shifted inside
    assert [(round(q["t0"]), round(q["t1"]), q["label"]) for q in ed.rois] ==         [(round(ed.full_xlim[0]), round(ed.full_xlim[0]) + 120, "MVC"), (300, 420, "AVC")]
    _drag(ed, 330.0, 345.0)                      # a click on a ROI still moves it, no new window
    assert len(ed.rois) == 2 and round(ed.rois[1]["t0"]) == 315


def test_roi_editor_accept_rules_and_preload_round_trip():
    ed = _roi_editor()
    ed.fig.canvas.stop_event_loop = lambda: None
    ed.on_key(_ev(ed, None, key="enter"))        # nothing drawn: not accepted
    assert ed.result is None
    ed.on_key(_ev(ed, None, key="n"))
    assert ed.result == dict(action="none", rois=[], clim_pct=99.5)
    ed2 = _roi_editor()
    ed2.fig.canvas.stop_event_loop = lambda: None
    _drag(ed2, 700.0, 780.0)
    ed2.on_key(_ev(ed2, None, key="up"))
    ed2.on_key(_ev(ed2, None, key="enter"))
    res = ed2.result
    assert res["action"] == "accept" and res["clim_pct"] == 99.8 and res["rois"][0]["label"] == "MVC"
    ed3 = _roi_editor(preload=res)
    assert ed3.records() == res["rois"] and ed3.clim_i == ed2.clim_i


def test_rois_are_keyed_to_the_general_line(tmp_path):
    import passive_roi as PR
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[0.0, 0.03], [0.0, 0.07]])
    _general(p, pts)
    h = S.points_hash(pts)
    assert PR._state(f, h)[0] == "todo"
    assert PR._state(f, "other")[0] == "stale-cache"
    S.write_json(p.rois_json, dict(general_hash=h, status="none", rois=[]))
    assert PR._state(f, h)[0] == "none"
    _general(p, pts + 1e-3)                      # redrawn general line: the cache and the ROIs are stale
    assert PR._state(f, h)[0] == "stale-cache"
    assert "rois.json" in S.downstream_files(p)  # archived with --redo general


# ------------------------------------------------------------------ automatic MVC / AVC windows
def _bursts(t_bursts, t_end=1.0, prf=925.9, c=3.0, seed=1):
    """Velocity space-time (n_t, n_r) with short propagating bursts (c m/s) at ``t_bursts`` plus noise."""
    t = np.arange(0.002, t_end, 1 / prf)
    r = np.linspace(0, 0.035, 60)
    v = np.random.default_rng(seed).standard_normal((t.size, r.size)) * 1e-3
    for tb in t_bursts:
        tau = t[:, None] - tb - (r[None, :] - r.mean()) / c
        v += 0.02 * np.exp(-0.5 * (tau / 0.006) ** 2) * np.cos(2 * np.pi * 60 * tau)
    return v, r, t


def test_valve_windows_centre_on_the_bursts_and_screen_empty_ones():
    from swp.passive_valves import valve_windows
    rr = 0.7                                       # HR 86: QS2 ~366 ms, AVC search 246-486 ms
    v, r, t = _bursts([0.05, 0.33, 0.75])          # MVC, AVC, MVC of the next beat; no 2nd AVC
    ws = valve_windows(v, r, t, [0.0, 0.7, 1.4], rr)
    got = [(w["label"], round(w["t_burst"], 3), w["screened"]) for w in ws]
    assert [g[0] for g in got] == ["MVC", "AVC", "MVC"]            # the 2nd AVC is < half inside
    assert all(abs(g[1] - tb) < 0.004 and not g[2] for g, tb in zip(got, (0.05, 0.33, 0.75)))
    for w in ws:
        assert abs(w["t1"] - w["t0"] - 0.12) < 1e-9 and w["t0"] >= t[0] - 1e-9 and w["t1"] <= t[-1] + 1e-9
        assert w["t0"] < w["t_burst"] < w["t1"]
    v2, _, _ = _bursts([0.05, 0.75])               # no AVC burst at all
    avc = [w for w in valve_windows(v2, r, t, [0.0, 0.7, 1.4], rr) if w["label"] == "AVC"]
    assert len(avc) == 1 and avc[0]["screened"]
    assert valve_windows(v, r, t, [], None) == []  # no ECG, no windows


# ------------------------------------------------------------------ window review (detector "valves")
def _valve_windows(p, gen_hash, spans, screened=()):
    ws = [dict(t_peak=(a + b) / 2, t0=a, t1=b, label=lab, expect=lab, screen=0.2 if k in screened else 0.7,
               screened=k in screened) for k, (a, b, lab) in enumerate(spans)]
    S.write_json(p.windows_json, dict(key=dict(general_hash=gen_hash, detect=dict(picker="valves")),
                                      needs_review=True, hash=S.windows_hash(ws), windows=ws))
    v, r, t = _bursts([(a + b) / 2 for a, b, _ in spans])
    np.savez_compressed(p.general_st, v=v.astype(np.float32), r=r, t=t, r_peaks_s=np.array([0.0, 0.7]),
                        rr_s=np.array(0.7), general_hash=np.array(gen_hash), view=np.array("velocity gauss"))
    return S.windows_hash(ws)


def _review(p, whash, records, status="done"):
    from swp.manual.session import review_hash, review_windows
    z = np.load(p.general_st)
    ws = review_windows(records, dict(v=z["v"], t_s=z["t"]))
    S.write_json(p.review_json, dict(windows_hash=whash, hash=review_hash(ws), status=status, windows=ws,
                                     phases=[{}] * len(ws)))
    return review_hash(ws)


def test_valves_windows_wait_for_the_review_and_the_review_keys_the_event_lines(tmp_path):
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    _general(p, pts)
    wh = _valve_windows(p, S.points_hash(pts), [(0.0, 0.12, "MVC"), (0.27, 0.39, "AVC"), (0.75, 0.87, "AVC")],
                        screened=(2,))
    s = S.state(f)
    assert s["stage"] == "need-review" and s["n_windows"] == 3
    rh = _review(p, wh, [dict(t0=0.0, t1=0.12, label="MVC"), dict(t0=0.30, t1=0.42, label="AVC")])
    s = S.state(f)
    assert s["stage"] == "need-events" and s["need_events"] == [0, 1] and s["screened"] == []
    ew = S.event_windows(p)
    assert ew["reviewed"] and ew["hash"] == rh and abs(ew["windows"][1]["t_peak"] - 0.33) < 0.004
    _event(p, rh, 0, pts)
    assert S.state(f)["need_events"] == [1]
    _review(p, wh, [dict(t0=0.0, t1=0.12, label="MVC"), dict(t0=0.31, t1=0.43, label="AVC")])  # moved
    assert S.state(f)["need_events"] == [0, 1]              # every event line is now stale
    _review(p, wh, [], status="skipped")
    s = S.state(f)
    assert s["stage"] == "no-windows" and s["review_skipped"]
    _valve_windows(p, S.points_hash(pts), [(0.0, 0.12, "MVC")])   # re-detected: the review is stale
    assert S.state(f)["stage"] == "need-review"


def test_review_proposals_prefer_the_redo_then_the_rois_then_the_automatic_windows(tmp_path):
    from swp.manual.session import load_review
    f = _ready_folder(tmp_path)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    _general(p, pts)
    wh = _valve_windows(p, S.points_hash(pts), [(0.0, 0.12, "MVC"), (0.27, 0.39, "AVC"), (0.75, 0.87, "AVC")],
                        screened=(2,))
    d = load_review(p)
    assert d["preload_what"] == "automatic windows" and len(d["preload"]["rois"]) == 2
    assert [h["text"] for h in d["hints"]] == ["AVC 0.20"]
    S.write_json(p.rois_json, dict(general_hash=S.points_hash(pts), status="done", rois=[
        dict(t0=0.003, t1=0.063, label="MVC"), dict(t0=0.30, t1=0.38, label="AVC"),
        dict(t0=0.60, t1=0.68, label="other")]))
    d = load_review(p)
    got = [(round(q["t0"], 3), round(q["t1"], 3), q["label"]) for q in d["preload"]["rois"]]
    assert d["preload_what"].startswith("your earlier ROIs") and got == [(0.002, 0.122, "MVC"), (0.28, 0.4, "AVC")]
    _review(p, wh, [dict(t0=0.5, t1=0.62, label="AVC")])
    d = load_review(p)
    assert d["preload_what"] == "current review (redo)" and len(d["preload"]["rois"]) == 1


def test_roi_editor_fixed_mode_places_and_moves_whole_windows():
    ed = _roi_editor()
    ed.fixed_ms = ed.click_ms = 120.0
    _drag(ed, 300.0, 340.0)                      # a drag places a fixed window on its centre
    _drag(ed, 600.0, 600.0)                      # so does a click
    assert [(round(q["t0"]), round(q["t1"])) for q in ed.rois] == [(260, 380), (540, 660)]
    _drag(ed, 380.0, 400.0)                      # grabbing an end moves the whole window
    assert (round(ed.rois[0]["t0"]), round(ed.rois[0]["t1"])) == (280, 400)


# ------------------------------------------------------------------ M-line editor: one ENTER on buffer 4
def _line_editor():
    import matplotlib
    matplotlib.use("Agg")
    from swp.manual import frames as F
    from swp.manual.line_gui import LineEditor
    x, z = np.linspace(-30, 30, 80), np.linspace(20, 90, 90)
    from scipy.ndimage import gaussian_filter
    env = gaussian_filter(np.random.default_rng(0).rayleigh(size=(z.size, x.size)), 1.5) + 1e-3
    panels = {b: F.Panel(b, env, x, z, 0, 1, 10, 0.0, "test") for b in F.BUFFERS}
    ed = LineEditor(panels, "test", lambda b, fr, n: panels[b], maximize=False)
    ed.fig.canvas.stop_event_loop = lambda: None
    return ed


def _click(ed, b, x, z):
    from types import SimpleNamespace
    ed.on_press(SimpleNamespace(inaxes=ed.ax[b], xdata=x, ydata=z, button=1))
    ed.on_release(None)


def test_line_on_buffer4_is_accepted_with_one_enter():
    from types import SimpleNamespace
    ed = _line_editor()
    _click(ed, 4, -10.0, 50.0)
    _click(ed, 4, 10.0, 60.0)
    ed.on_key(SimpleNamespace(key="enter", inaxes=None))
    res = ed.result
    assert res["action"] == "accept" and res["source_buffer"] == 4 and not res["motion_correction"]
    assert np.allclose(res["points4_mm"], res["points_src_mm"]) and res["mapping"] is None


def test_line_on_buffer1_is_still_registered_and_reviewed_first():
    from types import SimpleNamespace
    ed = _line_editor()
    _click(ed, 1, -10.0, 50.0)
    _click(ed, 1, 10.0, 60.0)
    ed.on_key(SimpleNamespace(key="enter", inaxes=None))
    assert ed.result is None and ed.phase == "review" and 4 in ed.map_info
    ed.on_key(SimpleNamespace(key="enter", inaxes=None))
    assert ed.result["action"] == "accept" and ed.result["source_buffer"] == 1


# ------------------------------------------------------------------ 2026-10-06: auto tilt + registered pre-loads
def _wave_st(c=3.0, length_mm=40.0, prf=925.9, t_mid=0.06):
    """A single Gaussian velocity band travelling at c m/s along the line (+ = away from r = 0)."""
    t = np.arange(0, 0.13, 1 / prf)
    r = np.linspace(0, length_mm * 1e-3, 250)
    arrival = t_mid + (r - r.mean()) / c
    d = np.exp(-0.5 * ((t[:, None] - arrival[None, :]) / 0.004) ** 2)
    d -= 0.6 * np.exp(-0.5 * ((t[:, None] - arrival[None, :] - 0.012) / 0.004) ** 2)
    return d, t, r


@pytest.mark.parametrize("c", [1.5, 3.0, -4.0])
def test_auto_tilt_follows_the_band_through_the_anchor(c):
    from swp.manual.slope_gui import auto_tilt
    d, t, r = _wave_st(c)
    r_a = 12e-3
    t_a = 0.06 + (r_a - r.mean()) / c
    got = auto_tilt(d, t, r, t_a * 1e3, r_a * 1e3)
    assert np.sign(got) == np.sign(c) and abs(got / c - 1) < 0.08


def _slope_editor(auto_tilt):
    import matplotlib
    matplotlib.use("Agg")
    from swp.manual.slope_gui import SlopeEditor
    d, t, r = _wave_st(2.0)
    views = [dict(name=n, quantity="velocity", data=d, r=r, t=t) for n in
             ("displacement gauss", "velocity median", "velocity gauss", "Keijzer velocity", "acceleration")]
    ed = SlopeEditor(dict(views=views), "test", init_speed=6.0, maximize=False, auto_tilt=auto_tilt)
    ed.fig.canvas.stop_event_loop = lambda: None
    return ed, t, r


def test_first_click_auto_tilts_later_clicks_keep_the_tilt_and_t_repeats_it():
    from types import SimpleNamespace
    ed, t, r = _slope_editor(True)
    r_a = 10e-3
    t_a = 0.06 + (r_a - r.mean()) / 2.0
    ev = SimpleNamespace(inaxes=ed.st_axes[4], button=1, xdata=t_a * 1e3, ydata=r_a * 1e3)
    ed.on_click(ev)
    sp = ed.lines["shared"]["speed"]
    assert abs(sp / 2.0 - 1) < 0.08 and ed.auto["shared"]["view"] == "acceleration"
    ed.on_key(SimpleNamespace(key="up"))                         # the reader tilts by hand
    ed.on_click(SimpleNamespace(inaxes=ed.st_axes[1], button=1, xdata=t_a * 1e3, ydata=r_a * 1e3))
    assert abs(ed.lines["shared"]["speed"] - (sp + 0.5)) < 1e-9  # a re-anchor keeps the hand tilt
    ed.on_key(SimpleNamespace(key="t"))                          # t: auto tilt again
    assert abs(ed.lines["shared"]["speed"] / 2.0 - 1) < 0.08
    ed.on_key(SimpleNamespace(key="3"))
    rec = ed.result["shared"]
    assert rec["auto_tilt"]["speed_m_s"] == pytest.approx(rec["speed_m_s"])
    assert rec["crossing_frames"] == pytest.approx(40.0 / abs(rec["speed_m_s"]) / (1e3 / 925.9), rel=0.02)


def test_without_auto_tilt_the_click_keeps_the_slider_start():
    from types import SimpleNamespace
    ed, t, r = _slope_editor(False)
    ed.on_click(SimpleNamespace(inaxes=ed.st_axes[4], button=1, xdata=60.0, ydata=20.0))
    assert ed.lines["shared"]["speed"] == pytest.approx(6.0) and not ed.auto


def _b4_file(folder, shifts_mm, seed=0):
    """A zea-layout buffer-4 file whose frames are one smooth speckle image shifted in z."""
    from scipy.ndimage import gaussian_filter, shift as ndshift
    x = np.linspace(-0.04, 0.04, 161)
    z = np.linspace(0.02, 0.10, 161)                             # 0.5 mm pixels
    base = gaussian_filter(np.random.default_rng(seed).normal(size=(z.size, x.size)), 3) ** 2 + 1e-3
    vals = np.stack([ndshift(base, (s / 0.5, 0), order=1, mode="nearest") for s in shifts_mm])
    iq = np.stack([np.sqrt(vals), np.zeros_like(vals)], -1).astype(np.float32)
    coords = np.zeros((z.size, x.size, 3))
    coords[..., 0], coords[..., 2] = x[None, :], z[:, None]
    out = os.path.join(folder, "output")
    os.makedirs(out, exist_ok=True)
    with h5py.File(os.path.join(out, "CombinedData_buffer4_iq.hdf5"), "w") as f:
        g = f.create_group("tracks/track_0/data/beamformed_data")
        g["values"], g["coordinates"] = iq, coords
    return os.path.join(out, "CombinedData_buffer4_iq.hdf5")


def test_previous_acquisition_is_the_nearest_earlier_one_with_a_general_line(tmp_path):
    from swp.manual import session as SE
    names = ["X_SW_data_10-July-2026_10-11-11", "X_SW_data_10-July-2026_10-12-47",
             "X_SW_data_10-July-2026_10-13-38", "X_SW_data_10-July-2026_10-14-12"]
    fs = [str(tmp_path / "C9" / n) for n in names]
    for f in fs:
        os.makedirs(f)
    pts = np.array([[0.0, 0.05], [0.02, 0.06]])
    for f in fs[:2]:
        S.write_json(S.Paths(f).general_json, dict(hash="h", points4_mm=(pts * 1e3).tolist()))
    S.write_json(S.Paths(fs[2]).general_json, dict(skipped=True, hash=None))
    assert SE.previous_acquisition(fs[3]) == fs[1]               # 10-13-38 skipped, 10-12-47 nearest
    assert SE.previous_acquisition(fs[0]) == fs[1]               # nothing earlier: the nearest later one


def test_registered_preloads_follow_the_anatomy(tmp_path):
    from swp.manual import frames as F
    from swp.manual import session as SE
    f = str(tmp_path / "C9" / "X_SW_data_10-July-2026_10-12-47")
    _b4_file(f, [0.0] * 12 + [2.4] * 12)                         # event frames: anatomy 2.4 mm deeper
    p = S.Paths(f)
    g4 = np.array([[-15.0, 55.0], [15.0, 65.0]])
    _general(p, g4 * 1e-3)
    gen = S.read_json(p.general_json)
    S.write_json(p.general_json, dict(gen, frames={"4": dict(frame=5)}, points4_mm=g4.tolist()))
    ev = F.load_panel(f, p.bmode(4), 4, 18, "event", n_avg=1)
    cfg = dict(registered=True, rotate_labels=["AVC"], max_angle_deg=4, angle_step_deg=2)
    pre = SE.registered_general(f, p, "MVC", ev, g4, cfg)
    assert np.allclose(pre["points_mm"] - g4, [[0.0, 2.4], [0.0, 2.4]], atol=0.5)
    assert "registered onto this event" in pre["what"]
    # the next acquisition: the previous one's line registered onto its R-peak frame
    f2 = str(tmp_path / "C9" / "X_SW_data_10-July-2026_10-13-38")
    _b4_file(f2, [-1.6])
    p2 = S.Paths(f2)
    d4 = F.load_panel(f2, p2.bmode(4), 4, 0, "general", n_avg=1)
    pre2 = SE.previous_general(f2, d4, dict(margins_mm=[12.0, 20.0], max_shift_mm=25.0))
    assert np.allclose(pre2["points_mm"] - g4, [[0.0, -1.6], [0.0, -1.6]], atol=0.5)
    assert pre2["registration"]["folder"] == os.path.basename(f)
