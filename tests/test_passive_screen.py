"""Semblance picker of the manual passive study (swp.passive_screen) and the 'screened' state."""
from __future__ import annotations

import numpy as np

from swp.manual import store as S
from swp.passive_screen import pick_windows, screen_track

RR = 0.8                                   # 75 bpm -> QS2 = 546 - 2.1 * 75 = 388.5 ms


def _track(bumps, t_end=1.2, base=0.1):
    """A synthetic score track on the buffer-4 clock with Gaussian bumps {t: height}."""
    t = np.arange(0.03, t_end - 0.03 + 1e-9, 0.005)
    sem = np.full_like(t, base)
    for tc, h in bumps.items():
        sem = np.maximum(sem, h * np.exp(-0.5 * ((t - tc) / 0.01) ** 2))
    return dict(t=t, sem=sem, c=np.full_like(t, 3.0), burst=1 + sem)


def test_picks_the_semblance_peak_inside_the_phase_window():
    # AVC window of the first beat: 0.2685-0.5085 s. A wave at 0.36 s (QS2 -29 ms) must win over
    # the late window edge, where the energy detector used to land.
    tr = _track({0.05: 0.9, 0.36: 0.85, 0.85: 0.9})
    w = pick_windows(tr, r_peaks_s=np.array([0.0, 0.8]), rr_s=RR)
    got = {x.expect: [] for x in w}
    for x in w:
        got[x.expect].append(round(x.t_peak, 3))
    assert got["AVC"] == [0.36]
    assert got["MVC"] == [0.05, 0.85]


def test_a_search_window_after_the_recording_is_dropped():
    # second beat at 0.8 s: its AVC window (1.07-1.31 s) lies mostly after the 1.2 s record
    tr = _track({0.05: 0.9, 0.36: 0.85, 0.85: 0.9})
    w = pick_windows(tr, r_peaks_s=np.array([0.0, 0.8]), rr_s=RR, max_events=3)
    assert [x.expect for x in w] == ["MVC", "AVC", "MVC"]
    w = pick_windows(tr, r_peaks_s=np.array([0.0, 0.8]), rr_s=RR, max_events=3, min_inside=0.0)
    assert sum(x.expect == "AVC" for x in w) >= 1        # without the rule it can come back


def test_low_score_windows_are_screened_not_dropped():
    tr = _track({0.05: 0.9, 0.36: 0.2, 0.85: 0.9}, base=0.05)
    w = pick_windows(tr, r_peaks_s=np.array([0.0, 0.8]), rr_s=RR, max_events=3, screen_min=0.3)
    avc = [x for x in w if x.expect == "AVC"][0]
    assert avc.screened and avc.screen < 0.3
    assert not any(x.screened for x in w if x.expect == "MVC")


def test_without_r_peaks_the_best_score_peaks_are_taken():
    tr = _track({0.2: 0.9, 0.6: 0.7, 1.0: 0.8})
    w = pick_windows(tr, r_peaks_s=None, rr_s=None, max_events=2)
    assert sorted(round(x.t_peak, 2) for x in w) == [0.2, 1.0]
    assert all(x.expect == "" for x in w)


def test_screen_track_finds_a_propagating_wave():
    fs, L = 926.0, 0.03
    t = np.arange(0, 0.6, 1 / fs)
    r = np.linspace(0, L, 120)
    c, t_arr = 3.0, 0.3
    v = np.exp(-0.5 * ((t[:, None] - t_arr - r[None, :] / c) / 0.004) ** 2)
    v += 0.05 * np.random.default_rng(0).standard_normal(v.shape)
    tr = screen_track(v, r, t, step=0.01)
    k = int(np.nanargmax(tr["sem"]))
    assert abs(tr["t"][k] - (t_arr + L / c / 2)) < 0.05
    assert tr["burst"][k] > 1.5


def test_screened_windows_do_not_ask_a_line_or_hold_the_folder(tmp_path):
    out = tmp_path / "C1" / "SW_data_x" / "output"
    out.mkdir(parents=True)
    (out / "CombinedData_buffer4_iq.hdf5").write_bytes(b"")
    (out / "CombinedData_buffer4_iq.gif").write_bytes(b"")
    f = str(out.parent)
    p = S.Paths(f)
    pts = np.array([[-0.01, 0.05], [0.012, 0.047]])
    S.save_line(p.general_npz, pts)
    S.write_json(p.general_json, dict(hash=S.points_hash(pts), source_buffer=4))
    ws = [dict(t_peak=0.05, t0=0.0, t1=0.1, score=1.0, screen=0.8, screened=False),
          dict(t_peak=0.45, t0=0.4, t1=0.5, score=1.0, screen=0.1, screened=True)]
    S.write_json(p.windows_json, dict(key=dict(general_hash=S.points_hash(pts)),
                                      hash=S.windows_hash(ws), windows=ws))
    s = S.state(f)
    assert s["need_events"] == [0] and s["screened"] == [1]
    ev = dict(windows_hash=S.windows_hash(ws),
              events={"0": dict(hash="h0", skipped=True)})
    S.write_json(p.events_json, ev)
    assert S.state(f)["stage"] == "done"                  # the screened window does not hold it


def test_screen_windows_scores_picked_windows_and_marks_low_ones():
    from swp.mline.select import BurstWindow
    from swp.passive_screen import screen_windows
    tr = _track({0.05: 0.9, 0.36: 0.2}, base=0.05)
    w = [BurstWindow(0.05, 0.0, 0.1, 1.0, expect="MVC"), BurstWindow(0.36, 0.31, 0.41, 1.0, expect="AVC")]
    screen_windows(w, tr, screen_min=0.3)
    assert not w[0].screened and w[0].screen > 0.8
    assert w[1].screened and w[1].screen < 0.3


def _energy_st(bumps, t):
    """(n_s, n_t) along-line strip whose energy has Gaussian bursts at the given times."""
    amp = 0.05 + sum(h * np.exp(-0.5 * ((t - tc) / 0.01) ** 2) for tc, h in bumps.items())
    return np.tile(amp, (10, 1))


def test_energy_detector_drops_a_search_window_after_the_recording():
    from swp.mline.select import detect_phase_windows
    t = np.arange(0, 1.2, 1 / 463.0)                     # stride-2 overview rate
    D = _energy_st({0.05: 1.0, 0.36: 0.8, 0.85: 1.0}, t)
    rp = np.array([0.0, 0.8])
    old, _ = detect_phase_windows(D, t, rp, RR, fill_with_energy=False)
    new, _ = detect_phase_windows(D, t, rp, RR, fill_with_energy=False, min_inside=0.5)
    assert [w.expect for w in old].count("AVC") == 2      # the phantom second AVC ...
    assert [w.expect for w in new] == ["MVC", "AVC", "MVC"]   # ... is gone


def test_old_windows_without_screen_fields_still_load():
    from swp.mline.select import BurstWindow
    w = BurstWindow(**dict(t_peak=0.1, t0=0.05, t1=0.15, score=1.0, label="MVC", expect="MVC"))
    assert w.screen is None and not w.screened
