"""Hand-drawn per-event M-lines must never be archived by an automatic re-detection.

On 2026-09-24 a study reprocess re-detected four folders whose cached windows predated a detection
setting in the cache key, and moved their hand-drawn lines away. ``process_single_line`` now
raises instead, before loading any data.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import swp.passive as SP

REPO = Path(__file__).resolve().parents[1]
CONFIG = str(REPO / "configs" / "passive.yaml")


def _folder(tmp_path, window_mlines):
    out = tmp_path / "meas" / "output"
    (out / "mlines").mkdir(parents=True)
    (out / "swp_passive").mkdir()
    pts = np.array([[-0.01, 0.05], [0.0, 0.055], [0.01, 0.06]])
    np.savez(out / "mlines" / SP.GENERAL_NPZ, points=pts, n_samples=250)
    np.savez(out / "mlines" / "passive_win0_mline.npz", points=pts + 0.002, n_samples=250)
    st = dict(key=dict(general_points=pts.tolist(), detect_band=[5, 150], window_ms=100.0,
                       max_events=4, overview_stride=2, mline_source="buffer3_frame0"),   # no detect_mode
              windows=[dict(t_peak=0.1, t0=0.05, t1=0.15, score=1.0, label="MVC")])
    if window_mlines:
        st["window_mlines"] = {"0": dict(buffer=1, frame=3, from_general=False)}
    (out / "swp_passive" / SP.WINDOWS_JSON).write_text(json.dumps(st))
    return tmp_path / "meas", out


def test_stale_windows_with_hand_lines_raise(tmp_path, monkeypatch):
    folder, out = _folder(tmp_path, window_mlines=True)
    monkeypatch.setattr(SP, "load_acq", lambda *a, **k: None)
    with pytest.raises(SP.StaleWindowsError, match="detect_mode"):
        SP.process_single_line(str(folder), CONFIG)
    assert (out / "mlines" / "passive_win0_mline.npz").exists()          # untouched
    assert not list((out / "mlines").glob("archive_*"))


def test_stale_windows_without_hand_lines_may_redetect(tmp_path, monkeypatch):
    folder, _ = _folder(tmp_path, window_mlines=False)
    monkeypatch.setattr(SP, "load_acq", lambda *a, **k: None)

    class Detected(Exception):
        pass

    def fake_detect(*a, **k):
        raise Detected
    monkeypatch.setattr(SP, "detect_windows", fake_detect)
    with pytest.raises(Detected):                     # got past the guard to the detection
        SP.process_single_line(str(folder), CONFIG)
