"""Output stamping and the data-location module."""
import importlib

import h5py
import pytest


def test_stamp_roundtrip(tmp_path):
    from swp.provenance import stamp_h5, read_h5
    p = tmp_path / "out.hdf5"
    with h5py.File(p, "w") as f:
        f["x"] = [1, 2, 3]
    stamp_h5(str(p), config={"band": [75, 750], "quantity": "velocity"})
    s = read_h5(str(p))
    for k in ("swp_git_commit", "zea_version", "config_sha256", "config_json", "created_utc", "command"):
        assert k in s
    stamp_h5(str(p), config={"quantity": "velocity", "band": [75, 750]})   # re-stamp replaces
    assert read_h5(str(p))["config_sha256"] == s["config_sha256"]          # key order irrelevant


def test_config_hash_changes_with_content():
    from swp.provenance import config_hash
    assert config_hash({"a": 1})[0] != config_hash({"a": 2})[0]


def test_stamp_text_is_comment_lines():
    from swp.provenance import stamp_text
    txt = stamp_text(config={"a": 1})
    assert txt and all(line.startswith("# ") for line in txt.splitlines())
    assert "config_json" not in txt


def test_paths_env_override(monkeypatch, tmp_path):
    import swp.paths as P
    monkeypatch.setenv("SWP_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("SWP_VOLTAGE_SWEEP", raising=False)
    monkeypatch.delenv("SWP_METRIC_EXPERIMENT", raising=False)
    P = importlib.reload(P)
    assert P.METRIC_EXPERIMENT.startswith(str(tmp_path))
    assert "MI estimation" in P.VOLTAGE_SWEEP
    monkeypatch.undo()
    importlib.reload(P)
