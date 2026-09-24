"""The passive default recipe (2026-09-24, report/passive_methods parts 1-2) is what configs/passive.yaml builds.

Velocity, band-pass 15-150 Hz, Gaussian 0.6 x 1.2 mm, moving mean 3, 5 M-lines x 0.5 mm, no
directional filter, no SVD, no CFWI. The three views differ only in the spatial filter (default /
unsmoothed / median) so a manual speed can be checked against smoothing.
"""
from pathlib import Path

import pytest
import yaml

from swp.viz.runconfig import build_pipeline_config, build_views

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(REPO / "configs" / "passive.yaml"))


def _core(pc):
    assert pc.quantity == "velocity"
    assert pc.mode == "frame_to_frame"
    assert pc.estimator == "loupas"
    assert pc.iq_filters == []                      # no SVD
    assert not pc.directional
    assert pc.mline_offsets == 5 and pc.mline_offset_step_m == pytest.approx(0.5e-3)
    names = [s.name for s in pc.field_filters]
    assert names[0] == "temporal_bandpass" and names[-1] == "temporal_moving_mean"
    bp = pc.field_filters[0].params
    assert (bp["f_lo"], bp["f_hi"]) == (15, 150)
    assert pc.field_filters[-1].params["window"] == 3
    return names[1:-1], pc.field_filters[1:-1]


def test_pipeline_default(cfg):
    names, steps = _core(build_pipeline_config(cfg))
    assert names == ["spatial_smooth"]
    assert steps[0].params["sigma_z_m"] == pytest.approx(0.6e-3)
    assert steps[0].params["sigma_x_m"] == pytest.approx(1.2e-3)


def test_views_differ_only_in_spatial_filter(cfg):
    views = build_views(cfg)
    assert len(views) == 3
    spatial = [_core(pc)[0] for _, pc in views]
    assert spatial == [["spatial_smooth"], [], ["spatial_median"]]
    assert views[0][1].field_filters == build_pipeline_config(cfg).field_filters


def test_v1_config_is_kept_for_the_scored_panels():
    v1 = yaml.safe_load(open(REPO / "configs" / "passive_v1.yaml"))
    names = [n for n, _ in build_views(v1)]
    assert "disp bp10-150 gauss mean3" in names and "velocity bp15-90 gauss1.0 mean5" in names
