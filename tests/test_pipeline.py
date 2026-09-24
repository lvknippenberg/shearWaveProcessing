"""End to end: phase-encoded synthetic ARF wave -> run_pipeline -> speed.

The synthetic wave is already 2 mm out at t = 0, as in vivo (it travels during the ~0.7 ms push).
Speed / truth measured 2026-09-24 over c = 2, 3, 4 m/s and both estimator modes
(``study/logs`` has no copy; rerun ``python -m pytest tests -k pipeline -rxX`` to see them fail/pass):

* estimators alone are exact: no band-pass, no directional -> ``tof_xcorr`` 1.00, ``ttp_ransac`` 1.00-1.09;
* the **outward directional filter biases speed high**, +5 to +37 % (tof_xcorr);
* the envelope slant stack ``radon`` is +20 to +94 % high in every configuration;
* a 75-750 Hz band-pass on the ~16 ms record makes the automatic speeds erratic (0.73-2.09x).

The last three are strict xfails: if one is fixed the suite reports it, and the xfail is removed.
"""
import numpy as np
import pytest

from conftest import PRF, DX, make_acquisition, outward_wave
from swp.viz.mline.mline import horizontal_mline
from swp.viz.pipeline import PipelineConfig, Step, run_pipeline

BAND = [Step("temporal_bandpass", {"f_lo": 75, "f_hi": 750, "order": 3})]
MODES = [("relative_to_reference", "displacement"), ("frame_to_frame", "velocity")]


def synthetic_push(c=3.0, n=59, nz=12, nx=100):
    x = (np.arange(nx) - nx // 2) * DX
    disp = np.broadcast_to(outward_wave(c=c, x=x, n=n)[:, None, :], (n, nz, nx)).copy()
    return make_acquisition(disp, np.zeros((39, nz, nx)), t_ref=(np.arange(39) - 39) / PRF - 970e-6,
                            meta={"push_gap_s": 970e-6})


def run(c, mode="relative_to_reference", quantity="displacement", directional=False,
        speed="tof_xcorr", field_filters=(), **kw):
    acq = synthetic_push(c=c)
    ml = horizontal_mline(acq.x, float(acq.push_z), 250)
    cfg = PipelineConfig(estimator="loupas", mode=mode, quantity=quantity, directional=directional,
                         field_filters=list(field_filters), speed=speed, mline_offsets=3,
                         mline_offset_step_m=0.4e-3, push_x_m=0.0, **kw)
    return run_pipeline(acq, ml, cfg), ml


@pytest.mark.parametrize("c", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("mode, quantity", MODES)
def test_estimators_recover_speed_exactly(c, mode, quantity):
    res, ml = run(c, mode, quantity)
    assert res.r0 == pytest.approx(float(np.interp(0.0, ml.x, ml.r)), abs=1e-3)
    assert res.speed.c_mean == pytest.approx(c, rel=0.03)
    res, _ = run(c, mode, quantity, speed="ttp_ransac")
    assert res.speed.c_mean == pytest.approx(c, rel=0.10)


@pytest.mark.xfail(strict=True, reason="outward directional filter biases speed high (+26 % here)")
def test_directional_filter_is_speed_neutral():
    res, _ = run(3.0, directional=True)
    assert res.speed.c_mean == pytest.approx(3.0, rel=0.05)


@pytest.mark.xfail(strict=True, reason="envelope slant stack 'radon' is +20-94 % high")
def test_radon_speed_unbiased():
    res, _ = run(3.0, speed="radon")
    assert res.speed.c_mean == pytest.approx(3.0, rel=0.10)


@pytest.mark.xfail(strict=True, reason="75-750 Hz band-pass on a 16 ms record distorts the "
                                       "automatic speed (0.73x at 4 m/s)")
def test_bandpass_is_speed_neutral():
    res, _ = run(4.0, field_filters=BAND)
    assert res.speed.c_mean == pytest.approx(4.0, rel=0.10)


def test_continuous_record_path_runs_and_crops_to_tracking():
    res, _ = run(3.0, mode="frame_to_frame", quantity="velocity", continuous_record=True,
                 field_filters=[Step("giannantonio_motion_filter", {"order": 1})] + BAND)
    assert res.st.t[0] >= 0
    assert np.isfinite(res.speed.c_mean)
