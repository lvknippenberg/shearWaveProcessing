"""Reference/tracking timing across the ARF push (fixed 2026-09-24)."""
import numpy as np
import pytest

from swp.acquisition.sequence import SWGeometry, assemble_tracking_frames


def geom(cycles=1500, switch=False, n_ref=20, n_trk=30):
    return SWGeometry(n_reference=n_ref, n_tracking=n_trk, na=1, harmonic=True, pri=270e-6,
                      focus_x=0.0, focus_z=0.045, roi_xlims=(0, 0), roi_zlims=(0, 0),
                      push_cycles=cycles, push_freq_hz=2.25e6, switch_tpc=switch)


@pytest.mark.parametrize("cycles, gap_us", [(1500, 970), (1900, 1170)])
def test_push_gap_follows_the_sequence(cycles, gap_us):
    # PRI + push burst (cycles / 2.25 MHz) rounded up to 100 us  (SeqControl 11 + 10)
    assert geom(cycles).push_gap_s() == pytest.approx(gap_us * 1e-6)


def test_push_gap_with_tpc_switch_adds_the_noops():
    assert geom(1500, switch=True).push_gap_s() == pytest.approx((970 + 20000 + 200) * 1e-6)


def test_push_gap_without_push_parameters_is_one_pri():
    g = geom(); g.push_cycles = 0
    assert g.push_gap_s() == pytest.approx(270e-6)


def _raw(g, n_meas=2, n_ax=8, n_el=4):
    n_tx = (g.n_reference + g.n_tracking) * 2
    return np.random.default_rng(0).integers(-50, 50, (n_meas, n_tx, n_ax, n_el, 1)).astype(np.int16)


def test_reference_times_include_the_push():
    g = geom(1900)
    tf = assemble_tracking_frames(_raw(g), g.n_reference, g.n_tracking, g.na, g.harmonic, g.pri,
                                  pi_mode="sliding", push_gap_s=g.push_gap_s())
    assert tf.reference.shape[1] == 2 * g.n_reference - 1           # sliding pairs
    assert tf.t_tracking[0] == 0 and np.allclose(np.diff(tf.t_tracking), g.pri)
    assert np.allclose(np.diff(tf.t_reference), g.pri)
    # last reference frame starts at the second-to-last reference transmit
    assert tf.t_reference[-1] == pytest.approx(-g.pri - g.push_gap_s(), abs=1e-9)


def test_retrofit_matches_assembly():
    from retrofit_push_gap import corrected_t_reference
    g = geom(1500)
    tf = assemble_tracking_frames(_raw(g), g.n_reference, g.n_tracking, g.na, g.harmonic, g.pri,
                                  pi_mode="sliding", push_gap_s=g.push_gap_s())
    assert np.allclose(corrected_t_reference(tf.reference.shape[1], g), tf.t_reference)
