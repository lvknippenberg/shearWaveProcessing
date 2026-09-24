"""M-line transfer between buffers: a known rigid motion of the anatomy must be recovered.

The source and target images are built analytically from the same blob "anatomy", the target
moved by a known transform, each with its own speckle (independent Rayleigh noise), as between a
focused and a diverging-wave image acquired in different beats.
"""
import numpy as np
import pytest

from swp.mline.transfer import Transform, transfer_line

PIX = 0.394
X = np.arange(-35, 35, PIX)
Z = np.arange(30, 110, PIX)


def _anatomy(xx, zz, seed=3):
    rng = np.random.default_rng(seed)
    img = np.full(xx.shape, 0.05)
    for _ in range(60):
        cx, cz = rng.uniform(-40, 40), rng.uniform(25, 115)
        sx, sz = rng.uniform(1.5, 5), rng.uniform(1.5, 5)
        img += rng.uniform(0.2, 1) * np.exp(-((xx - cx) / sx) ** 2 - ((zz - cz) / sz) ** 2)
    # a wall-like band (the septum) for the line to follow
    img += 1.5 * np.exp(-((zz - 70 - 0.3 * xx) / 2.0) ** 2) * (np.abs(xx) < 20)
    return img


def _images(t: Transform, speckle_seed=(1, 2), snr=1.0):
    Zg, Xg = np.meshgrid(Z, X, indexing="ij")
    src = _anatomy(Xg, Zg)
    # target(q) = source(T^-1 q): invert T analytically
    a = np.radians(t.angle)
    Rinv = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    q = np.stack([Xg, Zg], -1) - np.array([t.cx + t.dx, t.cz + t.dz])
    p = q @ Rinv.T + np.array([t.cx, t.cz])
    dst = _anatomy(p[..., 0], p[..., 1])
    out = []
    for img, s in zip((src, dst), speckle_seed):
        rng = np.random.default_rng(s)
        sp = np.abs(rng.normal(size=img.shape) + 1j * rng.normal(size=img.shape)) / np.sqrt(2)
        out.append(img * (1 + snr * (sp - 1)))
    return (out[0], X, Z), (out[1], X, Z)


LINE = np.c_[np.linspace(-15, 15, 40), 70 + 0.3 * np.linspace(-15, 15, 40)]


@pytest.mark.parametrize("dx,dz,angle", [(0, 0, 0), (3.0, -2.0, 0), (-4.5, 1.5, 4.0), (1.0, 5.0, -6.0)])
def test_recovers_known_rigid_motion(dx, dz, angle):
    c = LINE.mean(axis=0)
    truth = Transform(dx=dx, dz=dz, angle=angle, cx=c[0], cz=c[1])
    src, dst = _images(truth)
    res = transfer_line(LINE, src, dst, pix=PIX, angles=np.arange(-10.0, 10.01, 1.0), angle_tol=0.0)
    err = np.hypot(*(res.points - truth.apply(LINE)).T)
    assert np.median(err) < 0.4 and err.max() < 1.0, (res.transform, err.max())
    assert abs(res.transform.angle - angle) < 1.5


def test_translation_only_mode_and_confidence():
    c = LINE.mean(axis=0)
    truth = Transform(dx=2.0, dz=1.0, cx=c[0], cz=c[1])
    src, dst = _images(truth)
    res = transfer_line(LINE, src, dst, pix=PIX, angles=(0.0,))
    assert res.transform.angle == 0.0
    assert np.hypot(res.transform.dx - 2.0, res.transform.dz - 1.0) < 0.3
    assert res.transform.corr > res.transform.corr0      # the registration improved the match
    assert res.spread_mm < 0.5 and res.reliable()        # the ensemble agrees
