"""Quality metrics for shear-wave visualization (not tied to a fitted speed).

The goal (see the reference cardiac-SWE M-mode figure) is a **clear, symmetric propagating
wavefront** — two straight diagonal ridges radiating **from the known push origin r0, starting at
t≈0** — not a smooth blob and not late/reverberant energy that crosses the field without emanating
from the push.  ``origin_coherence`` is the recommended ranking metric: it is *origin-aware* and
directly measures that physical picture (see its docstring).  ``wavefront_coherence`` is an
origin-*agnostic* (k, ω) concentration score kept for reference/back-compat, but it cannot tell a
wave that radiates from r0 from one that merely produces coherent diagonals elsewhere (reflections,
standing waves), so it can rank reverberation-dominated plots above clean ones.  ``wavefront_visibility``
(outward/inward directional energy) is weakly discriminating and retained only for comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .filters.directional import directional_spacetime
from .speed.spacetime import SpaceTime


def origin_coherence(st: SpaceTime, r0: float, cmin: float = 1.0, cmax: float = 10.0,
                     n_speeds: int = 40, d_min: float = 2e-3, d_max: float = 16e-3,
                     t0_max: float = 3.0e-3, demean: bool = True, return_sides: bool = False):
    """How well the field is a **symmetric wave radiating OUTWARD from the known origin r0,
    starting near t≈0** — in [0, 1].  This is the recommended ranking metric.

    Physics it encodes (all three matter, and together they separate clean plots from reverberation):

    1. **Known origin.**  The ARF push is a vertical line source at x=0, so the wave origin on the
       M-line is fixed: ``r0`` = the M-line's x=0 crossing (config ``focus.mode: vertical``).  A
       genuine shear wave forms a "V" with its apex at ``r0``; reflections/standing waves form
       coherent diagonals that do *not* emanate from ``r0``.
    2. **Single outward speed.**  For each side of ``r0`` we slant-stack the temporal envelope along
       the outward moveout ``t = t0 + |r - r0| / c`` and take the best single speed ``c``.  A coherent
       outward wave stacks constructively (stack peak ≈ per-column peak → ratio → 1); incoherent or
       crossing energy does not.
    3. **Early origin time.**  The wave leaves ``r0`` at the push (t≈0), so the aligned origin time
       ``t0`` (the location of the stack peak) is required to fall in ``[0, t0_max]``.  This is the
       key discriminator: it rejects *late* reverberation and *inward*-converging (reflected) energy,
       whose apparent apex is late — exactly the features of the poor measurements.

    The two sides are combined as a **geometric mean**, so a one-sided field (wave on only one half of
    the M-line) is penalised — symmetry is built in, not a separate multiplier.  Only the near-field
    window ``|r - r0| ∈ [d_min, d_max]`` is used (skips the singular apex column and the far ends that
    run off the septum).  Envelope (Hilbert) is used so it applies equally to displacement and
    (bipolar) velocity.  With ``return_sides`` the per-side coherences (left, right) are also returned.
    """
    data = st.data - st.data.mean(axis=0, keepdims=True) if demean else st.data
    env = np.abs(hilbert(data, axis=0))
    nt = env.shape[0]
    dt = float(st.t[1] - st.t[0])
    d = np.abs(st.r - r0)
    base_i = np.arange(nt)
    i_early = max(1, int(round(t0_max / dt)))
    speeds = np.linspace(cmin, cmax, n_speeds)

    def side(mask):
        cols = np.where(mask & (d >= d_min) & (d <= d_max))[0]
        if cols.size < 4:
            return 0.0
        colpeak = env[:, cols].max(axis=0).mean() + 1e-12    # mean per-column envelope peak
        best = 0.0
        for c in speeds:
            shift = d[cols] / c / dt
            valid = shift <= (nt - 1)
            if int(valid.sum()) < 4:
                continue
            stacked = np.zeros(nt)
            for j, s in zip(cols[valid], shift[valid]):
                stacked += np.interp(base_i + s, base_i, env[:, j])   # align outward moveout to t0
            stacked /= float(valid.sum())
            best = max(best, float(stacked[:i_early].max()))          # origin time t0 must be early
        return best / colpeak

    left = side(st.r < r0)
    right = side(st.r >= r0)
    oc = float(np.sqrt(max(left, 0.0) * max(right, 0.0)))
    if return_sides:
        return oc, float(left), float(right)
    return oc


def passive_coherence(st: SpaceTime, r0: float, cmin: float = 0.5, cmax: float = 10.0,
                      n_speeds: int = 80, d_min: float = 2e-3, d_max: float | None = None,
                      t0_max: float | None = None, demean: bool = True,
                      return_speed: bool = False):
    """One-sided origin coherence for **passive** SWE, in [0, 1] — the recommended passive metric.

    Passive shear waves differ from ARF waves in ways this metric is built around:

    * **One-sided.**  The wave originates at a valve closure at **one end** of the M-line (``r0``)
      and travels across it in a **single direction** — there is no symmetric second lobe, so
      (unlike :func:`origin_coherence`) only the side that carries the wave is scored, and there is
      no geometric-mean symmetry penalty.
    * **Wide / long-range.**  Passive waves are higher-SNR and propagate across a **larger distance**,
      so the whole propagation extent is used (``d_max`` defaults to the full M-line length) and a
      coherent wave is rewarded for staying coherent far from the origin.
    * **Onset not at t≈0.**  The valve-closure onset falls somewhere inside the analysis window (not
      pinned to t=0 as the ARF push is), so the origin time ``t0`` is free by default
      (``t0_max=None``); pass ``t0_max`` to restrict it.

    Method: for columns with ``|r - r0| ∈ [d_min, d_max]`` slant-stack the temporal Hilbert envelope
    along the moveout ``t = t0 + |r - r0| / c`` and take the best single speed ``c`` (and origin time
    ``t0``).  Crucially the score is the **improvement of the best finite-speed stack over the
    no-moveout stack**, ``(best_stack − flat_stack) / colpeak``: a genuinely *propagating* wave stacks
    far better when aligned to its slope than when not (→ high), while a **non-propagating flat band**
    (or a blob, or noise) gains little from alignment (→ ~0).  This is what makes the metric reward
    *propagation*, not just a bright horizontal band.  ``return_speed`` also returns the best speed.
    """
    data = st.data - st.data.mean(axis=0, keepdims=True) if demean else st.data
    env = np.abs(hilbert(data, axis=0))
    nt = env.shape[0]
    dt = float(st.t[1] - st.t[0])
    d = np.abs(st.r - r0)
    if d_max is None:
        d_max = float(d.max())
    cols = np.where((d >= d_min) & (d <= d_max))[0]
    if cols.size < 4:
        return (0.0, float("nan")) if return_speed else 0.0
    colpeak = env[:, cols].max(axis=0).mean() + 1e-12
    base_i = np.arange(nt)
    i_hi = nt if t0_max is None else max(1, int(round(t0_max / dt)))
    # flat baseline: average the columns with NO moveout (best t0), i.e. the c -> inf / static case.
    flat = float(env[:, cols].mean(axis=1)[:i_hi].max())
    speeds = np.linspace(cmin, cmax, n_speeds)
    best, best_c = 0.0, float("nan")
    for c in speeds:
        shift = d[cols] / c / dt
        valid = shift <= (nt - 1)
        if int(valid.sum()) < 4:
            continue
        stacked = np.zeros(nt)
        for j, s in zip(cols[valid], shift[valid]):
            stacked += np.interp(base_i + s, base_i, env[:, j])   # align outward moveout to t0
        stacked /= float(valid.sum())
        peak = float(stacked[:i_hi].max())
        if peak > best:
            best, best_c = peak, float(c)
    # Reward propagation: how much of the alignable "headroom" (colpeak - flat) the best finite-speed
    # moveout recovers. Clean wave -> best ~ colpeak -> ~1; flat band -> colpeak ~ flat -> ~0
    # (headroom ~0); noise/blob -> small recovery -> low. Independent of the wave's speed.
    headroom = colpeak - flat
    coh = float(np.clip((best - flat) / headroom, 0.0, 1.0)) if headroom > 1e-6 * colpeak else 0.0
    return (coh, best_c) if return_speed else coh


def wavefront_coherence(st: SpaceTime, cmin: float = 1.0, cmax: float = 10.0,
                        rel_width: float = 0.25, ft_min: float = 80.0, ft_max: float = 800.0,
                        fr_min: float = 30.0, fr_max: float = 400.0,
                        n_speeds: int = 60, demean: bool = True, return_symmetry: bool = False):
    """How concentrated the space-time energy is on a **single** shear-wave speed, in [0,1].

    Target: two symmetric tilted bands and *nothing else*.  A propagating wave ``r = c·t`` occupies
    the lines ``|f_t / f_r| = c`` in the 2-D Fourier transform; a symmetric wave populates both signs
    of ``f_t·f_r``.  We search for the single speed ``c*`` whose narrow double-wedge
    ``|f_t/f_r| ∈ [c*(1∓rel_width)]`` captures the most power **within the shear-wave frequency band**
    ``[ft_min, ft_max] × [fr_min, fr_max]``, and return it as a fraction of the *total* non-DC power.

    The frequency band is the key: the shear wave lives at *moderate* spatial/temporal frequencies,
    whereas pixel noise is broadband up to Nyquist.  Numerator counts only the in-band wedge, but the
    denominator is all non-DC power (including the high-frequency noise), so:

    * **over-smooth blob** -> little in-band energy -> low;
    * **broadband/speckle noise** (e.g. raw velocity) -> most energy is high-frequency, outside the
      numerator band but inside the denominator -> low; **denoising (Gaussian/median/temporal) removes
      that high-frequency energy and raises the score** -> the metric now rewards a clean plot;
    * **two clean symmetric bands at one speed** -> most energy in the wedge -> high.

    ``return_symmetry`` also returns the left/right energy balance in [0,1] at ``c*``.
    """
    data = st.data - st.data.mean(axis=0, keepdims=True) if demean else st.data
    nt, nr = data.shape
    dt = float(st.t[1] - st.t[0])
    dr = float(st.r[1] - st.r[0])
    ft = np.fft.fftfreq(nt, dt)[:, None]        # temporal freq [Hz]
    fr = np.fft.fftfreq(nr, dr)[None, :]        # spatial freq  [1/m]
    power = np.abs(np.fft.fft2(data)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.abs(ft / fr)                 # phase speed |f_t/f_r| [m/s]
    nondc = np.ones((nt, nr), bool)
    nondc[0, :] = False
    nondc[:, 0] = False
    band = (nondc & (np.abs(ft) >= ft_min) & (np.abs(ft) <= ft_max)
            & (np.abs(fr) >= fr_min) & (np.abs(fr) <= fr_max))
    total = power[nondc].sum() + 1e-20

    # in-band bins only; find the single speed whose [c(1-w), c(1+w)] window holds the most power,
    # via a cumulative sum over speed-sorted bins (vectorised over candidate speeds).
    sp = speed[band]
    pw = power[band]
    sgn = (ft * np.ones_like(fr))[band] * (np.ones_like(ft) * fr)[band]  # sign of ft*fr per bin
    order = np.argsort(sp)
    sp_s, pw_s, sgn_s = sp[order], pw[order], sgn[order]
    cumP = np.concatenate([[0.0], np.cumsum(pw_s)])
    cumR = np.concatenate([[0.0], np.cumsum(np.where(sgn_s < 0, pw_s, 0.0))])
    cumL = np.concatenate([[0.0], np.cumsum(np.where(sgn_s > 0, pw_s, 0.0))])
    speeds = np.linspace(cmin, cmax, n_speeds)
    lo = np.searchsorted(sp_s, speeds * (1 - rel_width), side="left")
    hi = np.searchsorted(sp_s, speeds * (1 + rel_width), side="right")
    e = cumP[hi] - cumP[lo]
    ib = int(np.argmax(e)) if e.size else 0
    coh = float((e[ib] if e.size else 0.0) / total)
    if not return_symmetry:
        return coh
    er = cumR[hi[ib]] - cumR[lo[ib]] if e.size else 0.0
    el = cumL[hi[ib]] - cumL[lo[ib]] if e.size else 0.0
    sym = float(min(er, el) / (max(er, el) + 1e-20))
    return coh, sym


def wavefront_visibility(st: SpaceTime, r0: float, demean: bool = True) -> float:
    """Fraction of space-time energy in *outward*-travelling waves, in [0, 1].

    Splits the M-line at the push origin ``r0``; on the far (r >= r0) side an outward wave
    travels +r, on the near (r < r0) side it travels -r.  Returns
    ``E_outward / (E_outward + E_inward)`` from directional (k, omega) decomposition:
    ~0.5 for undirected motion/noise (no preferred direction), approaching 1 for a clean
    wave radiating from the push.  ``demean`` removes the temporal-DC (static) component first.
    """
    data = st.data - st.data.mean(axis=0, keepdims=True) if demean else st.data
    pos = directional_spacetime(data, "pos")   # +r-travelling component
    neg = directional_spacetime(data, "neg")   # -r-travelling component
    right = st.r >= r0
    left = st.r < r0
    e_out = float((pos[:, right] ** 2).sum() + (neg[:, left] ** 2).sum())
    e_in = float((neg[:, right] ** 2).sum() + (pos[:, left] ** 2).sum())
    return e_out / (e_out + e_in + 1e-20)
