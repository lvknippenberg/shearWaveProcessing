"""Experimental / novel processing steps for the interactive method-exploration GUI.

These are additional, individually-tunable steps beyond the settled recipe, several of which
use the **pre-push reference frames** (pure cardiac motion, no shear wave) to estimate and remove
cardiac motion better than a fixed band-pass. All follow the same signatures as the existing
filters so the pipeline / registry dispatches them uniformly.

IQ-space (before displacement estimation):
  * :func:`iq_spatial_lowpass`  - per-frame complex Gaussian low-pass (speckle denoise).
  * :func:`iq_slowtime_lowpass` - Butterworth low-pass along slow-time (ensemble denoise).

Displacement-space:
  * :func:`svd_clutter_field`          - low-rank (SVD) clutter removal on the displacement movie
                                         (Demene-style, but on displacement not IQ).
  * :func:`reference_subspace_projection` - build the cardiac-motion subspace from the reference
                                         frames and project it out of the tracking field (novel).
  * :func:`phase_unwrap_temporal`      - unwrap the phase-wrapped displacement along slow time
                                         (large in-vivo motion wraps the estimator).
"""
from __future__ import annotations

import numpy as np

from .context import FilterCtx


# ============================== IQ-space pre-filters ==============================
def iq_spatial_lowpass(iq: np.ndarray, sigma_z_m: float = 0.2e-3, sigma_x_m: float = 0.4e-3,
                       dz: float = None, dx: float = None) -> np.ndarray:
    """Per-frame complex Gaussian low-pass of the IQ ensemble (speckle denoise before tracking).

    Smooths I and Q separately with physical sigmas. ``dz``/``dx`` are injected by the pipeline
    from the acquisition (a plain IQ filter has no ctx), defaulting to a mild pixel-based blur if
    absent. Keep sigmas small - over-smoothing axially biases the phase-based displacement.
    """
    from scipy.ndimage import gaussian_filter
    sz = (sigma_z_m / dz) if dz else 1.0
    sx = (sigma_x_m / dx) if dx else 1.0
    r = gaussian_filter(iq.real, sigma=(0.0, sz, sx), mode="nearest")
    i = gaussian_filter(iq.imag, sigma=(0.0, sz, sx), mode="nearest")
    return r + 1j * i


def iq_slowtime_lowpass(iq: np.ndarray, fc_hz: float = 900.0, order: int = 2,
                        prf: float = None) -> np.ndarray:
    """Zero-phase Butterworth low-pass of the IQ ensemble along slow time (frame axis).

    Suppresses high-frequency electronic/thermal noise between frames before displacement
    estimation. ``prf`` is injected by the pipeline. The corner must stay *above* the shear
    wave's temporal content (hundreds of Hz) or the wave itself is attenuated.
    """
    if not prf:
        return iq
    from scipy.signal import butter, filtfilt
    wn = min(max(fc_hz / (0.5 * prf), 1e-3), 0.99)
    b, a = butter(order, wn, btype="lowpass")
    n = iq.shape[0]
    if n <= 3 * max(len(a), len(b)):
        return iq
    r = filtfilt(b, a, iq.real, axis=0)
    i = filtfilt(b, a, iq.imag, axis=0)
    return r + 1j * i


def iq_slowtime_highpass(iq: np.ndarray, fc_hz: float = 40.0, order: int = 2,
                         prf: float = None) -> np.ndarray:
    """Zero-phase Butterworth **high-pass** of the IQ ensemble along slow time (frame axis).

    Removes low-frequency slow-time content (bulk/clutter) from the IQ **before** displacement
    estimation - an IQ-space complement to the displacement-space motion filters. ``prf`` is injected
    by the pipeline; keep the corner below the shear-wave temporal band so the wave survives.
    """
    if not prf:
        return iq
    from scipy.signal import butter, filtfilt
    wn = min(max(fc_hz / (0.5 * prf), 1e-3), 0.99)
    b, a = butter(order, wn, btype="highpass")
    n = iq.shape[0]
    if n <= 3 * max(len(a), len(b)):
        return iq - iq.mean(axis=0, keepdims=True)
    r = filtfilt(b, a, iq.real, axis=0)
    i = filtfilt(b, a, iq.imag, axis=0)
    return r + 1j * i


# ============================== displacement-space ==============================
def svd_clutter_field(field: np.ndarray, ctx: FilterCtx = None, n_remove: int = 1,
                      n_high_remove: int = 0) -> np.ndarray:
    """Low-rank (SVD/Casorati) clutter removal on the **displacement** movie.

    Reshape the (n_frames, nz, nx) field to a Casorati matrix (frames x space), zero the first
    ``n_remove`` singular components (spatially-coherent bulk cardiac motion, which dominates the
    low-order subspace) and optionally the last ``n_high_remove`` (incoherent noise), reconstruct.
    This is the Demene 2015 spatiotemporal clutter filter applied to displacement rather than IQ,
    and directly targets the coherent whole-wall motion that a fixed band-pass leaves behind.
    """
    n = field.shape[0]
    A = field.reshape(n, -1)
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    keep = np.ones_like(S)
    if n_remove > 0:
        keep[:n_remove] = 0.0
    if n_high_remove > 0:
        keep[-n_high_remove:] = 0.0
    return ((U * (S * keep)) @ Vh).reshape(field.shape)


def reference_subspace_projection(field: np.ndarray, ctx: FilterCtx = None, n_components: int = 3,
                                  basis: str = "spatial") -> np.ndarray:
    """Remove the cardiac-motion subspace **learned from the pre-push reference frames**.

    The reference frames contain pure cardiac motion (no shear wave). We take their displacement
    trajectory ``ctx.ref_disp`` (n_ref, nz, nx), build the dominant ``n_components`` singular
    subspace of that motion, and **project it out of the tracking field** - so whatever spatial
    (or temporal) patterns the cardiac motion occupied are removed, while the ARF shear wave
    (absent from the reference, hence orthogonal to its subspace) is preserved.

    ``basis="spatial"`` (default): the reference's dominant **spatial** modes ``V`` (nz*nx) are the
    cardiac motion's spatial footprints; each tracking frame is projected onto the complement of
    ``span(V_1..V_k)``. This is direction/'frequency'-agnostic - it removes the *shape* of cardiac
    motion, not a band. ``basis="temporal"`` instead removes the reference's dominant temporal
    modes (less physical for a short reference, kept for experimentation).

    Novel realisation of "use the reference to estimate cardiac motion better than a band-pass":
    a data-driven, reference-trained clutter filter. Needs ``ctx.ref_disp`` (in-vivo reference).
    """
    if ctx is None or ctx.ref_disp is None:
        raise ValueError("reference_subspace_projection requires ctx.ref_disp (in-vivo reference)")
    ref = np.asarray(ctx.ref_disp)
    n_ref = ref.shape[0]
    n = field.shape[0]
    R = ref.reshape(n_ref, -1)                      # (n_ref, space)
    R = R - R.mean(axis=0, keepdims=True)           # remove the static pre-push offset
    k = int(np.clip(n_components, 0, min(R.shape) - 1))
    if k <= 0:
        return field
    # Right singular vectors of the reference motion = its dominant spatial footprints.
    _, _, Vh = np.linalg.svd(R, full_matrices=False)
    A = field.reshape(n, -1)
    if basis == "spatial":
        Vk = Vh[:k]                                 # (k, space) orthonormal spatial modes
        coeff = A @ Vk.T                            # (n, k) projection of each frame
        return (A - coeff @ Vk).reshape(field.shape)
    # temporal basis: remove components of each pixel's trajectory along the reference's temporal
    # modes (uses the left singular vectors, resampled onto the tracking length by simple tiling).
    U, _, _ = np.linalg.svd(R, full_matrices=False)  # (n_ref, n_ref)
    Uk = U[:, :k]
    # least-squares-fit the tracking trajectory to the (interpolated) reference temporal modes
    tt = np.linspace(0, 1, n)
    tr = np.linspace(0, 1, n_ref)
    B = np.stack([np.interp(tt, tr, Uk[:, j]) for j in range(k)], axis=1)  # (n, k)
    B, _ = np.linalg.qr(B)
    coeff = B.T @ A                                 # (k, space)
    return (A - B @ coeff).reshape(field.shape)


def phase_unwrap_temporal(field: np.ndarray, ctx: FilterCtx = None,
                          ambiguity_um: float = 0.0) -> np.ndarray:
    """Unwrap phase-wrapped displacement along slow time (per pixel).

    Phase-based estimators (Loupas/Kasai) wrap when the true displacement exceeds half the
    displacement ambiguity ``a = c / (2 f_demod)`` (= lambda/2): the estimate jumps by ``a``. Large
    in-vivo cardiac motion wraps repeatedly, corrupting everything downstream. We convert the
    displacement back to an angle ``theta = 2*pi*d/a``, ``np.unwrap`` it along the frame axis, and
    convert back - restoring continuous large-motion trajectories so a subsequent motion filter can
    actually remove them.

    ``ambiguity_um`` overrides ``a`` (0 = derive from ``ctx.c`` / ``ctx.f_demod``). If those are
    unavailable and no override is given, the field is returned unchanged.
    """
    a = ambiguity_um * 1e-6
    if a <= 0:
        if ctx is not None and ctx.c and ctx.f_demod:
            a = ctx.c / (2.0 * ctx.f_demod)
        else:
            return field
    theta = 2.0 * np.pi * field / a
    unwrapped = np.unwrap(theta, axis=0)
    return unwrapped * a / (2.0 * np.pi)


# ============================== bulk cardiac-motion compensation (IQ) ==============================
def bulk_motion_compensation(iq: np.ndarray, reference=None, dz: float = None, dx: float = None,
                             max_shift_m: float = 2.0e-3, upsample: int = 10) -> np.ndarray:
    """Compensate **global (rigid) tissue translation** per frame before displacement estimation.

    Axial phase estimators (Loupas/Kasai) see only axial motion and conflate bulk cardiac translation
    with the shear-wave displacement. Here each frame's global (axial + lateral) shift relative to the
    pre-push reference (mean envelope) is estimated by sub-pixel **phase cross-correlation** and undone
    with a Fourier shift on the complex IQ - removing the whole-tissue translation while leaving the
    local wave. This is the simplest bulk-motion model (a global rigid shift); affine / optical-flow
    background models are the natural extensions (see docs/literature_review.md).

    ``reference`` (the pre-push ensemble) is injected by the pipeline; without it the first frame is
    used as the reference. Shifts are clamped to ``max_shift_m``.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift

    iq = np.asarray(iq)
    n, nz, nx = iq.shape
    if reference is not None:
        ref = np.asarray(reference)
        ref = ref.mean(axis=0) if ref.ndim == 3 else ref
    else:
        ref = iq[0]
    ref_env = np.abs(ref)
    max_z = (max_shift_m / dz) if dz else nz
    max_x = (max_shift_m / dx) if dx else nx
    out = np.empty_like(iq)
    for i in range(n):
        shift, _, _ = phase_cross_correlation(ref_env, np.abs(iq[i]), upsample_factor=upsample,
                                              normalization=None)
        shift = np.array([np.clip(shift[0], -max_z, max_z), np.clip(shift[1], -max_x, max_x)])
        out[i] = np.fft.ifft2(fourier_shift(np.fft.fft2(iq[i]), -shift))
    return out


# ============================== advanced spatial denoisers (displacement) =========================
def _per_frame(field, fn):
    """Apply a 2-D (nz, nx) denoiser to every frame of a (n_frames, nz, nx) field, in um then back."""
    scale = 1e6
    out = np.empty_like(field)
    for i in range(field.shape[0]):
        out[i] = fn(field[i] * scale) / scale
    return out


def aniso_diffusion(field: np.ndarray, ctx: FilterCtx = None, n_iter: int = 5, kappa_um: float = 5.0,
                    gamma: float = 0.2) -> np.ndarray:
    """Perona-Malik **anisotropic diffusion** per frame: edge-preserving smoothing that diffuses
    along, but not across, wavefront edges (stops at gradients >> ``kappa``). ``n_iter`` steps,
    ``kappa_um`` the gradient scale (in um), ``gamma`` the step size (<0.25 for stability)."""
    def pm(img):
        out = img.astype(np.float64).copy()
        for _ in range(int(n_iter)):
            dN = np.vstack([out[:1], out[:-1]]) - out
            dS = np.vstack([out[1:], out[-1:]]) - out
            dW = np.hstack([out[:, :1], out[:, :-1]]) - out
            dE = np.hstack([out[:, 1:], out[:, -1:]]) - out
            g = lambda d: np.exp(-(d / (kappa_um + 1e-9)) ** 2)   # noqa: E731
            out = out + gamma * (g(dN) * dN + g(dS) * dS + g(dW) * dW + g(dE) * dE)
        return out
    return _per_frame(field, pm)


def coherence_diffusion(field: np.ndarray, ctx: FilterCtx = None, n_iter: int = 5,
                        sigma: float = 1.0, rho: float = 3.0, gamma: float = 0.15) -> np.ndarray:
    """Coherence-enhancing (**tensor-guided**) diffusion per frame (Weickert): smooth *along* the
    dominant local orientation (the wavefront), sharpening coherent line structures. Builds the
    structure tensor (gradients smoothed at ``sigma``, tensor smoothed at ``rho``), and diffuses
    preferentially along its principal (small-eigenvalue) direction. ``n_iter`` explicit steps."""
    from scipy.ndimage import gaussian_filter

    def ced(img):
        u = img.astype(np.float64).copy()
        for _ in range(int(n_iter)):
            gx = gaussian_filter(u, sigma, order=(0, 1), mode="nearest")
            gz = gaussian_filter(u, sigma, order=(1, 0), mode="nearest")
            Jxx = gaussian_filter(gx * gx, rho, mode="nearest")
            Jxz = gaussian_filter(gx * gz, rho, mode="nearest")
            Jzz = gaussian_filter(gz * gz, rho, mode="nearest")
            # eigen-decomposition of the 2x2 structure tensor (per pixel)
            tr = Jxx + Jzz
            det = Jxx * Jzz - Jxz ** 2
            disc = np.sqrt(np.maximum((Jxx - Jzz) ** 2 + 4 * Jxz ** 2, 0.0))
            mu1 = 0.5 * (tr + disc)                     # larger eigenvalue (across-structure)
            # principal direction (small eigenvalue) = along the structure
            coh = ((mu1 - 0.5 * (tr - disc)) / (tr + 1e-12)) ** 2
            # diffusivities: strong along structure, weak across (alpha small)
            alpha = 0.001
            lam_along = alpha + (1 - alpha) * np.exp(-1.0 / (coh + 1e-9))
            lam_across = alpha
            # orientation of the along-structure eigenvector
            theta = 0.5 * np.arctan2(2 * Jxz, Jxx - Jzz)
            cth, sth = np.cos(theta), np.sin(theta)
            Dxx = lam_along * cth ** 2 + lam_across * sth ** 2
            Dzz = lam_along * sth ** 2 + lam_across * cth ** 2
            Dxz = (lam_along - lam_across) * cth * sth
            ux = gaussian_filter(u, 0.4, order=(0, 1), mode="nearest")
            uz = gaussian_filter(u, 0.4, order=(1, 0), mode="nearest")
            jx = Dxx * ux + Dxz * uz
            jz = Dxz * ux + Dzz * uz
            div = (gaussian_filter(jx, 0.4, order=(0, 1), mode="nearest")
                   + gaussian_filter(jz, 0.4, order=(1, 0), mode="nearest"))
            u = u + gamma * div
        return u
    return _per_frame(field, ced)


def bilateral_denoise(field: np.ndarray, ctx: FilterCtx = None, sigma_color_um: float = 3.0,
                      sigma_spatial_px: float = 2.0) -> np.ndarray:
    """Edge-preserving **bilateral** filter per frame (skimage): averages neighbours weighted by both
    spatial distance (``sigma_spatial_px``) and value similarity (``sigma_color_um``)."""
    from skimage.restoration import denoise_bilateral

    def bil(img):
        return denoise_bilateral(img, sigma_color=sigma_color_um, sigma_spatial=sigma_spatial_px,
                                 channel_axis=None)
    return _per_frame(field, bil)


def nlm_denoise(field: np.ndarray, ctx: FilterCtx = None, h_um: float = 3.0, patch_size: int = 5,
                patch_distance: int = 6) -> np.ndarray:
    """**Non-local means** per frame (skimage): denoise by averaging similar patches across the frame;
    excellent for repetitive speckle-like texture. ``h_um`` sets the filter strength (in um)."""
    from skimage.restoration import denoise_nl_means

    def nlm(img):
        return denoise_nl_means(img, h=h_um, patch_size=int(patch_size),
                                patch_distance=int(patch_distance), channel_axis=None, fast_mode=True)
    return _per_frame(field, nlm)


# ============================== quality masking + Savitzky-Golay ==============================
def quality_mask(field: np.ndarray, ctx: FilterCtx = None, min_db: float = -40.0,
                 min_coherence: float = 0.0, soft: bool = True) -> np.ndarray:
    """Down-weight / remove pixels with **low B-mode intensity or poor RF coherence** (unstable phase).

    Uses the pipeline-populated ``ctx.bmode_db`` (reference envelope, dB, peak 0) and ``ctx.coherence``
    (slow-time lag-1 coherence in [0,1]). Pixels below ``min_db`` **or** ``min_coherence`` are removed.
    ``soft=True`` multiplies the displacement by a smooth 0..1 weight (a confidence weighting, so the
    M-line averaging naturally discounts them); ``soft=False`` hard-zeros them. Broadcast over frames.
    """
    if ctx is None or ctx.bmode_db is None:
        return field
    w = np.ones_like(ctx.bmode_db, dtype=float)
    if soft:
        # smooth ramps over ~10 dB and ~0.1 coherence around the thresholds
        w_db = np.clip((ctx.bmode_db - min_db) / 10.0 + 0.5, 0.0, 1.0)
        w = w_db
        if ctx.coherence is not None and min_coherence > 0:
            w = w * np.clip((ctx.coherence - min_coherence) / 0.1 + 0.5, 0.0, 1.0)
    else:
        w = (ctx.bmode_db >= min_db).astype(float)
        if ctx.coherence is not None and min_coherence > 0:
            w = w * (ctx.coherence >= min_coherence)
    return field * w[None, :, :]


def savgol_temporal(field: np.ndarray, ctx: FilterCtx = None, window: int = 7,
                    polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay temporal filter along slow time (per pixel).

    Fits a ``polyorder`` polynomial in a sliding ``window`` of frames - suppresses frame jitter while
    preserving the **timing and amplitude of the shear-wave transient** better than a moving average
    (which broadens peaks). ``window`` is forced odd and > ``polyorder``.
    """
    from scipy.signal import savgol_filter
    w = int(window) | 1                                    # force odd
    n = field.shape[0]
    if n < w or w <= polyorder:
        return field
    return savgol_filter(field, window_length=w, polyorder=int(polyorder), axis=0, mode="nearest")
