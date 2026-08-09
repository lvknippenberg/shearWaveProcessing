"""Swappable IQ-space and displacement-space filters (M3 motion, M4 wavefield).

Each filter is a small pure function.  Registries map names -> callables so pipelines can be
described as data (see ``iq2sws.pipeline``).  Filters take a field plus keyword params and
return a filtered field of the same shape (except transforms like strain).
"""
from .clutter import svd_clutter
from .motion import (polynomial_drift, temporal_highpass, axial_strain,
                     reference_motion_compensation, adaptive_highpass)
from .directional import directional_spacetime, outward_spacetime, directional_field
from .spatial import (spatial_smooth, spatial_median, temporal_bandpass,
                      temporal_moving_mean, temporal_moving_median)
from .experimental import (iq_spatial_lowpass, iq_slowtime_lowpass, iq_slowtime_highpass,
                           svd_clutter_field, reference_subspace_projection, phase_unwrap_temporal,
                           bulk_motion_compensation, optical_flow_compensation, bulk_displacement_removal,
                           aniso_diffusion, coherence_diffusion,
                           bilateral_denoise, nlm_denoise, quality_mask, savgol_temporal)

# field filters that need the pre-push reference trajectory (ctx.ref_disp / t_ref)
REFERENCE_FILTERS = {"reference_motion_comp", "adaptive_highpass", "reference_subspace_projection"}

# IQ-space filters: (iq_complex, **params) -> iq_complex. Those accepting `reference` are given the
# pre-push ensemble by the pipeline (bulk_motion_compensation).
IQ_FILTERS = {
    "svd_clutter": svd_clutter,
    "iq_spatial_lowpass": iq_spatial_lowpass,
    "iq_slowtime_lowpass": iq_slowtime_lowpass,
    "iq_slowtime_highpass": iq_slowtime_highpass,
    "bulk_motion_compensation": bulk_motion_compensation,
    "optical_flow_compensation": optical_flow_compensation,
}

# displacement/velocity-space filters on (n_frames, nz, nx): (field, ctx, **params) -> field
FIELD_FILTERS = {
    "polynomial_drift": polynomial_drift,
    "temporal_highpass": temporal_highpass,
    "reference_motion_comp": reference_motion_compensation,
    "adaptive_highpass": adaptive_highpass,
    "reference_subspace_projection": reference_subspace_projection,
    "svd_clutter_field": svd_clutter_field,
    "bulk_displacement_removal": bulk_displacement_removal,
    "phase_unwrap_temporal": phase_unwrap_temporal,
    "axial_strain": axial_strain,
    "quality_mask": quality_mask,
    "spatial_smooth": spatial_smooth,
    "spatial_median": spatial_median,
    "aniso_diffusion": aniso_diffusion,
    "coherence_diffusion": coherence_diffusion,
    "bilateral_denoise": bilateral_denoise,
    "nlm_denoise": nlm_denoise,
    "temporal_moving_mean": temporal_moving_mean,
    "temporal_moving_median": temporal_moving_median,
    "temporal_bandpass": temporal_bandpass,
    "savgol_temporal": savgol_temporal,
    "directional_field": directional_field,
}

__all__ = [
    "svd_clutter", "polynomial_drift", "temporal_highpass", "axial_strain",
    "reference_motion_compensation", "adaptive_highpass",
    "iq_spatial_lowpass", "iq_slowtime_lowpass", "svd_clutter_field",
    "reference_subspace_projection", "phase_unwrap_temporal", "bulk_motion_compensation",
    "aniso_diffusion", "coherence_diffusion", "bilateral_denoise", "nlm_denoise",
    "directional_spacetime", "outward_spacetime", "directional_field",
    "spatial_smooth", "spatial_median", "temporal_moving_mean", "temporal_moving_median",
    "temporal_bandpass", "IQ_FILTERS", "FIELD_FILTERS", "REFERENCE_FILTERS",
]
