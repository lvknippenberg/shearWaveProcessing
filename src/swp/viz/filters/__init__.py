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

# field filters that need the pre-push reference trajectory (ctx.ref_disp / t_ref)
REFERENCE_FILTERS = {"reference_motion_comp", "adaptive_highpass"}

# IQ-space filters: (iq_complex, **params) -> iq_complex
IQ_FILTERS = {
    "svd_clutter": svd_clutter,
}

# displacement/velocity-space filters on (n_frames, nz, nx): (field, ctx, **params) -> field
FIELD_FILTERS = {
    "polynomial_drift": polynomial_drift,
    "temporal_highpass": temporal_highpass,
    "reference_motion_comp": reference_motion_compensation,
    "adaptive_highpass": adaptive_highpass,
    "axial_strain": axial_strain,
    "spatial_smooth": spatial_smooth,
    "spatial_median": spatial_median,
    "temporal_moving_mean": temporal_moving_mean,
    "temporal_moving_median": temporal_moving_median,
    "temporal_bandpass": temporal_bandpass,
    "directional_field": directional_field,
}

__all__ = [
    "svd_clutter", "polynomial_drift", "temporal_highpass", "axial_strain",
    "reference_motion_compensation", "adaptive_highpass",
    "directional_spacetime", "outward_spacetime", "directional_field",
    "spatial_smooth", "spatial_median", "temporal_moving_mean", "temporal_moving_median",
    "temporal_bandpass", "IQ_FILTERS", "FIELD_FILTERS", "REFERENCE_FILTERS",
]
