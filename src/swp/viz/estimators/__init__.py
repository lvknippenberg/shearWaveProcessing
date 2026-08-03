from .common import DisplacementResult
from .loupas import loupas_displacement
from .kasai import kasai_displacement
from .xcorr import xcorr_displacement

# registry: name -> callable(iq, dz, dx, c, f_demod, prf, mode, reference, **params)
ESTIMATORS = {
    "loupas": loupas_displacement,
    "kasai": kasai_displacement,
    "xcorr": xcorr_displacement,
}

__all__ = [
    "DisplacementResult", "loupas_displacement", "kasai_displacement",
    "xcorr_displacement", "ESTIMATORS",
]
