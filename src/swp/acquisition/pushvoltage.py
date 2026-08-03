"""Read the ARF push voltage actually delivered in a measurement.

The push high voltage is set by a Verasonics TPC (Transmit Power Controller) *profile*.
A voltage sweep changes that profile's ``hv`` per acquisition, but the hardware clamps it
to the profile's ``highVoltageLimit`` / ``maxHighVoltage`` -- so a requested 50 V can be
silently delivered as, say, 30 V. This module reads the *delivered* value straight from the
runtime ``AcquisitionParametersAndECG.mat`` (or ``CombinedData.mat``) so an analysis labels
measurements by what actually happened, and a sweep that clamped is caught immediately.

Only scipy is needed (the TPC struct lives in the v5 runtime .mat); no zea / GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# TPC profile (1-based, as in MATLAB ``TPC(k)``) whose ``hv`` is the ARF push voltage.
# Profile 5 is the SWI push/high-power profile in the S5-1 SWI sequence.
PUSH_TPC_PROFILE = 5

# Runtime .mat holding the dynamic TPC (preferred: the small file that always exists).
_MAT_NAMES = ("AcquisitionParametersAndECG.mat", "CombinedData.mat")


@dataclass
class PushVoltage:
    """The push voltage delivered for one measurement, with its profile limits."""
    hv: Optional[float]                 # delivered push high voltage [V]
    high_voltage_limit: Optional[float]  # profile highVoltageLimit [V]
    max_high_voltage: Optional[float]    # profile maxHighVoltage [V]
    profile: int = PUSH_TPC_PROFILE
    source: Optional[str] = None         # .mat the value came from

    @property
    def at_profile_max(self) -> bool:
        """True when the delivered voltage sits at the profile's ceiling (likely clamped)."""
        lim = self._ceiling()
        return self.hv is not None and lim is not None and abs(self.hv - lim) < 0.5

    def _ceiling(self) -> Optional[float]:
        vals = [v for v in (self.high_voltage_limit, self.max_high_voltage) if v is not None]
        return min(vals) if vals else None

    def label(self) -> str:
        if self.hv is None:
            return "? V"
        return f"{self.hv:.0f} V" + (" (profile max)" if self.at_profile_max else "")


def _find_mat(folder_or_mat) -> Optional[Path]:
    p = Path(folder_or_mat)
    if p.is_file() and p.suffix == ".mat":
        return p
    if p.is_dir():
        for name in _MAT_NAMES:
            if (p / name).is_file():
                return p / name
    return None


def read_push_voltage(folder_or_mat, profile: int = PUSH_TPC_PROFILE) -> PushVoltage:
    """Read the delivered push voltage for a measurement folder (or a .mat directly)."""
    mat = _find_mat(folder_or_mat)
    if mat is None:
        return PushVoltage(None, None, None, profile, None)
    try:
        import scipy.io as sio
        m = sio.loadmat(str(mat), squeeze_me=True, struct_as_record=False)
        tpc = np.atleast_1d(m["TPC"])[profile - 1]

        def _f(field):
            try:
                return float(getattr(tpc, field))
            except Exception:
                return None

        return PushVoltage(_f("hv"), _f("highVoltageLimit"), _f("maxHighVoltage"),
                           profile, mat.name)
    except Exception:
        return PushVoltage(None, None, None, profile, mat.name)


def discover_measurements(root, profile: int = PUSH_TPC_PROFILE) -> List[tuple]:
    """Every measurement subfolder under ``root`` with its delivered push voltage.

    Returns ``[(folder_path, PushVoltage), ...]`` sorted by folder name (the folder name
    carries the acquisition timestamp, so this is acquisition order = the sweep order).
    A measurement folder is any immediate subdirectory that contains a runtime .mat.
    """
    root = Path(root)
    subs = [d for d in sorted(root.iterdir()) if d.is_dir() and _find_mat(d) is not None]
    return [(str(d), read_push_voltage(d, profile)) for d in subs]
