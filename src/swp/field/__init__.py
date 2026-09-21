"""Direction-resolved shear-wave speed estimation on the 2-D field.

See ``docs/field_estimator_plan.md``. The M-line estimator cannot separate speed from propagation
direction - the apparent speed along a line is ``c / cos(theta)`` - so every 1-D estimate is
biased high by an unknown per-window factor. Working on the field makes direction an output.
"""
from .structure import FieldEstimate, aggregate, estimate_field          # noqa: F401
