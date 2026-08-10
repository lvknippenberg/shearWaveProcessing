"""Draggable manual-speed line: a tiny Streamlit component wrapping a Plotly editable line shape.

The whole drag happens in-browser (no server round-trip per move); the endpoints are committed back
to Python only on mouse-release, so there is no per-click pause. Dragging an **endpoint** moves a single
point; dragging the **line body** moves the whole line. Live speed is shown in the browser while dragging.

Returns the current line as ``[[r0_mm, t0_ms], [r1_mm, t1_ms]]`` (or the ``init_line`` default until the
user drags).
"""
import os

import streamlit.components.v1 as components

_DIR = os.path.dirname(os.path.abspath(__file__))
_impl = components.declare_component("swp_speedline", path=_DIR)


def speed_line_picker(*, img, x0, x1, y0, y1, init_lines=None, reset_token=0, height=380, key=None):
    """Draggable manual-speed lines over a fixed space-time backdrop.

    ``img`` is a PNG data URI of the space-time (rendered by matplotlib so the colours match the other
    plots, with the fixed r0 line baked in); ``x0/x1`` are the r-axis range (mm) and ``y0/y1`` the t-axis
    range (ms, y0 = earliest at the top). The plot starts **empty**: click-drag to draw a wavefront line,
    add more with the '＋ add line' button; drag an endpoint (one point) or the line body (both points) to
    adjust any line. ``reset_token``, when changed, snaps back to ``init_lines`` ([] = empty) - bump it on
    a 'clear' button / push change. Returns the list of lines ``[[[r0,t0],[r1,t1]], ...]`` (possibly [])."""
    init_lines = init_lines or []
    return _impl(img=img, x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1),
                 init_lines=init_lines, reset_token=str(reset_token), height=int(height),
                 key=key, default=init_lines)
