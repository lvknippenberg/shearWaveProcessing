"""Interactive M-line selection + persistence, ported from SWI/Zea/swi_mline.py.

Only the *selection* half is used by this repo (the space-time computation is done by
:mod:`swp.viz`). The saved ``.npz`` (``points`` (k,2)=(x,z) m + ``n_samples``) is directly
loadable by ``swp.viz.io.load_mline`` / ``mline_from_points``, so selection here and
sampling in the viz core interoperate through the shared file.
"""

from .select import (
    MLine,
    MLineConfig,
    fit_spline,
    select_mline,
    get_mline,
    save_mline,
    load_mline,
    mline_store_path,
    load_bmode_frame,
    draw_mline_on_bmode,
    track_mline_cine,
    snap_to_band,
)

__all__ = [
    "MLine", "MLineConfig", "fit_spline", "select_mline", "get_mline",
    "save_mline", "load_mline", "mline_store_path", "load_bmode_frame",
    "draw_mline_on_bmode", "track_mline_cine", "snap_to_band",
]
