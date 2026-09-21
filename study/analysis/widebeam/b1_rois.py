"""Hand-drawn evaluation regions, read off the baseline reconstruction.

Hand-drawn, and therefore stated openly: they are the weakest link in the in-vivo numbers,
which is why every in-vivo verdict is cross-checked against the phantom (wire targets, a
lesion) and against the anatomy-free off-beam energy measurement.

Each entry: (name, (x0_mm, x1_mm), (z0_mm, z1_mm), colour).
"""

INVIVO_FRAME = 0

INVIVO = [
    # dark (blood-pool / chamber) regions - should be echo-free; whatever fills them is
    # clutter, reverberation or noise.
    ("dark_mid", (-26, -2), (80, 92), "r"),
    ("dark_upper", (2, 22), (54, 64), "r"),
    # tissue (myocardium / chest wall) - real backscatter.
    ("tissue_deep", (-26, 4), (96, 108), "lime"),
    ("tissue_near", (-18, 10), (26, 40), "lime"),
    # speckle-only region for speckle SNR (uniform, no specular boundary crossing it)
    ("speckle", (-16, 8), (30, 42), "yellow"),
]

PHANTOM = [
    # hyperechoic inclusion at ~ (24, 116) mm and matched background at the same depth
    ("lesion", (19, 30), (110, 122), "r"),
    ("lesion_bg", (-18, -7), (110, 122), "lime"),
    # uniform speckle, clear of the wire column at x ~ +14 mm
    ("speckle", (-30, -14), (45, 62), "yellow"),
]


def by_name(rois):
    return {r[0]: r for r in rois}
