"""Classify the echo view (PLAX / PSAX / Apical / Other) of buffer-3 GIFs with EchoPrime's view classifier.

EchoPrime (Vukadinovic et al. 2024, github.com/echonet/EchoPrime, Cedars-Sinai academic licence)
ships an 11-class ConvNeXt-base single-frame view classifier (model_data/weights/view_classifier.pt).
It was trained on clinical scan-converted DICOM frames; we feed it the scan-converted buffer-3 GIF
frames with the same preprocessing (centre square crop, 10 % zoom crop, 224x224 bicubic, its
mean/std), classify EVERY frame and average the softmax over the loop, then pool the 11 classes into
the 3 groups we care about.

Usage:
  python echoprime_views.py --weights <view_classifier.pt> --out views.csv GIF [GIF ...]
  python echoprime_views.py --weights ... --out views.csv --root Z:\\raw_data\\C000000049 [--root <local copy>]
(--root globs */output/CombinedData_buffer3_iq.gif one level below each root.)
Only reads the GIFs.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

COARSE_VIEWS = ['A2C', 'A3C', 'A4C', 'A5C', 'Apical_Doppler', 'Doppler_Parasternal_Long',
                'Doppler_Parasternal_Short', 'Parasternal_Long', 'Parasternal_Short', 'SSN', 'Subcostal']
GROUPS = {
    "PLAX": ['Parasternal_Long', 'Doppler_Parasternal_Long'],
    "PSAX": ['Parasternal_Short', 'Doppler_Parasternal_Short'],
    "Apical": ['A2C', 'A3C', 'A4C', 'A5C', 'Apical_Doppler'],
    "Other": ['SSN', 'Subcostal'],
}
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(1, 3, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(1, 3, 1, 1)


def gif_frames(path):
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.asarray(im.convert("L")))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return np.stack(frames)


def crop_and_scale(img, res=224, zoom=0.1):
    """EchoPrime utils.crop_and_scale for a square output: centre square crop, zoom crop, resize."""
    h, w = img.shape
    if w > h:
        p = int(round((w - h) / 2))
        img = img[:, p:-p]
    elif h > w:
        p = int(round((h - w) / 2))
        img = img[p:-p]
    py, px = round(int(img.shape[0] * zoom)), round(int(img.shape[1] * zoom))
    img = img[py:-py, px:-px]
    return np.asarray(Image.fromarray(img).resize((res, res), Image.BICUBIC), dtype=np.float32)


def load_model(weights, device):
    m = torchvision.models.convnext_base()
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, len(COARSE_VIEWS))
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def classify(model, gif, device):
    fr = gif_frames(gif)
    x = torch.from_numpy(np.stack([crop_and_scale(f) for f in fr]))[:, None].repeat(1, 3, 1, 1)
    x = (x - MEAN) / STD
    p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()      # (frames, 11)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gifs", nargs="*")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gifs = [Path(g) for g in a.gifs]
    for r in a.root:
        gifs += sorted(Path(r).glob("*/output/CombinedData_buffer3_iq.gif"))
    # per subject, in acquisition order (folder names end in <date>_<HH-MM-SS>)
    gifs = sorted(gifs, key=lambda g: (g.parents[2].name, g.parent.parent.name.split("_")[-1]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.weights, device)

    rows = []
    for g in gifs:
        p = classify(model, g, device)
        mean = p.mean(0)
        grp = {k: float(sum(mean[COARSE_VIEWS.index(v)] for v in vs)) for k, vs in GROUPS.items()}
        # per-frame group votes: how stable the call is over the loop
        frame_grp = np.stack([[sum(pf[COARSE_VIEWS.index(v)] for v in vs) for vs in GROUPS.values()]
                              for pf in p]).argmax(1)
        best = max(grp, key=grp.get)
        agree = float(np.mean(np.array(list(GROUPS))[frame_grp] == best))
        row = {"subject": g.parents[2].name, "folder": g.parent.parent.name, "view": best, "p_view": round(grp[best], 3),
               "frame_agreement": round(agree, 3), "top_fine": COARSE_VIEWS[int(mean.argmax())],
               **{f"p_{k}": round(v, 3) for k, v in grp.items()},
               **{f"p_{v}": round(float(mean[i]), 3) for i, v in enumerate(COARSE_VIEWS)},
               "gif": str(g)}
        rows.append(row)
        print(f"{row['subject']} {row['folder']:<48} {best:<7} p={grp[best]:.2f} frames={agree:.2f} fine={row['top_fine']}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(a.out)


if __name__ == "__main__":
    main()
