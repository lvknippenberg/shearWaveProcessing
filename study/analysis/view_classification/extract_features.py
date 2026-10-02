"""Cache label-free EchoPrime features for every buffer-3 loop (input to the unsupervised sorting).

Per loop (buffer-3 GIF, 26 or 32 frames, ~1 cardiac cycle):
  frame_feat  (N, 1024)  ConvNeXt-base penultimate features of EchoPrime's view classifier, per frame
  frame_prob  (N, 11)    its softmax per frame (COARSE_VIEWS order)
  video_emb   (C, 512)   EchoPrime video encoder (MViT-v2-S, 16 frames) embedding, for C clips that
                         sample the whole loop with different start frames (loops are not ECG-gated)
Preprocessing is EchoPrime's (centre square crop, 10 % zoom crop, 224 px bicubic, its mean/std).

Usage:
  python extract_features.py --views <views.csv with a gif column> --weights <EchoPrime weights dir> --out feats.npz
Only reads the GIFs.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).parent))
from echoprime_views import COARSE_VIEWS, MEAN, STD, crop_and_scale, gif_frames   # noqa: E402

N_CLIP_FRAMES = 16


def load_models(wdir, device):
    vc = torchvision.models.convnext_base()
    vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, len(COARSE_VIEWS))
    vc.load_state_dict(torch.load(Path(wdir) / "view_classifier.pt", map_location="cpu"))
    enc = torchvision.models.video.mvit_v2_s()
    enc.head[-1] = torch.nn.Linear(enc.head[-1].in_features, 512)
    enc.load_state_dict(torch.load(Path(wdir) / "echo_prime_encoder.pt", map_location="cpu"))
    return vc.to(device).eval(), enc.to(device).eval()


def clip_indices(n, n_clips=4):
    """16 frames spread over the whole loop, start shifted by a fraction of the step per clip."""
    step = (n - 1) / (N_CLIP_FRAMES - 1)
    return [np.round(np.linspace(c * step / n_clips, n - 1, N_CLIP_FRAMES)).astype(int) for c in range(n_clips)]


@torch.no_grad()
def features(vc, enc, gif, device):
    fr = gif_frames(gif)
    x = torch.from_numpy(np.stack([crop_and_scale(f) for f in fr]))[:, None].repeat(1, 3, 1, 1)
    x = ((x - MEAN) / STD).to(device)                                   # (N, 3, 224, 224)
    pen = vc.classifier[:-1](vc.avgpool(vc.features(x)))                # LayerNorm2d + flatten -> (N, 1024)
    prob = torch.softmax(vc.classifier[-1](pen), 1)
    clips = torch.stack([x[idx].permute(1, 0, 2, 3) for idx in clip_indices(len(x))])   # (C, 3, 16, H, W)
    emb = enc(clips)
    return pen.half().cpu().numpy(), prob.cpu().numpy(), emb.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.views)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vc, enc = load_models(a.weights, device)
    ff, fp, ve, nf = [], [], [], []
    t0 = time.time()
    for i, g in enumerate(d.gif):
        f, p, e = features(vc, enc, g, device)
        ff.append(f); fp.append(p); ve.append(e); nf.append(len(f))
        if i % 50 == 0:
            print(f"{i}/{len(d)}  {time.time() - t0:.0f} s", flush=True)
    off = np.concatenate([[0], np.cumsum(nf)])
    np.savez(a.out, subject=np.array(list(d.subject), dtype=str), folder=np.array(list(d.folder), dtype=str),
             frame_offsets=off, frame_feat=np.concatenate(ff), frame_prob=np.concatenate(fp),
             video_emb=np.stack(ve), classes=np.array(COARSE_VIEWS))
    print(a.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
