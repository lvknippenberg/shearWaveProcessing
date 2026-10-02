"""EchoPrime features of a buffer-3 loop (the input of every voter).

EchoPrime ships an 11-class single-frame view classifier (ConvNeXt-base) and a video encoder
(MViT-v2-S, 16 frames -> 512-d). Both were trained on clinical scan-converted DICOM frames; the
buffer-3 GIF frames get EchoPrime's own preprocessing (centre square crop, 10 % zoom crop, 224 px
bicubic, its mean/std).

Per loop (26 or 32 frames, about one cardiac cycle, not ECG-gated):
  frame_feat  (N, 1024) float16  classifier penultimate features, per frame
  frame_prob  (N, 11)            classifier softmax, per frame (``COARSE_VIEWS`` order)
  video_emb   (4, 512)           video embeddings of 4 clips of 16 frames spanning the loop with
                                 different start offsets
Weights: ``download_weights()`` (1.3 GB archive from the v1.0.0 release, two files kept, ~490 MB).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import CACHE_DIR, WEIGHTS_DIR

COARSE_VIEWS = ['A2C', 'A3C', 'A4C', 'A5C', 'Apical_Doppler', 'Doppler_Parasternal_Long',
                'Doppler_Parasternal_Short', 'Parasternal_Long', 'Parasternal_Short', 'SSN', 'Subcostal']
GROUPS = {"PLAX": ['Parasternal_Long', 'Doppler_Parasternal_Long'],
          "PSAX": ['Parasternal_Short', 'Doppler_Parasternal_Short'],
          "Apical": ['A2C', 'A3C', 'A4C', 'A5C', 'Apical_Doppler'],
          "Other": ['SSN', 'Subcostal']}
_MEAN = (29.110628, 28.076836, 29.096405)
_STD = (47.989223, 46.456997, 47.20083)
N_CLIP_FRAMES = 16
N_CLIPS = 4
RELEASE_ZIP = "https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip"
WEIGHT_FILES = ("view_classifier.pt", "echo_prime_encoder.pt")


def download_weights(dest: Path = WEIGHTS_DIR) -> Path:
    """Fetch EchoPrime's release archive and keep only the two weight files used here."""
    dest = Path(dest)
    if all((dest / f).is_file() for f in WEIGHT_FILES):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RELEASE_ZIP} (1.3 GB) ...", flush=True)
    with urllib.request.urlopen(RELEASE_ZIP) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for f in WEIGHT_FILES:
            (dest / f).write_bytes(z.read(f"model_data/weights/{f}"))
    return dest


def gif_frames(path) -> np.ndarray:
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.asarray(im.convert("L")))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return np.stack(frames)


def crop_and_scale(img: np.ndarray, res: int = 224, zoom: float = 0.1) -> np.ndarray:
    """EchoPrime ``utils.crop_and_scale`` for a square output (cv2-free)."""
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


def clip_indices(n: int, n_clips: int = N_CLIPS):
    """16 frames spread over the whole loop, the start shifted by a fraction of the step per clip."""
    step = (n - 1) / (N_CLIP_FRAMES - 1)
    return [np.round(np.linspace(c * step / n_clips, n - 1, N_CLIP_FRAMES)).astype(int) for c in range(n_clips)]


class Extractor:
    def __init__(self, weights_dir: Path = WEIGHTS_DIR, device: str | None = None):
        import torch
        import torchvision
        wd = Path(weights_dir)
        missing = [f for f in WEIGHT_FILES if not (wd / f).is_file()]
        if missing:
            raise FileNotFoundError(f"EchoPrime weights {missing} not in {wd}: run "
                                    f"`python scripts/view_sort.py download-weights` (or set SWP_ECHOPRIME_DIR)")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vc = torchvision.models.convnext_base()
        vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, len(COARSE_VIEWS))
        vc.load_state_dict(torch.load(wd / "view_classifier.pt", map_location="cpu"))
        enc = torchvision.models.video.mvit_v2_s()
        enc.head[-1] = torch.nn.Linear(enc.head[-1].in_features, 512)
        enc.load_state_dict(torch.load(wd / "echo_prime_encoder.pt", map_location="cpu"))
        self.vc, self.enc = vc.to(self.device).eval(), enc.to(self.device).eval()
        self.mean = torch.tensor(_MEAN).reshape(1, 3, 1, 1)
        self.std = torch.tensor(_STD).reshape(1, 3, 1, 1)

    def __call__(self, gif) -> dict:
        torch = self.torch
        fr = gif_frames(gif)
        with torch.no_grad():
            x = torch.from_numpy(np.stack([crop_and_scale(f) for f in fr]))[:, None].repeat(1, 3, 1, 1)
            x = ((x - self.mean) / self.std).to(self.device)                # (N, 3, 224, 224)
            pen = self.vc.classifier[:-1](self.vc.avgpool(self.vc.features(x)))   # (N, 1024)
            prob = torch.softmax(self.vc.classifier[-1](pen), 1)
            clips = torch.stack([x[i].permute(1, 0, 2, 3) for i in clip_indices(len(x))])
            emb = self.enc(clips)
        return {"frame_feat": pen.half().cpu().numpy(), "frame_prob": prob.cpu().numpy(),
                "video_emb": emb.cpu().numpy()}


def cache_path(subject: str, folder: str, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / subject / f"{folder}.npz"


def load_cached(subject: str, folder: str, cache_dir: Path = CACHE_DIR) -> dict | None:
    p = cache_path(subject, folder, cache_dir)
    if not p.is_file():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def save_cached(subject: str, folder: str, feats: dict, cache_dir: Path = CACHE_DIR) -> Path:
    p = cache_path(subject, folder, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **feats)
    return p
