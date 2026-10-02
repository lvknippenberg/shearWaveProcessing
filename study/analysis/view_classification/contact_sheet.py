"""Contact sheet of buffer-3 GIF frames, optionally labelled with the predicted view.

Usage:
  python contact_sheet.py Z:\\raw_data\\C000000049 [out.png]       # all SW folders of a subject
  python contact_sheet.py --csv views.csv [--subject C000000049] out.png   # tiles + labels from echoprime_views.py
Reads only the GIFs (nothing is written next to the data).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"PLAX": "tab:blue", "PSAX": "tab:orange", "Apical": "tab:green", "Other": "tab:red"}


def gif_frames(path):
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.asarray(im.convert("L"), dtype=np.float32))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="*")
    ap.add_argument("--csv")
    ap.add_argument("--subject")
    ap.add_argument("--cols", type=int, default=5)
    a = ap.parse_args()
    if a.csv:
        rows = [r for r in csv.DictReader(open(a.csv)) if not a.subject or r.get("subject") == a.subject]
        out = Path(a.args[0])
    else:
        subj = Path(a.args[0])
        rows = [{"folder": g.parent.parent.name, "gif": str(g)}
                for g in sorted(subj.glob("*SW_data*/output/CombinedData_buffer3_iq.gif"))]
        out = Path(a.args[1]) if len(a.args) > 1 else Path(__file__).with_name(f"{subj.name}_sheet.png")

    cols = a.cols
    nrows = int(np.ceil(len(rows) / cols))
    fig, axs = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 2.6), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), rows):
        fr = gif_frames(r["gif"])
        ax.imshow(fr[0][:, 60:-60], cmap="gray", vmin=0, vmax=255)
        kind = "Strain" if "strain" in r["folder"].lower() else "SW"
        title = f"{kind} {r['folder'][-8:]}"
        if "view" in r:
            title += f"\n{r['view']} p={float(r['p_view']):.2f} ({r['top_fine']})"
            ax.set_title(title, fontsize=8, color=COLOURS.get(r["view"], "k"))
        else:
            ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
