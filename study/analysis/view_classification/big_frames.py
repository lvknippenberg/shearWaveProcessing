import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from contact_sheet import gif_frames
subj = Path(sys.argv[1]); stamps = sys.argv[2].split(",")
tiles = []
for s in stamps:
    g = next(subj.glob(f"*{s}/output/CombinedData_buffer3_iq.gif"))
    fr = gif_frames(g)
    tiles.append(np.concatenate([fr[0], fr[10], fr[20]], axis=1))
img = np.concatenate(tiles, axis=0)
Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(sys.argv[3])
print(img.shape)
