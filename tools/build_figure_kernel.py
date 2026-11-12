"""Kaggle kernel that renders the qualitative report figure: two held-out
observations with two annotators each, shown next to the model's prediction.

    python tools/build_figure_kernel.py <output_dir>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = json.loads((ROOT / "notebooks" / "filament_pipeline.ipynb").read_text(encoding="utf-8"))
setup = [c for c in src["cells"] if c["cell_type"] == "code"][:3]      # repo + data discovery + imports

FIG = r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from filament_seg.data import load_samples, split_by_observation
from filament_seg.io_utils import decode_mask

segmenter = FilamentSegmenter(WEIGHTS, DEFAULT_CONFIG)
samples = load_samples(TRAIN_JSON, TRAIN_DIR)
_, val = split_by_observation(samples)
by_stem = {}
for s in val:
    by_stem.setdefault(s["stem"], []).append(s)
multi = {k: v for k, v in by_stem.items() if len(v) >= 2}

def ann_mask(ann, h, w):
    rles = mask_utils.frPyObjects(ann["segmentation"], h, w)
    m = mask_utils.decode(mask_utils.merge(rles) if isinstance(rles, list) else rles)
    return m.astype(bool)

def labels(masks):
    lab = np.zeros((2048, 2048), np.int32)
    for i, m in enumerate(masks, 1):
        lab[m & (lab == 0)] = i
    return lab

# choose observations where the two annotators differ visibly: moderate counts, unequal
cand = []
for stem, passes in multi.items():
    n = [len([a for a in p["annotations"] if a.get("segmentation")]) for p in passes[:2]]
    if 4 <= min(n) and max(n) <= 16:
        cand.append((abs(n[0] - n[1]), stem))
cand.sort(reverse=True)
chosen = [s for _, s in cand[:2]]
print("chosen", chosen, [len(multi[s]) for s in chosen])

palette = plt.get_cmap("tab20")
def overlay(gray, masks, color_offset=0):
    rgb = np.repeat(gray[..., None], 3, -1).astype(np.float32) / 255.0
    for i, m in enumerate(masks):
        c = np.array(palette((i + color_offset) % 20)[:3])
        rgb[m] = 0.35 * rgb[m] + 0.65 * c
        edge = m ^ cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        rgb[edge] = c
    return rgb

fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4.1 * len(chosen)))
stats = []
for r, stem in enumerate(chosen):
    passes = multi[stem][:2]
    gray = read_gray(passes[0]["path"])
    ann = [[ann_mask(a, p["height"], p["width"]) for a in p["annotations"] if a.get("segmentation")] for p in passes]
    pred = [decode_mask(x).astype(bool) for x in segmenter.predict_gray(gray)]
    # crop to the solar disk's bounding box for legibility
    ys, xs = np.nonzero(gray > 20)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    titles = [f"Annotator A ({len(ann[0])} filaments)", f"Annotator B ({len(ann[1])} filaments)",
              f"Our model ({len(pred)} filaments)"]
    for c, masks in enumerate([ann[0], ann[1], pred]):
        ax = axes[r, c] if len(chosen) > 1 else axes[c]
        ax.imshow(overlay(gray, masks)[y0:y1, x0:x1]); ax.set_title(titles[c], fontsize=11); ax.axis("off")
    stats.append({"stem": stem, "annotator_a": len(ann[0]), "annotator_b": len(ann[1]), "model": len(pred)})
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_examples.png", dpi=110, bbox_inches="tight")
(OUT_DIR / "fig_examples.json").write_text(json.dumps(stats, indent=1))
print(stats)
'''

cells = [dict(c, outputs=[], execution_count=None) for c in setup]
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": FIG.splitlines(True)})
nb = dict(src, cells=cells)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "figures.ipynb").write_text(json.dumps(nb), encoding="utf-8")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "syehsyoh/filament-solution-figures", "title": "filament-solution-figures", "code_file": "figures.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "dataset_sources": ["syehsyoh/filament-solution"],
    "competition_sources": ["filament-segmentation-2026"], "machine_shape": "NvidiaTeslaT4"}, indent=2), encoding="utf-8")
print("built", out / "figures.ipynb")
