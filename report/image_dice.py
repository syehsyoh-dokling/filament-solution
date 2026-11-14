"""Image-level (semantic) Dice of the final predictions, with torchmetrics' DiceScore.

    python report/image_dice.py <val_ground_truth.csv> <val_predictions.csv>

For every annotator pass, all ground-truth filaments are merged into one binary mask and all
predicted instances of that observation into another; DiceScore is computed on the pair. The
mean, a bootstrap interval over observations and the quartiles are printed.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu
from torchmetrics.segmentation import DiceScore

R = lambda c: {"size": [2048, 2048], "counts": c}


def union(counts):
    if not len(counts):
        return np.zeros((2048, 2048), np.uint8)
    return mu.decode(mu.merge([R(c) for c in counts], intersect=False))


def main():
    gt = pd.read_csv(sys.argv[1])
    pred = pd.read_csv(sys.argv[2])
    gt["key"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["img"] = gt["key"].str.split("-", n=1).str[1]
    pred["img"] = pred["filament_id"].str.rsplit("_", n=1).str[0]
    pred_mask = {i: union(g["segmentation_rle"].tolist()) for i, g in pred.groupby("img")}
    metric = DiceScore(num_classes=2, include_background=False, average="micro", input_format="index")
    rows = []
    for (img, key), g in gt.groupby(["img", "key"], sort=False):
        t = union(g["segmentation_rle"].tolist())
        p = pred_mask.get(img, np.zeros_like(t))
        tm = float(metric(torch.from_numpy(p[None].astype(np.int64)), torch.from_numpy(t[None].astype(np.int64))))
        metric.reset()
        inter = int((t & p).sum())
        rows.append({"img": img, "dice": tm, "dice_formula": 2 * inter / max(1, int(t.sum()) + int(p.sum()))})
    d = pd.DataFrame(rows)
    assert np.allclose(d["dice"], d["dice_formula"], atol=1e-6), "torchmetrics and 2|A.B|/(|A|+|B|) disagree"
    by_img = d.groupby("img")["dice"].mean()
    rng = np.random.default_rng(0)
    imgs = by_img.index.to_numpy()
    passes = d.set_index("img")["dice"]
    boots = [passes.loc[rng.choice(imgs, len(imgs))].mean() for _ in range(2000)]
    out = {"annotator_passes": len(d), "observations": int(d["img"].nunique()),
           "mean_dice": float(d["dice"].mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
           "quartiles": d["dice"].quantile([0.25, 0.5, 0.75]).tolist(),
           "min": float(d["dice"].min()), "max": float(d["dice"].max()),
           "share_below_0.5": float((d["dice"] < 0.5).mean())}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
