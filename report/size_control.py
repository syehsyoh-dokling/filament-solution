"""Is low recall on single-annotator filaments explained by their smaller size?

Recall by agreement level within area tertiles, on held-out observations with >= 2
annotator passes, plus a logistic regression of `found` on log-area and agreement.

    python report/size_control.py <val_ground_truth.csv> <val_predictions.csv>
"""
import json
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mu

R = lambda c: {"size": [2048, 2048], "counts": c}


def main():
    gt = pd.read_csv(sys.argv[1])
    gt["key"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["img"] = gt["key"].str.split("-", n=1).str[1]
    pred = pd.read_csv(sys.argv[2])
    pred["img"] = pred["filament_id"].str.rsplit("_", n=1).str[0]
    P = {i: [R(c) for c in g.segmentation_rle] for i, g in pred.groupby("img")}
    rows = []
    for img, gi in gt.groupby("img"):
        groups = {k: [R(c) for c in g.segmentation_rle] for k, g in gi.groupby("key")}
        if len(groups) < 2:
            continue
        p = P.get(img, [])
        for k, g in groups.items():
            found = (np.asarray(mu.iou(g, p, [0] * len(p))) > 0.5).any(1) if p else np.zeros(len(g), bool)
            for i, r in enumerate(g):
                agree = sum(bool((np.asarray(mu.iou([r], gg, [0] * len(gg))) > 0.5).any())
                            for kk, gg in groups.items() if kk != k)
                share = agree / (len(groups) - 1)          # fraction of the other annotators who marked it
                rows.append({"found": int(found[i]), "area": int(mu.area(r)), "share": share,
                             "level": "all" if share == 1 else ("none" if share == 0 else "some")})
    d = pd.DataFrame(rows)
    d["size"] = pd.qcut(d["area"], 3, labels=["small", "medium", "large"])
    tab = d.pivot_table(index="size", columns="level", values="found", aggfunc=["mean", "size"], observed=False)
    print("recall (mean) and counts (size) by area tertile and agreement level")
    print(tab.round(2).to_string())
    print("tertile edges (px):", np.percentile(d["area"], [33.33, 66.67]).round(0).tolist())

    # logistic regression: found ~ log(area) + share  (Newton-Raphson, no extra dependency)
    X = np.column_stack([np.ones(len(d)), np.log(d["area"]), d["share"]])
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    y = d["found"].to_numpy(float)
    w = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None])
        w += np.linalg.solve(H, X.T @ (y - p))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    out = {name: {"coef": float(c), "se": float(s), "z": float(c / s), "odds_ratio": float(np.exp(c))}
           for name, c, s in zip(["intercept", "log_area_std", "agreement_share"], w, se)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
