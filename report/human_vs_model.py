"""Model vs human annotators on the same held-out observations (those with >= 2 annotator passes).

Human: every annotator pass is scored, as if it were a prediction, against each OTHER pass of
the same observation. Model: scored against each pass, weighted (n-1) times so that both sides
are compared with exactly the same reference passes. 95% intervals: paired bootstrap over
observations.

    python report/human_vs_model.py <val_ground_truth.csv> <val_predictions.csv>
"""
import json
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mu

R = lambda c: {"size": [2048, 2048], "counts": c}


def stats(pred, ref):
    """(tp, fp, fn, iou_sum) of `pred` scored against reference masks `ref` (organizers' PQ)."""
    if not ref:
        return 0, len(pred), 0, 0.0
    if not pred:
        return 0, 0, len(ref), 0.0
    iou = np.asarray(mu.iou(ref, pred, [0] * len(pred)))
    hit = iou > 0.5
    return int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum()), float(iou[hit].sum())


def main():
    gt = pd.read_csv(sys.argv[1])
    gt["key"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["img"] = gt["key"].str.split("-", n=1).str[1]
    pred = pd.read_csv(sys.argv[2])
    pred["img"] = pred["filament_id"].str.rsplit("_", n=1).str[0]
    P = {i: [R(c) for c in g.segmentation_rle] for i, g in pred.groupby("img")}

    rows = []
    for img, gi in gt.groupby("img"):
        groups = [[R(c) for c in g.segmentation_rle] for _, g in gi.groupby("key")]
        if len(groups) < 2:
            continue
        h = np.zeros(4); m = np.zeros(4)
        for a, ga in enumerate(groups):
            for b, gb in enumerate(groups):
                if a != b:
                    h += stats(ga, gb)                 # annotator a judged by annotator b
                    m += stats(P.get(img, []), gb)     # model judged by the same reference b
        rows.append({"img": img, "n_passes": len(groups), "h": h, "m": m})

    def pq(v):
        d = v[0] + 0.5 * v[1] + 0.5 * v[2]
        return v[3] / d if d else 0.0

    H = np.array([r["h"] for r in rows]); M = np.array([r["m"] for r in rows])
    rng = np.random.default_rng(0)
    diffs, hs, ms = [], [], []
    for _ in range(5000):
        s = rng.integers(0, len(rows), len(rows))
        ph, pm = pq(H[s].sum(0)), pq(M[s].sum(0))
        hs.append(ph); ms.append(pm); diffs.append(pm - ph)
    th, tm = H.sum(0), M.sum(0)
    out = {
        "observations": len(rows),
        "passes_per_observation": dict(pd.Series([r["n_passes"] for r in rows]).value_counts().sort_index()),
        "human": {"pq": pq(th), "ci95": np.percentile(hs, [2.5, 97.5]).tolist(), "tp": int(th[0]), "fp": int(th[1]),
                  "fn": int(th[2]), "mean_tp_iou": th[3] / th[0]},
        "model": {"pq": pq(tm), "ci95": np.percentile(ms, [2.5, 97.5]).tolist(), "tp": int(tm[0]), "fp": int(tm[1]),
                  "fn": int(tm[2]), "mean_tp_iou": tm[3] / tm[0]},
        "model_minus_human": {"pq": pq(tm) - pq(th), "ci95": np.percentile(diffs, [2.5, 97.5]).tolist(),
                              "share_of_bootstrap_with_model_better": float(np.mean(np.array(diffs) > 0))},
        "model_better_on_observations": int(sum(pq(r["m"]) > pq(r["h"]) for r in rows)),
    }
    print(json.dumps(out, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))


if __name__ == "__main__":
    main()
