"""Every number and figure in the report, computed from validation CSVs.

    python report/analysis.py <final_val_gt.csv> <final_val_pred.csv> <out_dir> [<name>=<pred.csv> ...]

The extra predictions (other pipeline variants on the same split) are compared
with the final one by a paired bootstrap over observations.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mu

R = lambda c: {"size": [2048, 2048], "counts": c}
RNG = np.random.default_rng(0)
N_BOOT = 2000


def load(gt_path, pred_path):
    gt = pd.read_csv(gt_path)
    gt["key"] = gt["filament_id"].str.split("_", n=1).str[0]
    gt["img"] = gt["key"].str.split("-", n=1).str[1]
    pred = pd.read_csv(pred_path)
    pred["img"] = pred["filament_id"].str.rsplit("_", n=1).str[0]
    return gt, pred


def per_image(gt, pred):
    """Per observation: tp, fp, fn, iou_sum (summed over its annotator passes) + TP IoUs + m:n counts."""
    P = {i: [R(c) for c in g.segmentation_rle] for i, g in pred.groupby("img")}
    rows, tp_ious, gt_deg, pred_deg = [], [], [], []
    for img, gi in gt.groupby("img"):
        p = P.get(img, [])
        t = f = n = 0
        s = 0.0
        for _, g in gi.groupby("key"):
            gr = [R(c) for c in g.segmentation_rle]
            if not p:
                n += len(gr); gt_deg += [0] * len(gr); continue
            iou = np.asarray(mu.iou(gr, p, [0] * len(p)))
            hit = iou > 0.5
            t += int(hit.sum()); s += float(iou[hit].sum())
            f += int((hit.sum(0) == 0).sum()); n += int((hit.sum(1) == 0).sum())
            tp_ious += iou[hit].tolist()
            gt_deg += (iou > 0).sum(1).tolist(); pred_deg += (iou > 0).sum(0).tolist()
        rows.append({"img": img, "tp": t, "fp": f, "fn": n, "iou_sum": s})
    return pd.DataFrame(rows).set_index("img"), np.array(tp_ious), np.array(gt_deg), np.array(pred_deg)


def pq_of(df):
    d = df.tp.sum() + 0.5 * df.fp.sum() + 0.5 * df.fn.sum()
    return df.iou_sum.sum() / d if d else 0.0


def boot(df, other=None):
    idx = df.index.to_numpy()
    vals = []
    for _ in range(N_BOOT):
        s = RNG.choice(idx, len(idx), replace=True)
        v = pq_of(df.loc[s])
        if other is not None:
            v -= pq_of(other.loc[s])
        vals.append(v)
    return np.percentile(vals, [2.5, 97.5]).tolist()


def agreement_strata(gt, pred):
    P = {i: [R(c) for c in g.segmentation_rle] for i, g in pred.groupby("img")}
    rows, fp_confirmed = [], []
    for img, gi in gt.groupby("img"):
        groups = {k: [R(c) for c in g.segmentation_rle] for k, g in gi.groupby("key")}
        p = P.get(img, [])
        for k, g in groups.items():
            found = (np.asarray(mu.iou(g, p, [0] * len(p))) > 0.5).any(1) if p else np.zeros(len(g), bool)
            for i, r in enumerate(g):
                if len(groups) == 1:
                    cls = "single-annotator observations"
                else:
                    agree = sum(bool((np.asarray(mu.iou([r], gg, [0] * len(gg))) > 0.5).any())
                                for kk, gg in groups.items() if kk != k)
                    cls = ("marked by all annotators" if agree == len(groups) - 1 else
                           "marked by some" if agree else "marked by one annotator only")
                rows.append({"cls": cls, "found": bool(found[i]), "area": int(mu.area(r))})
            if len(groups) > 1 and p:
                others = [r for kk, gg in groups.items() if kk != k for r in gg]
                hit_k = (np.asarray(mu.iou(g, p, [0] * len(p))) > 0.5).any(0)
                for j in np.where(~hit_k)[0]:
                    fp_confirmed.append(bool((np.asarray(mu.iou([p[j]], others, [0] * len(others))) > 0.5).any()))
    d = pd.DataFrame(rows)
    out = d.groupby("cls").agg(n=("found", "size"), recall=("found", "mean"), median_area=("area", "median"))
    # 95% binomial (Wilson) intervals
    z = 1.96
    p, n = out.recall, out.n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    out["ci_low"], out["ci_high"] = centre - half, centre + half
    return out, {"fp_vs_one_annotator": len(fp_confirmed), "confirmed_by_other": int(sum(fp_confirmed))}


def main():
    gt_path, pred_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    gt, pred = load(gt_path, pred_path)
    df, tp_ious, gt_deg, pred_deg = per_image(gt, pred)
    res = {"final": {"pq": pq_of(df), "ci95": boot(df), "tp": int(df.tp.sum()), "fp": int(df.fp.sum()),
                     "fn": int(df.fn.sum()), "mean_tp_iou": float(tp_ious.mean()),
                     "mean_tp_dice": float(np.mean(2 * tp_ious / (1 + tp_ious))),
                     "observations": len(df)}}
    res["m2n"] = {"gt_missed(1:0)": int((gt_deg == 0).sum()), "gt_one_pred(1:1)": int((gt_deg == 1).sum()),
                  "gt_split(1:n>1)": int((gt_deg > 1).sum()), "pred_unmatched(0:1)": int((pred_deg == 0).sum()),
                  "pred_one_gt(1:1)": int((pred_deg == 1).sum()), "pred_merged(n>1:1)": int((pred_deg > 1).sum())}
    for arg in sys.argv[4:]:
        name, path = arg.split("=", 1)
        _, other_pred = load(gt_path, path)
        odf, *_ = per_image(gt, other_pred)
        odf = odf.reindex(df.index).fillna(0)
        res[name] = {"pq": pq_of(odf), "ci95": boot(odf), "delta_final_minus_this": pq_of(df) - pq_of(odf),
                     "delta_ci95": boot(df, odf)}
    strata, fp = agreement_strata(gt, pred)
    res["agreement_strata"] = strata.reset_index().to_dict(orient="records")
    res["false_positives"] = fp
    (out / "report_numbers.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))

    # Figure: (a) recall by annotator agreement, (b) IoU / Dice of matched pairs
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    order = ["marked by all annotators", "marked by some", "single-annotator observations", "marked by one annotator only"]
    s = strata.reindex(order)
    y = np.arange(len(order))[::-1]
    ax[0].barh(y, s.recall, xerr=[s.recall - s.ci_low, s.ci_high - s.recall], color="#3b6ea5", capsize=2)
    ax[0].set_yticks(y, [f"{o}\n(n={int(n)})" for o, n in zip(order, s.n)])
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("model recall (IoU > 0.5)")
    for yy, v in zip(y, s.recall):
        ax[0].text(0.02, yy, f"{v:.0%}", va="center", color="white", fontweight="bold")
    ax[0].set_title("(a) misses concentrate where annotators disagree", fontsize=8)
    bins = np.linspace(0.5, 1.0, 26)
    ax[1].hist(tp_ious, bins=bins, color="#3b6ea5", alpha=0.85, label=f"IoU (mean {tp_ious.mean():.2f})")
    dice = 2 * tp_ious / (1 + tp_ious)
    ax[1].hist(dice, bins=bins, color="#e08a2c", alpha=0.6, label=f"Dice (mean {dice.mean():.2f})")
    ax[1].set_xlabel("overlap of matched pairs"); ax[1].set_ylabel("pairs"); ax[1].legend(frameon=False)
    ax[1].set_title("(b) matched-pair IoU / Dice distribution", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_results.pdf", bbox_inches="tight")
    plt.savefig(out / "fig_results.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
