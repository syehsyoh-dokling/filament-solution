"""Panoptic Quality exactly as in the organizers' self-evaluation notebook.

A (ground truth, prediction) pair is a true positive when IoU > 0.5. Ground
truth is grouped per annotator pass of an image ("<annotator>-<image>" ids);
predictions are grouped per image. Predictions with no hit are false
positives, ground-truth filaments with no hit are false negatives.

    PQ = sum(IoU of TP pairs) / (|TP| + 0.5 |FP| + 0.5 |FN|)

Working directly on RLEs (pycocotools) gives the same numbers as the
organizers' dense-tensor implementation, without holding 2048 x 2048 masks.
"""
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .io_utils import counts_to_rle

IOU_THRESHOLD = 0.5


def group_ground_truth(gt_df: pd.DataFrame) -> list:
    """[(image_id, [rle, ...]), ...], one entry per annotator-image."""
    key = gt_df["filament_id"].str.split("_", n=1).str[0]
    groups = []
    for annotator_image, g in gt_df.groupby(key, sort=False):
        groups.append((annotator_image.split("-", 1)[1], [counts_to_rle(c) for c in g["segmentation_rle"]]))
    return groups


def group_predictions(pred_df: pd.DataFrame) -> dict:
    key = pred_df["filament_id"].str.split("_", n=1).str[0]
    return {img: [counts_to_rle(c) for c in g["segmentation_rle"]] for img, g in pred_df.groupby(key, sort=False)}


def panoptic_quality(gt_groups: list, pred_by_image: dict) -> dict:
    tp = fp = fn = 0
    iou_sum = 0.0
    tp_ious = []
    for image, gt in gt_groups:
        pred = pred_by_image.get(image, [])
        if not gt:
            fp += len(pred)
            continue
        if not pred:
            fn += len(gt)
            continue
        iou = np.asarray(mask_utils.iou(gt, pred, [0] * len(pred)))   # (n_gt, n_pred)
        hit = iou > IOU_THRESHOLD
        tp += int(hit.sum())
        iou_sum += float(iou[hit].sum())
        tp_ious.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": iou_sum / den if den else 0.0, "tp": tp, "fp": fp, "fn": fn,
            "mean_tp_iou": float(np.mean(tp_ious)) if tp_ious else 0.0,
            "mean_tp_dice": float(np.mean([2 * v / (1 + v) for v in tp_ious])) if tp_ious else 0.0}


def score_dataframes(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    return panoptic_quality(group_ground_truth(gt_df), group_predictions(pred_df))
