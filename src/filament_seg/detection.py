"""Stage 1: filament box detection with two YOLO11s detectors and weighted box fusion."""
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig

Box = list          # [x1, y1, x2, y2] in native pixels
Scored = tuple      # (confidence, Box)


def load_detectors(weights_dir, cfg: PipelineConfig):
    """Return [(YOLO model, imgsz), ...] in the order given by the config."""
    from ultralytics import YOLO
    return [(YOLO(str(Path(weights_dir) / spec.weights)), spec.imgsz) for spec in cfg.detectors]


def detect(model, gray: np.ndarray, imgsz: int, cfg: PipelineConfig, device) -> list:
    """All boxes above the confidence floor, in the order YOLO returns them."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(source=bgr, conf=cfg.det_conf_floor, iou=cfg.det_nms_iou,
                        imgsz=imgsz, verbose=False, device=device)[0]
    if res.boxes is None or not len(res.boxes):
        return []
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    confs = res.boxes.conf.detach().cpu().numpy()
    return [(float(c), [float(v) for v in b]) for c, b in zip(confs, xyxy)]


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(box_sets: list, iou_thr: float) -> list:
    """Fuse boxes from several detectors.

    Boxes are visited from the most to the least confident. A box joins the
    best-overlapping cluster (IoU > iou_thr) that has no box from the same
    detector yet; otherwise it starts a new cluster. The cluster box is the
    confidence-weighted mean of its members, and its score is the sum of member
    confidences divided by the number of detectors, so a box found by only one
    detector is down-weighted.
    """
    if len(box_sets) == 1:
        return sorted(box_sets[0], key=lambda t: -t[0])
    items = sorted(((c, m, b) for m, s in enumerate(box_sets) for c, b in s), key=lambda t: -t[0])
    clusters = []
    for conf, model_idx, box in items:
        best, best_iou = -1, iou_thr
        for ci, cl in enumerate(clusters):
            if model_idx in cl["models"]:
                continue
            v = box_iou(box, cl["box"])
            if v > best_iou:
                best, best_iou = ci, v
        if best < 0:
            clusters.append({"models": {model_idx}, "confs": [conf], "boxes": [box], "box": box})
        else:
            cl = clusters[best]
            cl["models"].add(model_idx)
            cl["confs"].append(conf)
            cl["boxes"].append(box)
            w = np.array(cl["confs"])[:, None]
            cl["box"] = list((np.array(cl["boxes"]) * w).sum(0) / w.sum())
    return sorted(((sum(cl["confs"]) / len(box_sets), cl["box"]) for cl in clusters), key=lambda t: -t[0])


def fused_boxes(detectors, gray: np.ndarray, cfg: PipelineConfig, device) -> list:
    sets = [detect(model, gray, imgsz, cfg, device) for model, imgsz in detectors]
    return weighted_box_fusion(sets, cfg.fusion_iou)
