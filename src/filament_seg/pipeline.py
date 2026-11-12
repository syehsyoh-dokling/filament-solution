"""Stage 3 and the end-to-end predictor: boxes -> refined masks -> disjoint instances."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_CONFIG, PipelineConfig
from .detection import fused_boxes, load_detectors
from .io_utils import decode_mask, encode_mask, list_test_images, read_gray, rle_counts
from .refinement import load_refiner, refine_box


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def assemble_instances(scored_rles, score_threshold: float, min_area: int) -> list:
    """Greedy, strictly non-overlapping instances.

    Candidates are visited from the highest to the lowest fused confidence;
    pixels already given to a stronger instance are removed, and what remains
    is kept only if it still covers at least `min_area` pixels.
    """
    occupied = None
    out = []
    for conf, rle in scored_rles:
        if conf < score_threshold:
            break
        m = decode_mask(rle).astype(bool)
        if occupied is None:
            occupied = np.zeros(m.shape, bool)
        m &= ~occupied
        if int(m.sum()) < max(min_area, 1):
            continue
        occupied |= m
        out.append(encode_mask(m))
    return out


class FilamentSegmenter:
    """Loads the three models once and predicts filament instances per image."""

    def __init__(self, weights_dir, cfg: PipelineConfig = DEFAULT_CONFIG, device=None):
        self.cfg = cfg
        cuda = torch.cuda.is_available()
        self.torch_device = torch.device(device or ("cuda" if cuda else "cpu"))
        self.yolo_device = 0 if self.torch_device.type == "cuda" else "cpu"
        self.detectors = load_detectors(weights_dir, cfg)
        self.refiner = load_refiner(weights_dir, cfg, self.torch_device)

    def boxes(self, gray):
        return fused_boxes(self.detectors, gray, self.cfg, self.yolo_device)

    def refine(self, gray, scored_boxes, floor: float) -> list:
        out = []
        for conf, box in scored_boxes:
            if conf < floor:
                break
            m = refine_box(self.refiner, gray, xyxy_to_xywh(box), self.cfg, self.torch_device)
            out.append((conf, encode_mask(m)))
        return out

    def predict_gray(self, gray) -> list:
        """Disjoint instance RLEs for one image, strongest first."""
        cfg = self.cfg
        boxes = self.boxes(gray)
        inst = assemble_instances(self.refine(gray, boxes, cfg.score_threshold),
                                  cfg.score_threshold, cfg.min_instance_area)
        if not inst and boxes:
            # nothing above the threshold: keep the single most confident candidate
            inst = assemble_instances(self.refine(gray, boxes[:1], 0.0), 0.0, 1)
        return inst

    def predict_path(self, path) -> list:
        return self.predict_gray(read_gray(path))


def predict_directory(segmenter: FilamentSegmenter, test_dir, progress=True) -> pd.DataFrame:
    """Submission dataframe (filament_id, segmentation_rle) for every image in test_dir."""
    paths = list_test_images(test_dir)
    if progress:
        from tqdm.auto import tqdm
        paths = tqdm(paths, desc="test images")
    rows = []
    for path in paths:
        for i, rle in enumerate(segmenter.predict_path(path), start=1):
            rows.append({"filament_id": f"{Path(path).stem}_{i}", "segmentation_rle": rle_counts(rle)})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
