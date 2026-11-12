"""Training annotations and the held-out validation split used for every number we report.

The split is by physical observation (file stem), so the several annotator
passes of one image are always on the same side. It reproduces the split the
models were trained with: seed 42, 15 % of observations held out.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pycocotools import mask as mask_utils

from .io_utils import rle_counts

SPLIT_SEED = 42
VAL_FRACTION = 0.15


def load_samples(train_json, train_images_dir) -> list:
    """One entry per COCO image record (= one annotator pass of one observation)."""
    with open(train_json) as f:
        coco = json.load(f)
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    samples = []
    for im in coco["images"]:
        path = Path(train_images_dir) / im["file_name"]
        if not path.exists():
            alt = path.with_suffix(".jpg") if path.suffix.lower() == ".jpeg" else path.with_suffix(".jpeg")
            if not alt.exists():
                continue
            path = alt
        samples.append({"image_id": im["id"], "path": path, "stem": path.stem,
                        "height": im.get("height", 2048), "width": im.get("width", 2048),
                        "annotations": anns_by_image[im["id"]]})
    return samples


def split_by_observation(samples: list, seed: int = SPLIT_SEED, val_fraction: float = VAL_FRACTION):
    stems = sorted({s["stem"] for s in samples})
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(stems)
    random.setstate(rng_state)
    n_val = max(1, int(len(stems) * val_fraction))
    val = set(stems[:n_val])
    return [s for s in samples if s["stem"] not in val], [s for s in samples if s["stem"] in val]


def ground_truth_dataframe(samples: list) -> pd.DataFrame:
    """Competition-format ground truth: filament_id = '<annotator pass id>-<stem>_<k>'.

    The organizers' evaluator groups ground truth by the text before '_' and
    takes the image id after the first '-'. MAGFiLO image record ids already
    have the form '<annotator>-<stem>', so the prefix is taken from them; this
    gives one group per annotator pass, as in the test set.
    """
    rows = []
    for s in samples:
        rec = str(s["image_id"])
        annotator = rec.split("-", 1)[0] if "-" in rec else rec
        for k, ann in enumerate(s["annotations"], start=1):
            if not ann.get("segmentation"):
                continue
            rles = mask_utils.frPyObjects(ann["segmentation"], s["height"], s["width"])
            rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
            rows.append({"filament_id": f"{annotator}-{s['stem']}_{k}", "segmentation_rle": rle_counts(rle)})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
