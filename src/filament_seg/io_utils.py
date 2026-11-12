"""Image loading and COCO run-length encoding helpers."""
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .config import IMAGE_SIZE


def read_gray(path) -> np.ndarray:
    """Read an H-alpha observation as a 2048 x 2048 uint8 grayscale array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def encode_mask(mask: np.ndarray) -> dict:
    """Binary mask -> compressed COCO RLE dict (counts as bytes)."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def decode_mask(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle)


def rle_counts(rle: dict) -> str:
    """The string the competition expects in `segmentation_rle`."""
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def counts_to_rle(counts: str) -> dict:
    return {"size": [IMAGE_SIZE, IMAGE_SIZE], "counts": counts}


def list_test_images(test_dir) -> list:
    test_dir = Path(test_dir)
    paths = sorted(test_dir.glob("*.jpeg")) + sorted(test_dir.glob("*.jpg"))
    seen, unique = set(), []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique
