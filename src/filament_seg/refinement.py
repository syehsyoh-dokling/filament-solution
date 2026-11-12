"""Stage 2: turn a detector box into a filament mask with a box-seeded U-Net refiner.

For each box, a square crop around it is resized to 512 x 512 and fed to a
ResNet34 U-Net with three input channels: the raw crop, its CLAHE-enhanced
version, and a filled (padded) box "seed" that tells the network which filament
in the crop to segment. The same seed construction is used in training.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PipelineConfig


def load_refiner(weights_dir, cfg: PipelineConfig, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    state = torch.load(Path(weights_dir) / cfg.refiner_weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return model.to(device).eval()


def square_crop_bounds(bbox, h: int, w: int, cfg: PipelineConfig):
    """Square window around an [x, y, w, h] box, shifted (not shrunk) to stay inside the image."""
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    side = max(int(math.ceil(max(bw, bh) * cfg.crop_context)), cfg.min_crop)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > w:
        x0 -= x1 - w; x1 = w
    if y1 > h:
        y0 -= y1 - h; y1 = h
    return max(0, x0), max(0, y0), x1, y1


def box_seed(h: int, w: int, bbox, pad_ratio: float) -> np.ndarray:
    """Filled box padded by pad_ratio of its size on every side."""
    x, y, bw, bh = bbox
    px, py = bw * pad_ratio, bh * pad_ratio
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
    seed = np.zeros((h, w), np.uint8)
    seed[y0:y1, x0:x1] = 1
    return seed


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 2:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


@torch.no_grad()
def refine_box(refiner, gray: np.ndarray, bbox, cfg: PipelineConfig, device) -> np.ndarray:
    """Full-image boolean mask for one [x, y, w, h] box.

    The probability map is the mean of the identity, horizontal-flip and
    vertical-flip predictions; it is upsampled to the crop's native size with
    linear interpolation before thresholding, and only the largest connected
    component is kept.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = square_crop_bounds(bbox, h, w, cfg)
    size = (cfg.crop_size, cfg.crop_size)
    img_c = cv2.resize(gray[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(img_c)
    seed_c = cv2.resize(box_seed(h, w, bbox, cfg.seed_pad_ratio)[y0:y1, x0:x1], size,
                        interpolation=cv2.INTER_NEAREST)
    mean = np.array(cfg.imagenet_mean, np.float32)[:, None, None]
    std = np.array(cfg.imagenet_std, np.float32)[:, None, None]
    x = (np.stack([img_c, clahe, seed_c * 255], 0).astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x[None]).to(device)
    prob = (torch.sigmoid(refiner(x))
            + torch.sigmoid(refiner(x.flip(-1))).flip(-1)
            + torch.sigmoid(refiner(x.flip(-2))).flip(-2)) / 3.0
    prob = prob[0, 0].float().cpu().numpy().astype(np.float32)
    prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= cfg.mask_threshold).astype(np.uint8)
    if cfg.keep_largest_component:
        mask = _largest_component(mask)
    full = np.zeros((h, w), bool)
    full[y0:y1, x0:x1] = mask.astype(bool)
    return full
