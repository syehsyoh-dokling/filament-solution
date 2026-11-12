"""All settings of the final pipeline in one place.

Every value here is the one used to produce the submitted predictions; the
validation PQ of 0.444 (seed-42 held-out split) was measured with exactly these.
"""
from dataclasses import dataclass, field

IMAGE_SIZE = 2048                      # every GONG observation is 2048 x 2048


@dataclass(frozen=True)
class DetectorSpec:
    """One YOLO detector and the resolution it runs at."""
    weights: str                       # file name inside the weights directory
    imgsz: int


@dataclass(frozen=True)
class PipelineConfig:
    # --- stage 1: detection (two YOLO11s detectors, boxes fused) ---
    detectors: tuple = (
        DetectorSpec("detector_yolo11s_1280.pt", 1280),
        DetectorSpec("detector_yolo11s_1920.pt", 1920),
    )
    det_conf_floor: float = 0.05       # keep weak boxes; the final threshold is applied after fusion
    det_nms_iou: float = 0.50
    fusion_iou: float = 0.55           # weighted box fusion: boxes above this IoU are merged

    # --- stage 2: box-seeded mask refinement (ResNet34 U-Net on a square crop) ---
    refiner_weights: str = "refiner_unet_resnet34.pt"
    crop_size: int = 512               # refiner input resolution
    crop_context: float = 1.5          # crop side = context x longer box side
    min_crop: int = 96                 # in native pixels
    seed_pad_ratio: float = 0.60       # padding of the filled box seed (must match training)
    mask_threshold: float = 0.50       # on the flip-TTA averaged probability
    keep_largest_component: bool = True

    # --- stage 3: instance assembly ---
    score_threshold: float = 0.32      # on the fused detector confidence
    min_instance_area: int = 200       # pixels, after removing overlap with stronger instances

    imagenet_mean: tuple = field(default=(0.485, 0.456, 0.406))
    imagenet_std: tuple = field(default=(0.229, 0.224, 0.225))


DEFAULT_CONFIG = PipelineConfig()
