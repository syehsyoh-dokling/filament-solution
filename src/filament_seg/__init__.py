"""Solar filament instance segmentation: YOLO11s box detection (two resolutions, fused)
followed by a box-seeded ResNet34 U-Net mask refiner."""
from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import FilamentSegmenter, assemble_instances, predict_directory

__all__ = ["DEFAULT_CONFIG", "PipelineConfig", "FilamentSegmenter", "assemble_instances", "predict_directory"]
