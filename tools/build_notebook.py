"""Generate notebooks/filament_pipeline.ipynb (run: python tools/build_notebook.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md(r"""
# Solar filament instance segmentation — end-to-end pipeline

This notebook reproduces our submitted predictions for the
**Solar Filament Segmentation Challenge 2026** from the trained weights stored in
`weights/`; no retraining is needed.

**Pipeline** (details in the cells below and in `src/filament_seg/`):

1. **Detection** — two YOLO11s detectors (input 1280 px and 1920 px) propose
   filament boxes; their boxes are merged by weighted box fusion.
2. **Refinement** — every box is turned into a mask by a ResNet34 U-Net that sees a
   square crop, its CLAHE-enhanced copy and a filled box "seed" marking which
   filament to segment (flip test-time augmentation, linear upsampling, largest
   connected component).
3. **Instance assembly** — masks are taken from the most to the least confident,
   pixels already claimed are removed (strictly non-overlapping output), and
   small remnants are dropped.

**Measured results** (held-out 15 % of training observations, organizers' PQ):
PQ 0.444, 750 TP / 326 FP / 508 FN. Public leaderboard of the same predictions:
see `README.md`.
""")

code(r"""
# --- locate the repository (works locally and as a Kaggle dataset) and install deps if missing ---
import glob, importlib.util, os, subprocess, sys
from pathlib import Path

def find_repo():
    for start in [Path.cwd(), Path.cwd().parent, Path("/kaggle/input")]:
        hits = sorted(glob.glob(str(start / "**" / "src" / "filament_seg" / "__init__.py"), recursive=True))
        if hits:
            return Path(hits[0]).parents[2]
    raise FileNotFoundError("repository root (src/filament_seg) not found")

def find_near(repo, pattern):
    # the file next to the repo root; on Kaggle zipped folders may be extracted one level deeper
    for base in (repo, repo.parent, Path("/kaggle/input")):
        hits = sorted(glob.glob(str(base / "**" / pattern), recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(pattern)

REPO = find_repo()
sys.path.insert(0, str(REPO / "src"))
WEIGHTS = find_near(REPO, "detector_yolo11s_1280.pt").parent
REQUIREMENTS = find_near(REPO, "requirements.txt")
print("repository:", REPO, "| weights:", WEIGHTS)

if importlib.util.find_spec("ultralytics") is None or importlib.util.find_spec("segmentation_models_pytorch") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)])
""")

code(r"""
# --- locate the competition data ---
def find_one(pattern, want_dir):
    roots = [Path("/kaggle/input"), REPO / "data", Path.cwd() / "data"]
    for root in roots:
        for p in sorted(glob.glob(str(root / "**" / pattern), recursive=True)):
            if Path(p).is_dir() == want_dir:
                return Path(p)
    return None

TEST_DIR = find_one("test_images", True)
TRAIN_DIR = find_one("train_images", True)
TRAIN_JSON = find_one("*Annotations*train*.json", False) or find_one("*train*.json", False)
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("test images :", TEST_DIR)
print("train images:", TRAIN_DIR)
print("train json  :", TRAIN_JSON)
print("outputs     :", OUT_DIR)
assert TEST_DIR is not None, "put the competition data under ./data or attach it on Kaggle"
""")

code(r"""
import json, time
import numpy as np, pandas as pd, torch, cv2, ultralytics, segmentation_models_pytorch as smp, pycocotools
from filament_seg import DEFAULT_CONFIG, FilamentSegmenter, predict_directory
from filament_seg.io_utils import list_test_images, read_gray, rle_counts

versions = {"python": sys.version.split()[0], "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "segmentation_models_pytorch": smp.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "pycocotools": getattr(pycocotools, "__version__", "n/a"),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(json.dumps(versions, indent=1))
print(DEFAULT_CONFIG)
""")

md(r"""
## Load the three models
`weights/detector_yolo11s_1280.pt`, `weights/detector_yolo11s_1920.pt` and
`weights/refiner_unet_resnet34.pt` were trained by us on the competition training
annotations only (see `training/` for the exact training notebooks).
""")

code(r"""
segmenter = FilamentSegmenter(WEIGHTS, DEFAULT_CONFIG)
print("detectors:", [imgsz for _, imgsz in segmenter.detectors], "| refiner on", segmenter.torch_device)
""")

md("## Qualitative example on one test observation")

code(r"""
import matplotlib.pyplot as plt
from filament_seg.io_utils import decode_mask

test_paths = list_test_images(TEST_DIR)
example = test_paths[0]
gray = read_gray(example)
instances = segmenter.predict_gray(gray)
overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
rng = np.random.default_rng(0)
for rle in instances:
    m = decode_mask(rle).astype(bool)
    overlay[m] = (0.45 * overlay[m] + 0.55 * rng.integers(60, 255, 3)).astype(np.uint8)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(gray, cmap="gray"); ax[0].set_title(example.stem)
ax[1].imshow(overlay); ax[1].set_title(f"{len(instances)} predicted filaments")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "example_prediction.png", dpi=80); plt.show()
""")

md("## Predict every test observation and write `submission.csv`")

code(r"""
t0 = time.time()
sub = predict_directory(segmenter, TEST_DIR)
sub_path = OUT_DIR / "submission.csv"
sub.to_csv(sub_path, index=False)
runtime = time.time() - t0

stems = sub["filament_id"].str.rsplit("_", n=1).str[0]
assert list(sub.columns) == ["filament_id", "segmentation_rle"]
assert not sub["filament_id"].duplicated().any() and not sub.isnull().any().any()
print(f"{len(sub)} instances on {stems.nunique()}/{len(test_paths)} observations in {runtime/60:.1f} min -> {sub_path}")
""")

md(r"""
## Validation on held-out training observations (optional)
Re-creates the observation-level split the models were trained with (seed 42,
15 % held out) and scores the same pipeline with the organizers' PQ definition.
Expected: PQ ≈ 0.444 (750 TP, 326 FP, 508 FN).
""")

code(r"""
RUN_VALIDATION = TRAIN_JSON is not None and TRAIN_DIR is not None
summary = {"versions": versions, "test": {"rows": len(sub), "observations": int(stems.nunique()),
                                           "runtime_s": round(runtime, 1)}}
if RUN_VALIDATION:
    from filament_seg.data import ground_truth_dataframe, load_samples, split_by_observation
    from filament_seg.metrics import score_dataframes
    samples = load_samples(TRAIN_JSON, TRAIN_DIR)
    train_s, val_s = split_by_observation(samples)
    val_paths = {}
    for s in val_s:
        val_paths.setdefault(s["stem"], s["path"])
    rows = []
    for stem, path in val_paths.items():
        for i, rle in enumerate(segmenter.predict_path(path), start=1):
            rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": rle_counts(rle)})
    val_pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    val_gt = ground_truth_dataframe(val_s)
    val_pred.to_csv(OUT_DIR / "val_predictions.csv", index=False)
    val_gt.to_csv(OUT_DIR / "val_ground_truth.csv", index=False)
    score = score_dataframes(val_gt, val_pred)
    summary["validation"] = {"observations": len(val_paths), "annotator_passes": len(val_s), **score}
    print(json.dumps(summary["validation"], indent=1))
(OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "filament_pipeline.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
