"""Score a prediction CSV against a ground-truth CSV with the organizers' PQ.

Both files use the submission format (filament_id, segmentation_rle). Ground-truth
ids must look like '<annotator>-<image>_<k>', as produced by
filament_seg.data.ground_truth_dataframe.

    python scripts/evaluate.py val_ground_truth.csv val_predictions.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.metrics import score_dataframes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ground_truth")
    ap.add_argument("predictions")
    a = ap.parse_args()
    print(json.dumps(score_dataframes(pd.read_csv(a.ground_truth), pd.read_csv(a.predictions)), indent=2))


if __name__ == "__main__":
    main()
