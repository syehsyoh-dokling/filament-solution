"""Write a competition submission for a directory of H-alpha test images.

    python scripts/predict.py --test-dir data/test_images --out submission.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg import DEFAULT_CONFIG, FilamentSegmenter, predict_directory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    sub = predict_directory(FilamentSegmenter(a.weights, DEFAULT_CONFIG), a.test_dir)
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sub)} instances on "
          f"{sub['filament_id'].str.rsplit('_', n=1).str[0].nunique()} observations")


if __name__ == "__main__":
    main()
