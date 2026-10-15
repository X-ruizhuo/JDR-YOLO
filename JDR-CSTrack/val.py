"""Evaluate a trained detector on converted ShipsMOT validation or test images."""

import argparse
import os
from pathlib import Path

DEFAULT_DATA = Path("/home/ShipsMOT_full_v2/shipsmot_full.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    for path in (args.data, args.weights):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.split == "test":
        import yaml
        with args.data.open(encoding="utf-8") as source:
            if "test" not in yaml.safe_load(source):
                parser.error("Dataset YAML has no labeled test split; use --split val")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from ultralytics import YOLO
    YOLO(str(args.weights)).val(
        data=str(args.data), imgsz=args.imgsz, batch=args.batch, split=args.split,
    )


if __name__ == "__main__":
    main()
