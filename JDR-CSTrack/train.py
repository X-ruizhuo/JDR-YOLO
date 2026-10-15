"""Train the original YOLOv8x example on the converted ShipsMOT dataset."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/home/ShipsMOT_full_v2/shipsmot_full.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=ROOT / "ultralytics/yolov8x.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    for path in (args.data, args.weights):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from ultralytics import YOLO
    YOLO(str(args.weights)).train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, save_period=10, amp=False,
    )


if __name__ == "__main__":
    main()
