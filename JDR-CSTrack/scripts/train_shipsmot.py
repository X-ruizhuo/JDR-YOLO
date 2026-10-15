"""Train ShipsMOT with the detector settings reported in the ShipsMOT paper."""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8l on ShipsMOT")
    parser.add_argument("--data", type=Path, required=True, help="ShipsMOT dataset YAML file")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="shipsmot_paper_v8l_80")
    parser.add_argument("--resume", type=Path, help="Unfinished run's last.pt checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data}")
    if args.epochs < 1 or args.batch < 1:
        raise SystemExit("--epochs and --batch must be positive")

    # This repository predates PyTorch 2.6's weights_only default.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from ultralytics import YOLO

    if args.resume:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"Resume checkpoint does not exist: {checkpoint}")
        YOLO(str(checkpoint)).train(resume=True)
        return

    weights = root / "yolov8l.pt"

    YOLO(str(weights)).train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=False,
        amp=False,
        optimizer="SGD",
        lr0=0.01,
        warmup_epochs=3.0,
        patience=0,
        save_period=10,
        project=str(root / "runs" / "detect"),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
