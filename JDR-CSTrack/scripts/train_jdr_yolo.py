"""Train the JDR-YOLO detector on converted ShipsMOT data."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ultralytics/cfg/models/v8/yolov8l-BasicRFB.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/home/ShipsMOT_full_v2/shipsmot_full.yaml"))
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="shipsmot_full_jdr_yolo")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")
    if args.batch < 1 or args.epochs < 1:
        parser.error("--batch and --epochs must be positive")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, str(ROOT))
    import ultralytics
    if Path(ultralytics.__file__).resolve().parent != ROOT / "ultralytics":
        raise RuntimeError(f"Expected repository Ultralytics, got {ultralytics.__file__}")
    from ultralytics import YOLO

    if args.resume:
        if not args.resume.is_file():
            parser.error(f"Checkpoint not found: {args.resume}")
        YOLO(str(args.resume)).train(resume=True)
        return
    if not args.pretrained.is_file():
        parser.error(f"Pretrained weights not found: {args.pretrained}")
    model = YOLO(str(MODEL))
    model.load(str(args.pretrained))
    modules = list(model.model.modules())
    rfb_count = sum(type(module).__name__ == "BasicRFB" for module in modules)
    csp_modules = [module for module in modules if type(module).__name__ == "BottleneckCSP"]
    rep_count = sum(type(child).__name__ == "RepBlock"
                    for module in csp_modules for child in module.modules())
    print(f"JDR-YOLO modules: BasicRFB={rfb_count}, BottleneckCSP={len(csp_modules)}, "
          f"RepBlock={rep_count}; config={MODEL}", flush=True)
    if rfb_count != 3 or not csp_modules or not rep_count:
        raise RuntimeError(
            "JDR-YOLO modules are incomplete. Sync yolov8l-BasicRFB.yaml and "
            "ultralytics/nn/modules/block.py from the updated workspace; "
            "BasicRFB=3, BottleneckCSP>0, and RepBlock>0 are required."
        )
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        cache=False, amp=False, optimizer="SGD", lr0=0.01,
        warmup_epochs=3.0, patience=0, save_period=10,
        project=str(ROOT / "runs/detect"), name=args.name, exist_ok=False,
    )


if __name__ == "__main__":
    main()
