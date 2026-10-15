"""Track converted ShipsMOT image sequences and write MOT-format result files."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sequence_images(directory):
    images = [path for path in directory.iterdir()
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"No images found in {directory}")
    try:
        return sorted(images, key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"Image filenames must contain numeric frame IDs: {directory}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path,
                        default=Path("/home/ShipsMOT_full_v2/images/test"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tracker", type=Path,
                        default=ROOT / "ultralytics/cfg/trackers/cstrack.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/track/shipsmot_full")
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f"Sequence directory not found: {args.images}")
    for path in (args.weights, args.tracker):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    sequences = sorted((path for path in args.images.iterdir() if path.is_dir()),
                       key=lambda path: int(path.name))
    if not sequences:
        parser.error(f"No sequence subdirectories found in {args.images}")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import cv2
    from ultralytics import YOLO
    args.output.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        images = sequence_images(sequence)
        model = YOLO(str(args.weights))  # New tracker state for every sequence.
        output = args.output / f"{sequence.name}.txt"
        with output.open("w", encoding="utf-8") as target:
            for frame_number, image_path in enumerate(images, start=1):
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise ValueError(f"Cannot read image: {image_path}")
                result = model.track(frame, persist=True, tracker=str(args.tracker),
                                     show=False, verbose=False)[0]
                if result.boxes.id is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                for (left, top, right, bottom), track_id in zip(boxes, ids):
                    target.write(f"{frame_number},{track_id},{left:.2f},{top:.2f},"
                                 f"{right-left:.2f},{bottom-top:.2f},1,0,1\n")
        print(f"{sequence.name}: {len(images)} frames -> {output}", flush=True)


if __name__ == "__main__":
    main()
